"""
KnowledgeBase Module

This module provides functionality for creating and managing a searchable knowledge base
using FAISS for fast similarity search and ChromaDB for persistence.
"""

import json
import logging
import time
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import faiss
import chromadb
from chromadb.config import Settings
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        import tomli_w as tomllib
from embedding_manager import EmbeddingManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_config():
    """Load configuration from pyproject.toml."""
    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
                return config.get("tool", {}).get("knowledgebase", {})
    except Exception as e:
        logger.warning(f"Failed to load config from pyproject.toml: {e}")
    return {}


class SyncManager:
    """
    Synchronization manager for FAISS and ChromaDB consistency.

    Features:
    - Thread-safe operations
    - Automatic index reconstruction
    - Batch synchronization
    - Error handling and recovery
    """

    def __init__(self, knowledge_base):
        """
        Initialize synchronization manager.

        Args:
            knowledge_base: KnowledgeBase instance
        """
        self.knowledge_base = knowledge_base
        self.lock = threading.Lock()
        self.operation_queue = Queue()
        self.is_processing = False

    def add_vector(
        self,
        vector: np.ndarray,
        text: str,
        metadata: Dict[str, Any] = None,
        cluster_id: Optional[int] = None,
    ) -> str:
        """
        Add a vector with thread-safe synchronization.

        Args:
            vector: Vector to add
            text: Text content
            metadata: Optional metadata
            cluster_id: Optional cluster ID

        Returns:
            Entry ID
        """
        with self.lock:
            import time

            entry_id = f"vec_{int(time.time() * 1000000)}"

            # Create entry
            entry = KnowledgeBaseEntry(
                id=entry_id,
                text=text,
                embedding=vector,
                metadata=metadata or {},
                cluster_id=cluster_id,
                timestamp=time.time(),
            )

            # Add to ChromaDB
            self.knowledge_base.chroma_db.add_entry(entry)

            # Add to local entries
            self.knowledge_base.entries[entry_id] = entry

            # Rebuild FAISS index if needed
            if not self.knowledge_base.is_index_built:
                self.knowledge_base._build_index()
            else:
                # Add to existing FAISS index
                self.knowledge_base.faiss_index.add_embeddings(vector.reshape(1, -1))

            return entry_id

    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector with thread-safe synchronization.

        Args:
            vector_id: ID of vector to delete

        Returns:
            Success status
        """
        with self.lock:
            if vector_id not in self.knowledge_base.entries:
                return False

            # Delete from ChromaDB
            self.knowledge_base.chroma_db.delete_entry(vector_id)

            # Delete from local entries
            del self.knowledge_base.entries[vector_id]

            # Rebuild FAISS index
            self.knowledge_base._build_index()

            return True

    def rebuild_faiss_index(self):
        """Rebuild FAISS index from all entries."""
        with self.lock:
            self.knowledge_base._build_index()

    def get_sync_status(self) -> Dict[str, Any]:
        """
        Get synchronization status.

        Returns:
            Dictionary with sync status
        """
        return {
            "is_index_built": self.knowledge_base.is_index_built,
            "total_entries": len(self.knowledge_base.entries),
            "lock_acquired": self.lock.locked(),
            "queue_size": self.operation_queue.qsize(),
        }


@dataclass
class KnowledgeBaseEntry:
    """Data class representing a single entry in the knowledge base."""

    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    cluster_id: Optional[int] = None
    summary: Optional[str] = None
    explanation: Optional[str] = None
    key_points: Optional[List[str]] = None
    timestamp: float = 0.0


@dataclass
class SearchResult:
    """Data class representing search result."""

    entry: KnowledgeBaseEntry
    similarity_score: float
    rank: int


class FAISSIndex:
    """
    FAISS index manager for fast similarity search.

    Features:
    - Automatic GPU/CPU detection and usage
    - Index persistence and loading
    - Batch operations
    - Product Quantization (PQ) for memory efficiency
    - Inverted File System (IVF) for faster search
    """

    def __init__(self, dimension: int, use_gpu: bool = False, index_type: str = "flat"):
        """
        Initialize FAISS index.

        Args:
            dimension: Dimension of embeddings
            use_gpu: Whether to use GPU if available
            index_type: Type of FAISS index ('flat', 'ivf', 'pq')
        """
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.index_type = index_type
        self.index = None
        self.is_trained = False
        self.nlist = 100  # Number of clusters for IVF
        self.m = 8  # Number of subquantizers for PQ

        # Detect GPU availability
        gpu_available = self._check_gpu_availability()

        if use_gpu and gpu_available:
            logger.info("GPU available, using GPU for FAISS operations")
            self._create_gpu_index(index_type)
        else:
            if use_gpu:
                logger.warning("GPU not available, falling back to CPU")
            else:
                logger.info("Using CPU for FAISS operations")
            self._create_cpu_index(index_type)

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for FAISS operations."""
        try:
            if faiss.get_num_gpus() > 0:
                return True
            return False
        except Exception as e:
            logger.warning(f"Could not check GPU availability: {e}")
            return False

    def _create_gpu_index(self, index_type: str):
        """Create GPU-based FAISS index."""
        try:
            if index_type == "flat":
                self.index = faiss.GpuIndexFlatL2(self.dimension)
            elif index_type == "ivf":
                quantizer = faiss.GpuIndexFlatL2(self.dimension)
                self.index = faiss.GpuIndexIVFFlat(
                    quantizer, self.dimension, self.nlist
                )
            elif index_type == "pq":
                self.index = faiss.GpuIndexIVFPQ(
                    faiss.GpuIndexFlatL2(self.dimension),
                    self.dimension,
                    self.nlist,
                    self.m,
                    8,  # 8 bits per subquantizer
                )
            else:
                raise ValueError(f"Unknown index type: {index_type}")

            self.use_gpu = True
            self.is_trained = False
        except Exception as e:
            logger.error(f"Failed to create GPU index: {e}")
            self._create_cpu_index(index_type)

    def _create_cpu_index(self, index_type: str):
        """Create CPU-based FAISS index."""
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        elif index_type == "pq":
            self.index = faiss.IndexIVFPQ(
                faiss.IndexFlatL2(self.dimension),
                self.dimension,
                self.nlist,
                self.m,
                8,  # 8 bits per subquantizer
            )
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.use_gpu = False
        self.is_trained = False

    def add_embeddings(self, embeddings: np.ndarray):
        """
        Add embeddings to the index.

        Args:
            embeddings: Array of embeddings to add
        """
        if not self.is_trained and self.index_type != "flat":
            # For flat index, we don't need training
            pass

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        # Convert to float32
        embeddings = embeddings.astype(np.float32)

        # Train index if needed (for IVF and PQ)
        if not self.is_trained and self.index_type in ["ivf", "pq"]:
            logger.info("Training FAISS index...")
            self.index.train(embeddings)
            self.is_trained = True

        # Add embeddings to index
        if self.use_gpu:
            # For GPU, we need to handle device transfer
            embeddings_gpu = faiss.vector_to_array(embeddings).copy()
            self.index.add(embeddings_gpu)
        else:
            # For CPU, we can add directly
            self.index.add(embeddings)

        logger.info(f"Added {embeddings.shape[0]} embeddings to FAISS index")

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query embedding
            k: Number of results to return

        Returns:
            Tuple of (distances, indices)
        """
        # For flat index, we don't need training
        if not self.is_trained and self.index_type != "flat":
            raise ValueError("Index not trained")

        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        if self.use_gpu:
            # For GPU, we need to convert query and get results from GPU
            distances, indices = self.index.search(query_embedding, k)
        else:
            distances, indices = self.index.search(query_embedding, k)

        return distances[0], indices[0]

    def save(self, path: str):
        """Save FAISS index to file."""
        if self.index is None:
            raise ValueError("No index to save")

        faiss.write_index(self.index, path)
        logger.info(f"FAISS index saved to {path}")

    @classmethod
    def load(
        cls, path: str, dimension: int, use_gpu: bool = False, index_type: str = "flat"
    ):
        """
        Load FAISS index from file.

        Args:
            path: Path to index file
            dimension: Dimension of embeddings
            use_gpu: Whether to use GPU
            index_type: Type of FAISS index

        Returns:
            FAISSIndex instance
        """
        index = cls(dimension, use_gpu, index_type)
        index.index = faiss.read_index(path)
        index.is_trained = True
        logger.info(f"FAISS index loaded from {path}")
        return index


class ChromaDBManager:
    """
    ChromaDB manager for persistent storage.

    Features:
    - Persistent storage of knowledge base entries
    - Metadata filtering
    - Batch operations
    """

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Initialize ChromaDB manager.

        Args:
            persist_directory: Directory for persistent storage
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"description": "KnowledgeBase entries with metadata"},
        )

        logger.info(f"ChromaDB initialized with persist directory: {persist_directory}")

    def add_entry(self, entry: KnowledgeBaseEntry):
        """
        Add a single entry to ChromaDB.

        Args:
            entry: KnowledgeBaseEntry to add
        """
        # Prepare metadata
        metadata = entry.metadata.copy()
        metadata.update(
            {
                "cluster_id": entry.cluster_id,
                "timestamp": entry.timestamp,
                "text_length": len(entry.text),
                "has_summary": entry.summary is not None,
                "has_explanation": entry.explanation is not None,
            }
        )

        # Add to collection
        self.collection.add(
            documents=[entry.text],
            embeddings=[entry.embedding.tolist()],
            metadatas=[metadata],
            ids=[entry.id],
        )

        logger.debug(f"Added entry {entry.id} to ChromaDB")

    def add_entries_batch(self, entries: List[KnowledgeBaseEntry]):
        """
        Add multiple entries to ChromaDB in batch.

        Args:
            entries: List of KnowledgeBaseEntry objects
        """
        if not entries:
            return

        # Prepare batch data
        documents = [entry.text for entry in entries]
        embeddings = [entry.embedding.tolist() for entry in entries]
        metadatas = []
        ids = []

        for entry in entries:
            metadata = entry.metadata.copy()
            metadata.update(
                {
                    "cluster_id": entry.cluster_id,
                    "timestamp": entry.timestamp,
                    "text_length": len(entry.text),
                    "has_summary": entry.summary is not None,
                    "has_explanation": entry.explanation is not None,
                }
            )
            metadatas.append(metadata)
            ids.append(entry.id)

        # Add to collection
        self.collection.add(
            documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

        logger.info(f"Added {len(entries)} entries to ChromaDB in batch")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar entries in ChromaDB.

        Args:
            query_embedding: Query embedding
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results
        """
        # Convert numpy array to list for ChromaDB
        query_embedding = query_embedding.tolist()

        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=k, where=filters
        )

        return results

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            Entry data or None if not found
        """
        try:
            result = self.collection.get(ids=[entry_id])
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "text": result["documents"][0],
                    "embedding": result["embeddings"][0],
                    "metadata": result["metadatas"][0],
                }
            return None
        except Exception as e:
            logger.error(f"Error getting entry {entry_id}: {e}")
            return None

    def get_entries_by_cluster(self, cluster_id: int) -> List[Dict[str, Any]]:
        """
        Get all entries belonging to a specific cluster.

        Args:
            cluster_id: Cluster ID

        Returns:
            List of entries in the cluster
        """
        try:
            results = self.collection.get(where={"cluster_id": str(cluster_id)})
            return [
                {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "embedding": results["embeddings"][i],
                    "metadata": results["metadatas"][i],
                }
                for i in range(len(results["ids"]))
            ]
        except Exception as e:
            logger.error(f"Error getting entries for cluster {cluster_id}: {e}")
            return []

    def delete_entry(self, entry_id: str):
        """
        Delete an entry from ChromaDB.

        Args:
            entry_id: Entry ID to delete
        """
        try:
            self.collection.delete(ids=[entry_id])
            logger.info(f"Deleted entry {entry_id} from ChromaDB")
        except Exception as e:
            logger.error(f"Error deleting entry {entry_id}: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "total_entries": count,
                "persist_directory": str(self.persist_directory),
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "total_entries": 0,
                "persist_directory": str(self.persist_directory),
            }


class KnowledgeBase:
    """
    Main KnowledgeBase class combining FAISS and ChromaDB.

    Features:
    - Fast similarity search with FAISS
    - Persistent storage with ChromaDB
    - Batch operations
    - Metadata management
    - Search and retrieval
    """

    def __init__(
        self,
        persist_directory: str = None,
        embedding_dimension: int = None,
        use_gpu: bool = None,
        index_type: str = None,
    ):
        """
        Initialize KnowledgeBase.

        Args:
            persist_directory: Directory for persistent storage
            embedding_dimension: Dimension of embeddings
            use_gpu: Whether to use GPU for FAISS
            index_type: Type of FAISS index ('flat', 'ivf', 'pq')
        """
        # Load configuration from pyproject.toml
        self.config = _load_config()

        # Use provided parameters or fall back to config
        self.persist_directory = Path(
            persist_directory
            or self.config.get("database_persist_directory", "./data/knowledge_base")
        )
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager()
        self.embedding_dimension = (
            embedding_dimension or self.embedding_manager.get_embedding_dimension()
        )

        # Override embedding dimension if explicitly provided
        if embedding_dimension:
            logger.warning(f"Overriding embedding dimension to {embedding_dimension}")
            self.embedding_dimension = embedding_dimension

        self.use_gpu = (
            use_gpu
            if use_gpu is not None
            else self.config.get("database_use_gpu", False)
        )
        self.index_type = index_type or self.config.get("database_index_type", "flat")

        # Initialize storage components
        self.chroma_db = ChromaDBManager(str(self.persist_directory / "chroma_db"))
        self.faiss_index = FAISSIndex(
            self.embedding_dimension, self.use_gpu, self.index_type
        )

        # Initialize synchronization manager
        self.sync_manager = SyncManager(self)

        # Entry tracking
        self.entries: Dict[str, KnowledgeBaseEntry] = {}
        self.is_index_built = False

        logger.info(
            f"KnowledgeBase initialized with persist directory: {self.persist_directory}, "
            f"index_type: {self.index_type}, use_gpu: {self.use_gpu}, "
            f"embedding_method: {self.embedding_manager.get_method_info()}"
        )

        logger.info(
            f"KnowledgeBase initialized with persist directory: {self.persist_directory}, "
            f"index_type: {self.index_type}, use_gpu: {self.use_gpu}"
        )

    def add_entry(self, entry: KnowledgeBaseEntry):
        """
        Add a single entry to the knowledge base.

        Args:
            entry: KnowledgeBaseEntry to add
        """
        # Store entry
        self.entries[entry.id] = entry

        # Add to ChromaDB
        self.chroma_db.add_entry(entry)

        # Build index if not built
        if not self.is_index_built:
            self._build_index()

        logger.debug(f"Added entry {entry.id} to knowledge base")

    def add_entries_batch(self, entries: List[KnowledgeBaseEntry]):
        """
        Add multiple entries to the knowledge base in batch.

        Args:
            entries: List of KnowledgeBaseEntry objects
        """
        if not entries:
            return

        # Store entries
        for entry in entries:
            self.entries[entry.id] = entry

        # Add to ChromaDB in batch
        self.chroma_db.add_entries_batch(entries)

        # Build index if not built
        if not self.is_index_built:
            self._build_index()

        logger.info(f"Added {len(entries)} entries to knowledge base in batch")

    def _build_index(self):
        """Build FAISS index from all entries."""
        if not self.entries:
            return

        logger.info("Building FAISS index...")

        # Extract embeddings
        embeddings = np.array([entry.embedding for entry in self.entries.values()])

        # Add to FAISS index
        self.faiss_index.add_embeddings(embeddings)
        self.is_index_built = True

        logger.info(f"FAISS index built with {len(embeddings)} embeddings")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar entries in the knowledge base.

        Args:
            query_embedding: Query embedding
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        if not self.is_index_built:
            self._build_index()

        # Search with FAISS
        distances, indices = self.faiss_index.search(query_embedding, k)

        # Convert to SearchResult objects
        results = []
        for i, (distance, idx) in enumerate(zip(distances, indices)):
            if idx != -1:  # Valid index
                entry_id = list(self.entries.keys())[idx]
                entry = self.entries[entry_id]

                # Convert distance to similarity score (lower distance = higher similarity)
                similarity_score = 1.0 / (1.0 + distance)

                results.append(
                    SearchResult(
                        entry=entry, similarity_score=similarity_score, rank=i + 1
                    )
                )

        # Apply filters if provided
        if filters:
            results = [
                result
                for result in results
                if self._matches_filters(result.entry, filters)
            ]

        return results[:k]  # Return top k results

    def _matches_filters(
        self, entry: KnowledgeBaseEntry, filters: Dict[str, Any]
    ) -> bool:
        """Check if entry matches given filters."""
        for key, value in filters.items():
            if key == "cluster_id" and entry.cluster_id != value:
                return False
            elif key in entry.metadata and entry.metadata[key] != value:
                return False
        return True

    def get_entry(self, entry_id: str) -> Optional[KnowledgeBaseEntry]:
        """
        Get a specific entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            KnowledgeBaseEntry or None if not found
        """
        return self.entries.get(entry_id)

    def get_entries_by_cluster(self, cluster_id: int) -> List[KnowledgeBaseEntry]:
        """
        Get all entries belonging to a specific cluster.

        Args:
            cluster_id: Cluster ID

        Returns:
            List of entries in the cluster
        """
        return [
            entry for entry in self.entries.values() if entry.cluster_id == cluster_id
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Dictionary with statistics
        """
        chroma_stats = self.chroma_db.get_collection_stats()

        # Calculate cluster distribution
        cluster_counts = {}
        for entry in self.entries.values():
            if entry.cluster_id is not None:
                cluster_counts[entry.cluster_id] = (
                    cluster_counts.get(entry.cluster_id, 0) + 1
                )

        return {
            "total_entries": len(self.entries),
            "unique_clusters": len(cluster_counts),
            "cluster_distribution": cluster_counts,
            "chroma_stats": chroma_stats,
            "faiss_index_built": self.is_index_built,
            "use_gpu": self.use_gpu,
            "index_type": self.index_type,
            "embedding_dimension": self.embedding_dimension,
            "embedding_method": self.embedding_manager.get_method_info(),
        }

    def save(self, path: str):
        """
        Save knowledge base to file.

        Args:
            path: Path to save the knowledge base
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss_path = save_path / "faiss.index"
        self.faiss_index.save(str(faiss_path))

        # Save entries
        entries_data = {
            entry_id: asdict(entry) for entry_id, entry in self.entries.items()
        }

        with open(save_path / "entries.json", "w", encoding="utf-8") as f:
            json.dump(entries_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Knowledge base saved to {path}")

    def load(self, path: str):
        """
        Load knowledge base from file.

        Args:
            path: Path to load the knowledge base from
        """
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Knowledge base not found at {path}")

        # Load FAISS index
        faiss_path = load_path / "faiss.index"
        if faiss_path.exists():
            self.faiss_index = FAISSIndex.load(
                str(faiss_path), self.embedding_dimension, self.use_gpu, self.index_type
            )
            self.is_index_built = True

        # Load entries
        with open(load_path / "entries.json", "r", encoding="utf-8") as f:
            entries_data = json.load(f)

        # Recreate entries
        self.entries = {}
        for entry_id, entry_data in entries_data.items():
            entry_data["embedding"] = np.array(entry_data["embedding"])
            self.entries[entry_id] = KnowledgeBaseEntry(**entry_data)

        logger.info(f"Knowledge base loaded from {path}")

    def clear(self):
        """Clear all entries from the knowledge base."""
        self.entries.clear()
        self.is_index_built = False

        # Clear ChromaDB collection
        try:
            self.chroma_db.client.delete_collection("knowledge_base")
            self.chroma_db.collection = self.chroma_db.client.get_or_create_collection(
                "knowledge_base"
            )
            logger.info("Cleared ChromaDB collection")
        except Exception as e:
            logger.error(f"Error clearing ChromaDB: {e}")

        logger.info("Cleared all entries from knowledge base")


def create_knowledge_base_from_clusters(
    cluster_data: List[Tuple[int, List[str]]],
    embeddings: Optional[List[np.ndarray]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    embedding_manager: Optional[EmbeddingManager] = None,
) -> KnowledgeBase:
    """
    Create a knowledge base from clustered data.

    Args:
        cluster_data: List of (cluster_id, texts) tuples
        embeddings: Optional list of embeddings corresponding to texts
        metadata: Optional metadata for each entry
        embedding_manager: Optional EmbeddingManager instance for auto-generation

    Returns:
        KnowledgeBase instance
    """
    kb = KnowledgeBase()

    # If no embeddings provided, generate them using the embedding manager
    if embeddings is None:
        if embedding_manager is None:
            embedding_manager = EmbeddingManager()

        # Collect all texts
        all_texts = []
        for cluster_id, texts in cluster_data:
            all_texts.extend(texts)

        # Generate embeddings
        logger.info(
            f"Generating embeddings for {len(all_texts)} texts using {embedding_manager.get_method_info()}"
        )
        embeddings = embedding_manager.generate_embeddings(all_texts)

        # Split embeddings back by cluster
        embeddings_by_cluster = []
        current_idx = 0
        for cluster_id, texts in cluster_data:
            cluster_embeddings = embeddings[current_idx : current_idx + len(texts)]
            embeddings_by_cluster.append(cluster_embeddings)
            current_idx += len(texts)
        embeddings = embeddings_by_cluster

    entries = []
    for i, (cluster_id, texts) in enumerate(cluster_data):
        for j, text in enumerate(texts):
            entry_id = f"cluster_{cluster_id}_entry_{j}"

            entry_metadata = (
                metadata[i][j]
                if metadata and i < len(metadata) and j < len(metadata[i])
                else {}
            )

            # Get embedding for this text
            if i < len(embeddings) and j < len(embeddings[i]):
                embedding = embeddings[i][j]
            else:
                # Fallback to zero embedding
                embedding = np.zeros(kb.embedding_dimension)

            entry = KnowledgeBaseEntry(
                id=entry_id,
                text=text,
                embedding=embedding,
                metadata=entry_metadata,
                cluster_id=cluster_id,
                timestamp=time.time(),
            )
            entries.append(entry)

    kb.add_entries_batch(entries)
    return kb


if __name__ == "__main__":
    # Example usage
    try:
        # Load configuration
        config = _load_config()

        # Create sample data
        cluster_data = [
            (0, ["Dies ist Text 1 aus Cluster 0", "Dies ist Text 2 aus Cluster 0"]),
            (
                1,
                [
                    "Dies ist Text 1 aus Cluster 1",
                    "Dies ist Text 2 aus Cluster 1",
                    "Dies ist Text 3 aus Cluster 1",
                ],
            ),
        ]

        # Create knowledge base with auto-generated embeddings
        kb = create_knowledge_base_from_clusters(cluster_data)

        # Print statistics
        stats = kb.get_statistics()
        print(f"KnowledgeBase statistics: {stats}")

        # Test search with auto-generated query embedding
        query_texts = ["Was ist der wichtigste Punkt?"]
        query_embedding = kb.embedding_manager.generate_embeddings(query_texts)[0]
        results = kb.search(query_embedding, k=5)

        print(f"Search results: {len(results)} entries found")
        for result in results:
            print(
                f"  - Entry {result.entry.id}: similarity={result.similarity_score:.3f}"
            )

    except Exception as e:
        print(f"Error: {e}")
