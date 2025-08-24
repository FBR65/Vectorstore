"""
FastAPI Server for KnowledgeBase Vector Database.

This module provides a REST API for vector database operations using FAISS and ChromaDB.
"""

import logging
import time
import threading
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tomllib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
from ..knowledge_base import KnowledgeBase, KnowledgeBaseEntry, SearchResult


def _load_config():
    """Load configuration from pyproject.toml."""
    try:
        import tomllib

        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
                return config.get("tool", {}).get("knowledgebase", {})
    except Exception as e:
        logger.warning(f"Failed to load config from pyproject.toml: {e}")
    return {}


# Pydantic models for API
class SearchRequest(BaseModel):
    """Request model for vector search."""

    vector: List[float]
    k: int = 10
    filters: Optional[Dict[str, Any]] = None
    index_type: str = "flat"


class AddVectorRequest(BaseModel):
    """Request model for adding a vector."""

    vector: List[float]
    text: str
    metadata: Optional[Dict[str, Any]] = None
    cluster_id: Optional[int] = None
    index_type: str = "flat"


class BatchAddRequest(BaseModel):
    """Request model for batch adding vectors."""

    vectors: List[List[float]]
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    cluster_ids: Optional[List[int]] = None
    index_type: str = "flat"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    faiss_index_built: bool
    total_entries: int
    use_gpu: bool
    index_type: str


class KnowledgeBaseServer:
    """
    FastAPI server for KnowledgeBase vector database operations.

    Features:
    - REST API for vector search and management
    - Automatic GPU/CPU detection
    - Multiple index types (flat, IVF, PQ)
    - Thread-safe operations
    - Background synchronization
    - Health monitoring
    """

    def __init__(
        self,
        persist_directory: str = "./data/knowledge_base",
        embedding_dimension: int = 1024,
        use_gpu: bool = False,
        index_type: str = "flat",
        max_workers: int = 4,
    ):
        """
        Initialize the KnowledgeBase server.

        Args:
            persist_directory: Directory for persistent storage
            embedding_dimension: Dimension of embeddings
            use_gpu: Whether to use GPU for FAISS
            index_type: Type of FAISS index
            max_workers: Maximum number of worker threads
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.embedding_dimension = embedding_dimension
        self.use_gpu = use_gpu
        self.index_type = index_type
        self.max_workers = max_workers

        # Initialize components
        self.knowledge_base = None
        self.sync_manager = None
        self.concurrency_manager = None
        self._initialize_components()

        # Setup FastAPI app
        self.app = FastAPI(
            title="KnowledgeBase Vector Database",
            description="A scalable vector database with FAISS and ChromaDB",
            version="0.2.0",
        )

        # Setup middleware
        self._setup_middleware()

        # Setup routes
        self._setup_routes()

        # Setup background tasks
        self._setup_background_tasks()

        logger.info(
            f"KnowledgeBase server initialized with index_type: {index_type}, use_gpu: {use_gpu}"
        )

    def _initialize_components(self):
        """Initialize core components."""
        try:
            # Initialize knowledge base
            self.knowledge_base = KnowledgeBase(
                persist_directory=str(self.persist_directory),
                embedding_dimension=self.embedding_dimension,
                use_gpu=self.use_gpu,
                index_type=self.index_type,
            )

            # Sync manager is now integrated into the KnowledgeBase class
            self.sync_manager = self.knowledge_base.sync_manager

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/health")
        async def health_check() -> HealthResponse:
            """Health check endpoint."""
            try:
                stats = self.knowledge_base.get_statistics()
                return HealthResponse(
                    status="healthy",
                    version="0.2.0",
                    faiss_index_built=stats.get("faiss_index_built", False),
                    total_entries=stats.get("total_entries", 0),
                    use_gpu=stats.get("use_gpu", False),
                    index_type=self.index_type,
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/vectors/search")
        async def search_vectors(request: SearchRequest) -> Dict[str, Any]:
            """
            Search for similar vectors in the knowledge base.

            Args:
                request: Search request containing vector and parameters

            Returns:
                Dictionary with search results
            """
            try:
                # Validate input
                if len(request.vector) != self.embedding_dimension:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Vector dimension mismatch: expected {self.embedding_dimension}, got {len(request.vector)}",
                    )

                # Convert to numpy array
                query_vector = np.array(request.vector, dtype=np.float32)

                # Perform search
                results = self.knowledge_base.search(
                    query_embedding=query_vector,
                    k=request.k,
                    filters=request.filters,
                )

                # Format results
                formatted_results = []
                for result in results:
                    formatted_results.append(
                        {
                            "id": result.entry.id,
                            "text": result.entry.text,
                            "similarity_score": result.similarity_score,
                            "rank": result.rank,
                            "metadata": result.entry.metadata,
                            "cluster_id": result.entry.cluster_id,
                        }
                    )

                return {
                    "results": formatted_results,
                    "total_results": len(formatted_results),
                    "query_dimension": len(request.vector),
                    "index_type": self.index_type,
                    "use_gpu": self.use_gpu,
                }

            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/vectors/add")
        async def add_vector(request: AddVectorRequest) -> Dict[str, Any]:
            """
            Add a single vector to the knowledge base.

            Args:
                request: Add vector request containing vector and metadata

            Returns:
                Dictionary with operation result
            """
            try:
                # Validate input
                if len(request.vector) != self.embedding_dimension:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Vector dimension mismatch: expected {self.embedding_dimension}, got {len(request.vector)}",
                    )

                # Create entry
                entry = KnowledgeBaseEntry(
                    id=f"vector_{int(time.time())}_{np.random.randint(0, 1000000)}",
                    text=request.text,
                    embedding=np.array(request.vector, dtype=np.float32),
                    metadata=request.metadata or {},
                    cluster_id=request.cluster_id,
                    timestamp=time.time(),
                )

                # Add to knowledge base
                self.knowledge_base.add_entry(entry)

                return {
                    "id": entry.id,
                    "message": "Vector added successfully",
                    "total_entries": len(self.knowledge_base.entries),
                    "index_type": self.index_type,
                    "use_gpu": self.use_gpu,
                }

            except Exception as e:
                logger.error(f"Add vector failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/vectors/batch-add")
        async def batch_add_vectors(request: BatchAddRequest) -> Dict[str, Any]:
            """
            Add multiple vectors to the knowledge base in batch.

            Args:
                request: Batch add request containing vectors and metadata

            Returns:
                Dictionary with operation result
            """
            try:
                # Validate input
                if len(request.vectors) != len(request.texts):
                    raise HTTPException(
                        status_code=400,
                        detail="Number of vectors must match number of texts",
                    )

                if any(len(v) != self.embedding_dimension for v in request.vectors):
                    raise HTTPException(
                        status_code=400,
                        detail=f"All vectors must have dimension {self.embedding_dimension}",
                    )

                # Create entries
                entries = []
                for i, (vector, text) in enumerate(zip(request.vectors, request.texts)):
                    entry = KnowledgeBaseEntry(
                        id=f"vector_{int(time.time())}_{i}",
                        text=text,
                        embedding=np.array(vector, dtype=np.float32),
                        metadata=request.metadatas[i]
                        if request.metadatas and i < len(request.metadatas)
                        else {},
                        cluster_id=request.cluster_ids[i]
                        if request.cluster_ids and i < len(request.cluster_ids)
                        else None,
                        timestamp=time.time(),
                    )
                    entries.append(entry)

                # Add to knowledge base in batch
                self.knowledge_base.add_entries_batch(entries)

                return {
                    "message": f"Successfully added {len(entries)} vectors",
                    "total_entries": len(self.knowledge_base.entries),
                    "index_type": self.index_type,
                    "use_gpu": self.use_gpu,
                }

            except Exception as e:
                logger.error(f"Batch add vectors failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/vectors/{vector_id}")
        async def get_vector(vector_id: str) -> Dict[str, Any]:
            """
            Get a specific vector by ID.

            Args:
                vector_id: ID of the vector to retrieve

            Returns:
                Dictionary with vector data
            """
            try:
                entry = self.knowledge_base.get_entry(vector_id)
                if not entry:
                    raise HTTPException(status_code=404, detail="Vector not found")

                return {
                    "id": entry.id,
                    "text": entry.text,
                    "embedding": entry.embedding.tolist(),
                    "metadata": entry.metadata,
                    "cluster_id": entry.cluster_id,
                    "timestamp": entry.timestamp,
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get vector failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/vectors/{vector_id}")
        async def delete_vector(vector_id: str) -> Dict[str, Any]:
            """
            Delete a specific vector by ID.

            Args:
                vector_id: ID of the vector to delete

            Returns:
                Dictionary with operation result
            """
            try:
                if vector_id not in self.knowledge_base.entries:
                    raise HTTPException(status_code=404, detail="Vector not found")

                # Remove from memory
                del self.knowledge_base.entries[vector_id]

                # Remove from ChromaDB
                self.knowledge_base.chroma_db.delete_entry(vector_id)

                # Rebuild FAISS index
                self.knowledge_base._build_index()

                return {
                    "message": "Vector deleted successfully",
                    "total_entries": len(self.knowledge_base.entries),
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Delete vector failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/statistics")
        async def get_statistics() -> Dict[str, Any]:
            """
            Get knowledge base statistics.

            Returns:
                Dictionary with statistics
            """
            try:
                stats = self.knowledge_base.get_statistics()
                return {
                    "total_entries": stats.get("total_entries", 0),
                    "unique_clusters": stats.get("unique_clusters", 0),
                    "cluster_distribution": stats.get("cluster_distribution", {}),
                    "faiss_index_built": stats.get("faiss_index_built", False),
                    "use_gpu": stats.get("use_gpu", False),
                    "index_type": self.index_type,
                    "chroma_stats": stats.get("chroma_stats", {}),
                    "server_name": "KnowledgeBase FastAPI Server",
                    "server_version": "0.2.0",
                    "host": self.host,
                    "port": self.port,
                    "api_version": "v1",
                    "database_persist_directory": str(
                        self.knowledge_base.persist_directory
                    ),
                    "database_embedding_dimension": self.knowledge_base.embedding_dimension,
                    "database_use_gpu": self.knowledge_base.use_gpu,
                    "database_index_type": self.knowledge_base.index_type,
                }

            except Exception as e:
                logger.error(f"Get statistics failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/backup")
        async def create_backup(background_tasks: BackgroundTasks) -> Dict[str, Any]:
            """
            Create a backup of the knowledge base.

            Args:
                background_tasks: Background tasks for backup creation

            Returns:
                Dictionary with backup result
            """
            try:
                # Create backup in background
                background_tasks.add_task(self._create_backup)

                return {
                    "message": "Backup creation started",
                    "backup_directory": str(self.persist_directory / "backups"),
                }

            except Exception as e:
                logger.error(f"Create backup failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/clear")
        async def clear_knowledge_base() -> Dict[str, Any]:
            """
            Clear all entries from the knowledge base.

            Returns:
                Dictionary with operation result
            """
            try:
                self.knowledge_base.clear()

                return {
                    "message": "Knowledge base cleared successfully",
                    "total_entries": 0,
                }

            except Exception as e:
                logger.error(f"Clear knowledge base failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_background_tasks(self):
        """Setup background tasks."""

        @self.app.on_event("startup")
        async def startup_event():
            """Initialize server on startup."""
            logger.info("Starting KnowledgeBase server...")

            # Try to load existing data
            try:
                backup_path = self.persist_directory / "backup"
                if backup_path.exists():
                    self.knowledge_base.load(str(backup_path))
                    logger.info("Loaded existing knowledge base from backup")
            except Exception as e:
                logger.warning(f"Could not load existing data: {e}")

            logger.info("KnowledgeBase server started successfully")

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup server on shutdown."""
            logger.info("Shutting down KnowledgeBase server...")

            # Save current state
            try:
                backup_path = self.persist_directory / "backup"
                self.knowledge_base.save(str(backup_path))
                logger.info("Saved knowledge base to backup")
            except Exception as e:
                logger.error(f"Failed to save backup: {e}")

            logger.info("KnowledgeBase server shutdown complete")

    async def _create_backup(self):
        """Create backup in background."""
        try:
            backup_path = self.persist_directory / "backups"
            backup_path.mkdir(exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_file = backup_path / f"backup_{timestamp}"

            self.knowledge_base.save(str(backup_file))
            logger.info(f"Backup created: {backup_file}")

        except Exception as e:
            logger.error(f"Backup creation failed: {e}")

    def run(self, host: str = None, port: int = None, reload: bool = None):
        """
        Run the FastAPI server.

        Args:
            host: Host to bind to (from config if not provided)
            port: Port to bind to (from config if not provided)
            reload: Whether to enable auto-reload (from config if not provided)
        """
        # Load configuration if not provided
        config = _load_config()
        host = host or config.get("server_host", "0.0.0.0")
        port = port or config.get("server_port", 8000)
        reload = reload if reload is not None else config.get("server_reload", False)

        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )


def create_server(
    persist_directory: str = None,
    embedding_dimension: int = None,
    use_gpu: bool = None,
    index_type: str = None,
    max_workers: int = None,
) -> KnowledgeBaseServer:
    """
    Create a KnowledgeBase server instance.

    Args:
        persist_directory: Directory for persistent storage
        embedding_dimension: Dimension of embeddings
        use_gpu: Whether to use GPU for FAISS
        index_type: Type of FAISS index
        max_workers: Maximum number of worker threads

    Returns:
        KnowledgeBaseServer instance
    """
    # Load configuration from pyproject.toml
    config = _load_config()

    # Use provided parameters or fall back to config
    persist_directory = persist_directory or config.get(
        "database_persist_directory", "./data/knowledge_base"
    )
    embedding_dimension = embedding_dimension or config.get(
        "database_embedding_dimension", 1024
    )
    use_gpu = use_gpu if use_gpu is not None else config.get("database_use_gpu", False)
    index_type = index_type or config.get("database_index_type", "flat")
    max_workers = max_workers or config.get("performance_max_workers", 4)

    return KnowledgeBaseServer(
        persist_directory=persist_directory,
        embedding_dimension=embedding_dimension,
        use_gpu=use_gpu,
        index_type=index_type,
        max_workers=max_workers,
    )


def main():
    """Main entry point for the server."""
    import argparse

    # Load configuration from pyproject.toml
    config = _load_config()

    parser = argparse.ArgumentParser(description="KnowledgeBase Vector Database Server")
    parser.add_argument("--host", help="Host to bind to (default from config)")
    parser.add_argument(
        "--port", type=int, help="Port to bind to (default from config)"
    )
    parser.add_argument(
        "--persist-dir", help="Persistence directory (default from config)"
    )
    parser.add_argument(
        "--embedding-dim", type=int, help="Embedding dimension (default from config)"
    )
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for FAISS")
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf", "pq"],
        help="FAISS index type (default from config)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of worker threads (default from config)",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    # Create and run server
    server = create_server(
        persist_directory=args.persist_dir,
        embedding_dimension=args.embedding_dim,
        use_gpu=args.use_gpu,
        index_type=args.index_type,
        max_workers=args.max_workers,
    )

    # Load server configuration
    server_host = args.host or config.get("server_host", "0.0.0.0")
    server_port = args.port or config.get("server_port", 8000)
    server_reload = (
        args.reload if args.reload is not None else config.get("server_reload", False)
    )

    server.run(host=server_host, port=server_port, reload=server_reload)


if __name__ == "__main__":
    main()
