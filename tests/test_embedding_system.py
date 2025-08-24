"""
Test Suite for Embedding System

This test suite validates the embedding functionality including:
- Ollama integration with OpenAI client
- Sentence Transformers integration
- KnowledgeBase integration
- Batch processing
- Error handling
"""

import unittest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the src directory to the Python path
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import the modules to test
from embedding_manager import EmbeddingManager
from knowledge_base import (
    KnowledgeBase,
    KnowledgeBaseEntry,
    create_knowledge_base_from_clusters,
)


class TestEmbeddingManager(unittest.TestCase):
    """Test cases for EmbeddingManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_texts = [
            "Dies ist ein Testtext für die Embedding-Generierung",
            "Noch ein Beispieltext für die Validierung",
            "Dritter Text für die Testumgebung",
        ]

    @patch("embedding_manager.OpenAI")
    def test_ollama_initialization(self, mock_openai):
        """Test Ollama client initialization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        manager = EmbeddingManager(method="ollama")

        # Verify OpenAI client was created with correct parameters
        mock_openai.assert_called_once_with(
            base_url="http://localhost:11434/v1/",
            api_key="ollama",
        )

        self.assertEqual(manager.method, "ollama")
        self.assertEqual(manager.ollama_model, "bge-m3")
        self.assertEqual(manager.openai_client, mock_client)

    @patch("embedding_manager.OpenAI")
    def test_ollama_embedding_generation(self, mock_openai):
        """Test Ollama embedding generation."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock embeddings response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3, 0.4]),
            Mock(embedding=[0.5, 0.6, 0.7, 0.8]),
            Mock(embedding=[0.9, 1.0, 1.1, 1.2]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        manager = EmbeddingManager(method="ollama")

        # Generate embeddings
        embeddings = manager.generate_embeddings(self.test_texts)

        # Verify the call
        mock_client.embeddings.create.assert_called_once_with(
            model="bge-m3", input=self.test_texts, encoding_format="float"
        )

        # Verify result
        expected_shape = (3, 4)
        self.assertEqual(embeddings.shape, expected_shape)
        self.assertTrue(isinstance(embeddings, np.ndarray))

    @patch("embedding_manager.SentenceTransformer")
    def test_sentence_transformer_initialization(self, mock_sentence_transformer):
        """Test SentenceTransformer initialization."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        manager = EmbeddingManager(method="sentence-transformer")

        # Verify SentenceTransformer was created with correct model
        mock_sentence_transformer.assert_called_once_with(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.assertEqual(manager.method, "sentence-transformer")
        self.assertEqual(manager.sentence_transformer_model, mock_model)

    @patch("embedding_manager.SentenceTransformer")
    def test_sentence_transformer_embedding_generation(self, mock_sentence_transformer):
        """Test SentenceTransformer embedding generation."""
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )
        mock_sentence_transformer.return_value = mock_model

        manager = EmbeddingManager(method="sentence-transformer")

        # Generate embeddings
        embeddings = manager.generate_embeddings(self.test_texts)

        # Verify the call
        mock_model.encode.assert_called_once_with(
            self.test_texts, convert_to_numpy=True
        )

        # Verify result
        expected_shape = (3, 3)
        self.assertEqual(embeddings.shape, expected_shape)
        self.assertTrue(isinstance(embeddings, np.ndarray))

    def test_embedding_dimension_detection(self):
        """Test embedding dimension detection."""
        with patch("embedding_manager.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model

            manager = EmbeddingManager(method="sentence-transformer")
            dimension = manager.get_embedding_dimension()

            self.assertEqual(dimension, 384)

    def test_ollama_dimension_detection(self):
        """Test Ollama embedding dimension detection."""
        manager = EmbeddingManager(method="ollama")
        dimension = manager.get_embedding_dimension()

        self.assertEqual(dimension, 1024)

    def test_invalid_method(self):
        """Test invalid embedding method."""
        with self.assertRaises(ValueError):
            EmbeddingManager(method="invalid-method")

    @patch("embedding_manager.OpenAI")
    def test_ollama_connection_test(self, mock_openai):
        """Test Ollama connection test."""
        mock_client = Mock()
        mock_client.models.list.return_value = []
        mock_openai.return_value = mock_client

        manager = EmbeddingManager(method="ollama")

        # Test connection
        result = manager._test_ollama_connection()

        # Verify the call
        mock_client.models.list.assert_called()
        self.assertTrue(result)

    @patch("embedding_manager.OpenAI")
    def test_ollama_connection_failure(self, mock_openai):
        """Test Ollama connection failure."""
        mock_client = Mock()
        mock_client.models.list.side_effect = Exception("Connection failed")
        mock_openai.return_value = mock_client

        manager = EmbeddingManager(method="ollama")

        # Test connection
        result = manager._test_ollama_connection()

        self.assertFalse(result)


class TestKnowledgeBaseIntegration(unittest.TestCase):
    """Test cases for KnowledgeBase integration with embeddings."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_texts = [
            "Dies ist ein Testtext für die KnowledgeBase",
            "Noch ein Beispieltext für die Integration",
            "Dritter Text für die Testumgebung",
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        try:
            # Try to close any open file handles
            import gc

            gc.collect()
            # Wait a moment for file handles to be released
            import time

            time.sleep(0.1)
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            # If cleanup fails, continue with test
            pass

    @patch("embedding_manager.OpenAI")
    def test_knowledge_base_with_ollama(self, mock_openai):
        """Test KnowledgeBase with Ollama embeddings."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1024),  # 1024 dimension embeddings (BGE-M3)
            Mock(embedding=[0.2] * 1024),  # 1024 dimension embeddings (BGE-M3)
            Mock(embedding=[0.3] * 1024),  # 1024 dimension embeddings (BGE-M3)
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Create KnowledgeBase with correct embedding dimension
        kb = KnowledgeBase(persist_directory=self.temp_dir, embedding_dimension=1024)

        # Add entries with mocked embeddings
        for i, text in enumerate(self.test_texts):
            entry = KnowledgeBaseEntry(
                id=f"entry_{i}",
                text=text,
                embedding=np.array(mock_response.data[i].embedding),
                metadata={"source": "test"},
                cluster_id=0,
            )
            kb.add_entry(entry)

        # Verify entries were added
        self.assertEqual(len(kb.entries), 3)

        # Test search with correct dimension
        query_embedding = kb.embedding_manager.generate_embeddings(
            ["Was ist der wichtigste Punkt?"]
        )[0]
        # Rebuild index to ensure it's trained
        kb._build_index()
        results = kb.search(query_embedding, k=3)

        self.assertEqual(len(results), 3)
        self.assertTrue(all(hasattr(result, "entry") for result in results))
        self.assertTrue(all(hasattr(result, "similarity_score") for result in results))

    @patch("embedding_manager.SentenceTransformer")
    def test_knowledge_base_with_sentence_transformer(self, mock_sentence_transformer):
        """Test KnowledgeBase with SentenceTransformer embeddings."""
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array(
            [
                [0.1] * 384,
                [0.2] * 384,
                [0.3] * 384,
            ]  # 384 dimension embeddings (SentenceTransformer standard)
        )
        mock_sentence_transformer.return_value = mock_model

        # Create KnowledgeBase with correct embedding dimension
        kb = KnowledgeBase(persist_directory=self.temp_dir, embedding_dimension=384)

        # Add entries with mocked embeddings
        for i, text in enumerate(self.test_texts):
            entry = KnowledgeBaseEntry(
                id=f"entry_{i}",
                text=text,
                embedding=np.array(mock_model.encode.return_value[i]),
                metadata={"source": "test"},
                cluster_id=0,
            )
            kb.add_entry(entry)

        # Verify entries were added
        self.assertEqual(len(kb.entries), 3)

    @patch("embedding_manager.OpenAI")
    def test_create_knowledge_base_from_clusters(self, mock_openai):
        """Test create_knowledge_base_from_clusters with auto-generated embeddings."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1024),  # 1024 dimension embeddings
            Mock(embedding=[0.2] * 1024),  # 1024 dimension embeddings
            Mock(embedding=[0.3] * 1024),  # 1024 dimension embeddings
            Mock(embedding=[0.4] * 1024),  # 1024 dimension embeddings
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Test data
        cluster_data = [
            (0, ["Text 1 aus Cluster 0", "Text 2 aus Cluster 0"]),
            (1, ["Text 1 aus Cluster 1"]),
        ]

        # Create knowledge base with auto-generated embeddings
        kb = create_knowledge_base_from_clusters(cluster_data)

        # Verify entries were created
        self.assertEqual(len(kb.entries), 3)

        # Verify cluster distribution
        cluster_counts = {}
        for entry in kb.entries.values():
            if entry.cluster_id is not None:
                cluster_counts[entry.cluster_id] = (
                    cluster_counts.get(entry.cluster_id, 0) + 1
                )

        self.assertEqual(cluster_counts[0], 2)
        self.assertEqual(cluster_counts[1], 1)

    def test_knowledge_base_statistics(self):
        """Test KnowledgeBase statistics."""
        kb = KnowledgeBase(persist_directory=self.temp_dir)

        # Add some test entries
        for i in range(3):
            entry = KnowledgeBaseEntry(
                id=f"entry_{i}",
                text=f"Testtext {i}",
                embedding=np.random.rand(1024),
                metadata={"source": "test"},
                cluster_id=i % 2,
            )
            kb.add_entry(entry)

        # Get statistics
        stats = kb.get_statistics()

        # Verify statistics
        self.assertEqual(stats["total_entries"], 3)
        self.assertEqual(stats["unique_clusters"], 2)
        self.assertIn("embedding_method", stats)
        self.assertIn("embedding_dimension", stats)


class TestBatchProcessing(unittest.TestCase):
    """Test cases for batch processing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_texts = [f"Testtext {i}" for i in range(100)]

    @patch("embedding_manager.OpenAI")
    def test_batch_embedding_generation(self, mock_openai):
        """Test batch embedding generation with Ollama."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]) for i in range(100)
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Create manager
        manager = EmbeddingManager(method="ollama")

        # Generate embeddings for large batch
        embeddings = manager.generate_embeddings(self.test_texts)

        # Verify the call
        mock_client.embeddings.create.assert_called_once()

        # Verify result
        expected_shape = (100, 4)
        self.assertEqual(embeddings.shape, expected_shape)

    @patch("embedding_manager.SentenceTransformer")
    def test_batch_sentence_transformer(self, mock_sentence_transformer):
        """Test batch processing with SentenceTransformer."""
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(100, 384)
        mock_sentence_transformer.return_value = mock_model

        # Create manager
        manager = EmbeddingManager(method="sentence-transformer")

        # Generate embeddings for large batch
        embeddings = manager.generate_embeddings(self.test_texts)

        # Verify the call
        mock_model.encode.assert_called_once()

        # Verify result
        expected_shape = (100, 384)
        self.assertEqual(embeddings.shape, expected_shape)


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling."""

    def test_empty_text_list(self):
        """Test handling of empty text list."""
        manager = EmbeddingManager(method="sentence-transformer")

        with self.assertRaises(ValueError):
            manager.generate_embeddings([])

    @patch("embedding_manager.OpenAI")
    def test_ollama_api_error(self, mock_openai):
        """Test handling of Ollama API errors."""
        # Mock OpenAI client that raises an exception
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        manager = EmbeddingManager(method="ollama")

        with self.assertRaises(Exception):
            manager.generate_embeddings(["Test text"])

    @patch("embedding_manager.SentenceTransformer")
    def test_sentence_transformer_error(self, mock_sentence_transformer):
        """Test handling of SentenceTransformer errors."""
        # Mock SentenceTransformer that raises an exception
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Model Error")
        mock_sentence_transformer.return_value = mock_model

        manager = EmbeddingManager(method="sentence-transformer")

        with self.assertRaises(Exception):
            manager.generate_embeddings(["Test text"])


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
