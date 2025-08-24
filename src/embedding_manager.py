"""
Embedding Manager Module

This module provides functionality for generating embeddings using different methods:
- Ollama with BGE-M3 model
- Sentence Transformers with paraphrase-multilingual-MiniLM-L12-v2
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class EmbeddingManager:
    """
    Manager for generating embeddings using different methods.

    Supports:
    - Ollama with BGE-M3 model
    - Sentence Transformers with paraphrase-multilingual-MiniLM-L12-v2
    """

    def __init__(self, method: str = None, model_name: str = None):
        """
        Initialize embedding manager.

        Args:
            method: Embedding method ('ollama' or 'sentence-transformer')
            model_name: Model name to use
        """
        self.method = method or os.getenv("KB_EMBEDDING_METHOD", "ollama")
        self.model_name = model_name or os.getenv(
            "KB_SENTENCE_TRANSFORMER_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Ollama configuration
        self.ollama_url = os.getenv("KB_OLLAMA_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("KB_BGE_MODEL", "bge-m3")

        # Initialize models
        self.sentence_transformer_model = None
        self.openai_client = None
        self._initialize_model()

        logger.info(
            f"EmbeddingManager initialized with method: {self.method}, model: {self.model_name}"
        )

    def _initialize_model(self):
        """Initialize the appropriate embedding model."""
        if self.method == "sentence-transformer":
            try:
                logger.info(f"Loading sentence transformer model: {self.model_name}")
                self.sentence_transformer_model = SentenceTransformer(self.model_name)
                logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model: {e}")
                raise
        elif self.method == "ollama":
            logger.info(f"Using Ollama model: {self.ollama_model}")
            # Initialize OpenAI client for Ollama
            self.openai_client = OpenAI(
                base_url=f"{self.ollama_url}/v1/",
                api_key="ollama",
            )
            # Test Ollama connection
            if not self._test_ollama_connection():
                logger.warning("Ollama connection failed, embeddings may not work")
        else:
            raise ValueError(f"Unsupported embedding method: {self.method}")

    def _test_ollama_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            # Test with OpenAI client
            response = self.openai_client.models.list()
            logger.info("Ollama connection successful")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to encode

        Returns:
            numpy array of embeddings
        """
        if not texts:
            raise ValueError("Text list cannot be empty")

        if self.method == "sentence-transformer":
            return self._generate_with_sentence_transformer(texts)
        elif self.method == "ollama":
            return self._generate_with_ollama(texts)
        else:
            raise ValueError(f"Unsupported embedding method: {self.method}")

    def _generate_with_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence transformers."""
        try:
            logger.info(
                f"Generating embeddings for {len(texts)} texts using sentence transformer"
            )
            embeddings = self.sentence_transformer_model.encode(
                texts, convert_to_numpy=True
            )
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(
                f"Failed to generate embeddings with sentence transformer: {e}"
            )
            raise

    def _generate_with_ollama(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Ollama with OpenAI client."""
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using Ollama")

            # Use OpenAI client for Ollama
            response = self.openai_client.embeddings.create(
                model=self.ollama_model, input=texts, encoding_format="float"
            )

            # Extract embeddings
            embeddings = np.array([data.embedding for data in response.data])

            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings with Ollama: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if self.method == "sentence-transformer":
            # For sentence transformers, we can get the dimension from the model
            if self.sentence_transformer_model:
                return (
                    self.sentence_transformer_model.get_sentence_embedding_dimension()
                )
            else:
                # Default dimension for paraphrase-multilingual-MiniLM-L12-v2
                return 384
        elif self.method == "ollama":
            # BGE-M3 produces 1024-dimensional embeddings
            return 1024
        else:
            raise ValueError(f"Unsupported embedding method: {self.method}")

    def get_method_info(self) -> Dict[str, Any]:
        """Get information about the current embedding method."""
        return {
            "method": self.method,
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "ollama_url": self.ollama_url,
            "ollama_model": self.ollama_model,
        }

    def switch_method(self, method: str, model_name: str = None):
        """
        Switch to a different embedding method.

        Args:
            method: New embedding method ('ollama' or 'sentence-transformer')
            model_name: New model name (optional)
        """
        logger.info(f"Switching embedding method from {self.method} to {method}")
        self.method = method
        if model_name:
            self.model_name = model_name
        self._initialize_model()


# Factory function for creating embedding managers
def create_embedding_manager(
    method: str = None, model_name: str = None
) -> EmbeddingManager:
    """
    Create an embedding manager instance.

    Args:
        method: Embedding method ('ollama' or 'sentence-transformer')
        model_name: Model name to use

    Returns:
        EmbeddingManager instance
    """
    return EmbeddingManager(method, model_name)


# Example usage
if __name__ == "__main__":
    # Test with sentence transformers
    print("Testing sentence transformer embeddings...")
    try:
        st_manager = create_embedding_manager("sentence-transformer")
        texts = ["Dies ist ein Testtext", "Noch ein Beispieltext"]
        embeddings = st_manager.generate_embeddings(texts)
        print(f"Sentence transformer embeddings shape: {embeddings.shape}")
        print(f"Method info: {st_manager.get_method_info()}")
    except Exception as e:
        print(f"Sentence transformer test failed: {e}")

    # Test with Ollama
    print("\nTesting Ollama embeddings...")
    try:
        ollama_manager = create_embedding_manager("ollama")
        texts = ["Dies ist ein Testtext", "Noch ein Beispieltext"]
        embeddings = ollama_manager.generate_embeddings(texts)
        print(f"Ollama embeddings shape: {embeddings.shape}")
        print(f"Method info: {ollama_manager.get_method_info()}")
    except Exception as e:
        print(f"Ollama test failed: {e}")
