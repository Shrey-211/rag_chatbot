"""Local text embedding adapter using sentence-transformers."""

import logging
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.adapters.embedding.base import EmbeddingAdapter

logger = logging.getLogger(__name__)


class LocalTextEmbeddingAdapter(EmbeddingAdapter):
    """Local embedding adapter using sentence-transformers.

    Uses lightweight transformer models that run locally without API calls.
    Default model: all-MiniLM-L6-v2 (384 dimensions, good quality/speed tradeoff)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """Initialize local embedding adapter.

        Args:
            model_name: Name of sentence-transformers model
            device: Device to use ('cpu' or 'cuda')
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self._embedding_dim}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of text strings

        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([]).reshape(0, self._embedding_dim)

        try:
            # Only log for bulk embedding operations (not single queries)
            if len(texts) > 1:
                logger.info(f"🔢 Embedding {len(texts)} text chunks using {self.model_name}...")
            
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
            )
            
            if len(texts) > 1:
                logger.info(f"   ✓ Generated {len(embeddings)} embeddings ({self._embedding_dim}D vectors)")
            
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            raise

    def embed_file(self, path: str) -> np.ndarray:
        """Embed a single file by reading its content.

        Args:
            path: Path to the text file

        Returns:
            numpy array of shape (1, embedding_dim)
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            return self.embed_texts([content])
        except Exception as e:
            logger.error(f"Error embedding file {path}: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self._embedding_dim

    def health_check(self) -> bool:
        """Check if the model is loaded and working.

        Returns:
            True if model is healthy
        """
        try:
            # Test with a simple sentence
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            return test_embedding.shape == (1, self._embedding_dim)
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

