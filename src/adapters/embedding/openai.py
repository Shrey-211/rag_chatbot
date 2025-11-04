"""OpenAI embedding adapter implementation."""

import logging
from pathlib import Path
from typing import List

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.adapters.embedding.base import EmbeddingAdapter

logger = logging.getLogger(__name__)


class OpenAIEmbeddingAdapter(EmbeddingAdapter):
    """Embedding adapter using OpenAI's embedding API.

    Uses text-embedding-ada-002 or newer models.
    """

    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize OpenAI embedding adapter.

        Args:
            api_key: OpenAI API key
            model: Embedding model name
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self._embedding_dim = self.MODEL_DIMENSIONS.get(model, 1536)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using OpenAI API.

        Args:
            texts: List of text strings

        Returns:
            numpy array of embeddings

        Raises:
            openai.OpenAIError: If API request fails
        """
        if not texts:
            return np.array([]).reshape(0, self._embedding_dim)

        try:
            # OpenAI has a limit on batch size, split if needed
            max_batch = 100
            all_embeddings = []

            for i in range(0, len(texts), max_batch):
                batch = texts[i : i + max_batch]
                response = self.client.embeddings.create(input=batch, model=self.model)

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return np.array(all_embeddings)
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
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
        """Check if OpenAI embedding service is available.

        Returns:
            True if service is healthy
        """
        try:
            # Test with minimal input
            response = self.client.embeddings.create(input=["test"], model=self.model)
            return len(response.data) == 1
        except Exception as e:
            logger.warning(f"OpenAI embedding health check failed: {e}")
            return False

