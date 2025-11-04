"""Base interface for embedding adapters."""

from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np


class EmbeddingAdapter(ABC):
    """Abstract base class for embedding adapters.

    This interface ensures all embedding providers have consistent methods
    for text and file embedding.
    """

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)

        Raises:
            Exception: If embedding fails
        """
        pass

    @abstractmethod
    def embed_file(self, path: str) -> np.ndarray:
        """Embed a single file.

        Args:
            path: Path to the file

        Returns:
            numpy array of shape (1, embedding_dim)

        Raises:
            Exception: If embedding fails
        """
        pass

    def embed_batch(
        self, text_iterator: Iterator[str], batch_size: int = 32
    ) -> Iterator[np.ndarray]:
        """Embed texts in batches for efficiency.

        Args:
            text_iterator: Iterator of text strings
            batch_size: Number of texts to process at once

        Yields:
            numpy arrays of embeddings
        """
        batch = []
        for text in text_iterator:
            batch.append(text)
            if len(batch) >= batch_size:
                yield self.embed_texts(batch)
                batch = []

        # Process remaining items
        if batch:
            yield self.embed_texts(batch)

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this adapter.

        Returns:
            Embedding dimension
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the embedding service is available and healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        pass


# Future extension: Media embedding adapter interface
class MediaEmbeddingAdapter(ABC):
    """Abstract base class for media (image/video) embedding adapters.

    TODO: Implement this interface when adding image/video support.
    Potential methods:
    - embed_image(path: str) -> np.ndarray
    - embed_video(path: str) -> np.ndarray
    - embed_frames(frames: List[np.ndarray]) -> np.ndarray
    """

    pass

