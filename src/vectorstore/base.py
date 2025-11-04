"""Base interface for vector stores."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SearchResult:
    """Result from vector similarity search."""

    id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class VectorStore(ABC):
    """Abstract base class for vector stores.

    Provides consistent interface for different vector database backends.
    """

    @abstractmethod
    def upsert(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert or update documents with their embeddings.

        Args:
            ids: Unique identifiers for documents
            embeddings: Document embeddings (n_docs, embedding_dim)
            documents: Document text content
            metadatas: Optional metadata for each document

        Raises:
            Exception: If upsert fails
        """
        pass

    @abstractmethod
    def query(
        self, query_embedding: np.ndarray, top_k: int = 5, filter_dict: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Query for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of SearchResult objects

        Raises:
            Exception: If query fails
        """
        pass

    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None, filter_dict: Optional[Dict] = None) -> None:
        """Delete documents from the store.

        Args:
            ids: Document IDs to delete
            filter_dict: Metadata filter for deletion

        Raises:
            Exception: If deletion fails
        """
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to disk.

        Raises:
            Exception: If persistence fails
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total number of documents in store.

        Returns:
            Document count
        """
        pass

    @abstractmethod
    def get_by_ids(self, ids: List[str]) -> List[SearchResult]:
        """Retrieve documents by their IDs.

        Args:
            ids: Document IDs to retrieve

        Returns:
            List of SearchResult objects
        """
        pass

