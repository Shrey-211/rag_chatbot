"""In-memory vector store for testing."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.vectorstore.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing.

    Not recommended for production use with large datasets.
    """

    def __init__(self):
        """Initialize in-memory vector store."""
        self.embeddings: Dict[str, np.ndarray] = {}
        self.documents: Dict[str, str] = {}
        self.metadatas: Dict[str, Dict[str, Any]] = {}
        logger.info("InMemoryVectorStore initialized")

    def upsert(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert or update documents.

        Args:
            ids: Document IDs
            embeddings: Document embeddings
            documents: Document content
            metadatas: Document metadata
        """
        if metadatas is None:
            metadatas = [{} for _ in ids]

        for i, doc_id in enumerate(ids):
            self.embeddings[doc_id] = embeddings[i]
            self.documents[doc_id] = documents[i]
            self.metadatas[doc_id] = metadatas[i]

        logger.debug(f"Upserted {len(ids)} documents to memory")

    def query(
        self, query_embedding: np.ndarray, top_k: int = 5, filter_dict: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Query for similar documents using cosine similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_dict: Metadata filter

        Returns:
            List of search results
        """
        if not self.embeddings:
            return []

        # Ensure query is 1D
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()

        # Calculate cosine similarities
        similarities = []
        for doc_id, doc_embedding in self.embeddings.items():
            # Apply metadata filter if provided
            if filter_dict:
                metadata = self.metadatas.get(doc_id, {})
                if not all(metadata.get(k) == v for k, v in filter_dict.items()):
                    continue

            # Cosine similarity
            doc_emb = doc_embedding.flatten()
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            similarities.append((doc_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        search_results = []
        for doc_id, score in similarities[:top_k]:
            search_results.append(
                SearchResult(
                    id=doc_id,
                    content=self.documents[doc_id],
                    metadata=self.metadatas.get(doc_id, {}),
                    score=float(score),
                )
            )

        return search_results

    def delete(self, ids: Optional[List[str]] = None, filter_dict: Optional[Dict] = None) -> None:
        """Delete documents.

        Args:
            ids: Document IDs to delete
            filter_dict: Metadata filter
        """
        if ids:
            for doc_id in ids:
                self.embeddings.pop(doc_id, None)
                self.documents.pop(doc_id, None)
                self.metadatas.pop(doc_id, None)
            logger.debug(f"Deleted {len(ids)} documents from memory")

        elif filter_dict:
            # Delete by filter
            to_delete = []
            for doc_id, metadata in self.metadatas.items():
                if all(metadata.get(k) == v for k, v in filter_dict.items()):
                    to_delete.append(doc_id)
            self.delete(ids=to_delete)

    def persist(self) -> None:
        """No-op for in-memory store."""
        logger.debug("InMemoryVectorStore persist (no-op)")

    def count(self) -> int:
        """Get document count.

        Returns:
            Number of documents
        """
        return len(self.documents)

    def get_by_ids(self, ids: List[str]) -> List[SearchResult]:
        """Get documents by IDs.

        Args:
            ids: Document IDs

        Returns:
            List of search results
        """
        search_results = []
        for doc_id in ids:
            if doc_id in self.documents:
                search_results.append(
                    SearchResult(
                        id=doc_id,
                        content=self.documents[doc_id],
                        metadata=self.metadatas.get(doc_id, {}),
                        score=1.0,
                    )
                )
        return search_results

