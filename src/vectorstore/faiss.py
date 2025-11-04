"""FAISS vector store implementation."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from src.vectorstore.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """Vector store implementation using FAISS.

    FAISS (Facebook AI Similarity Search) is a library for efficient
    similarity search of dense vectors.
    """

    def __init__(
        self,
        embedding_dim: int,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        index_type: str = "Flat",
    ):
        """Initialize FAISS vector store.

        Args:
            embedding_dim: Dimension of embeddings
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata
            index_type: FAISS index type (Flat, IVFFlat, HNSW)
        """
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index_type = index_type

        # Storage for documents and metadata
        self.documents: Dict[str, str] = {}
        self.metadatas: Dict[str, Dict[str, Any]] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}

        # Initialize or load index
        if index_path and Path(index_path).exists():
            self._load()
        else:
            self._create_index()

        logger.info(
            f"FAISS initialized: dim={embedding_dim}, "
            f"type={index_type}, index_path={index_path}"
        )

    def _create_index(self):
        """Create a new FAISS index."""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

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

        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)

        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings)

        # Update mappings
        for i, doc_id in enumerate(ids):
            idx = start_idx + i

            # If ID exists, remove old entry
            if doc_id in self.id_to_idx:
                old_idx = self.id_to_idx[doc_id]
                del self.idx_to_id[old_idx]

            self.id_to_idx[doc_id] = idx
            self.idx_to_id[idx] = doc_id
            self.documents[doc_id] = documents[i]
            self.metadatas[doc_id] = metadatas[i]

        logger.debug(f"Upserted {len(ids)} documents to FAISS")

    def query(
        self, query_embedding: np.ndarray, top_k: int = 5, filter_dict: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Query for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_dict: Metadata filter

        Returns:
            List of search results
        """
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)

        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        # Parse results
        search_results = []
        for i in range(indices.shape[1]):
            idx = int(indices[0][i])
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            doc_id = self.idx_to_id.get(idx)
            if doc_id is None:
                continue

            # Apply metadata filter if provided
            if filter_dict:
                metadata = self.metadatas.get(doc_id, {})
                if not all(metadata.get(k) == v for k, v in filter_dict.items()):
                    continue

            # Convert distance to similarity score (inverse)
            score = 1.0 / (1.0 + float(distances[0][i]))

            search_results.append(
                SearchResult(
                    id=doc_id,
                    content=self.documents.get(doc_id, ""),
                    metadata=self.metadatas.get(doc_id, {}),
                    score=score,
                )
            )

        return search_results[:top_k]

    def delete(self, ids: Optional[List[str]] = None, filter_dict: Optional[Dict] = None) -> None:
        """Delete documents.

        Note: FAISS doesn't support true deletion, so we just remove from metadata.
        Consider rebuilding index periodically.

        Args:
            ids: Document IDs to delete
            filter_dict: Metadata filter
        """
        if ids:
            for doc_id in ids:
                if doc_id in self.id_to_idx:
                    idx = self.id_to_idx[doc_id]
                    del self.id_to_idx[doc_id]
                    del self.idx_to_id[idx]
                    del self.documents[doc_id]
                    del self.metadatas[doc_id]
            logger.debug(f"Deleted {len(ids)} documents from FAISS metadata")

    def persist(self) -> None:
        """Persist index and metadata to disk."""
        if self.index_path:
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            logger.debug(f"FAISS index saved to {self.index_path}")

        if self.metadata_path:
            Path(self.metadata_path).parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "documents": self.documents,
                "metadatas": self.metadatas,
                "id_to_idx": self.id_to_idx,
                "idx_to_id": self.idx_to_id,
            }
            with open(self.metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            logger.debug(f"FAISS metadata saved to {self.metadata_path}")

    def _load(self) -> None:
        """Load index and metadata from disk."""
        if self.index_path and Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
            logger.debug(f"FAISS index loaded from {self.index_path}")
        else:
            self._create_index()

        if self.metadata_path and Path(self.metadata_path).exists():
            with open(self.metadata_path, "rb") as f:
                metadata = pickle.load(f)
            self.documents = metadata["documents"]
            self.metadatas = metadata["metadatas"]
            self.id_to_idx = metadata["id_to_idx"]
            self.idx_to_id = metadata["idx_to_id"]
            logger.debug(f"FAISS metadata loaded from {self.metadata_path}")

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

