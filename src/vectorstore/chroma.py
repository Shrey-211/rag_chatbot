"""ChromaDB vector store implementation."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from src.vectorstore.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """Vector store implementation using ChromaDB.

    ChromaDB is a lightweight, persistent vector database with built-in
    embedding support and efficient similarity search.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        distance_metric: str = "cosine",
    ):
        """Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
            distance_metric: Distance metric (cosine, l2, ip)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric

        # Configure client
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.Client(settings=Settings(anonymized_telemetry=False))

        # Create or get collection
        metadata = {"hnsw:space": distance_metric}
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata=metadata
        )

        logger.info(
            f"ChromaDB initialized: collection={collection_name}, "
            f"persist_dir={persist_directory}, metric={distance_metric}"
        )

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

        # ChromaDB expects list of lists for embeddings
        embeddings_list = embeddings.tolist()

        try:
            logger.info(f"💾 Indexing {len(ids)} chunks into ChromaDB collection '{self.collection_name}'...")
            self.collection.upsert(
                ids=ids, embeddings=embeddings_list, documents=documents, metadatas=metadatas
            )
            total_count = self.collection.count()
            logger.info(f"   ✓ Successfully indexed {len(ids)} chunks. Total documents in store: {total_count}")
        except Exception as e:
            error_msg = str(e)
            # Check if this is a dimension mismatch error
            if "expecting embedding with dimension" in error_msg.lower() and "got" in error_msg.lower():
                logger.warning(f"⚠️  Embedding dimension mismatch detected!")
                logger.warning(f"   Error: {error_msg}")
                logger.warning(f"   This happens when you change the embedding model.")
                logger.warning(f"   Recreating collection '{self.collection_name}' with new dimensions...")
                
                # Delete the old collection
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info(f"   ✓ Deleted old collection")
                except Exception as del_error:
                    logger.warning(f"   Could not delete old collection: {del_error}")
                
                # Recreate the collection
                metadata = {"hnsw:space": self.distance_metric}
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name, metadata=metadata
                )
                logger.info(f"   ✓ Recreated collection with new embedding dimensions")
                
                # Retry the upsert
                logger.info(f"💾 Retrying: Indexing {len(ids)} chunks into ChromaDB collection '{self.collection_name}'...")
                self.collection.upsert(
                    ids=ids, embeddings=embeddings_list, documents=documents, metadatas=metadatas
                )
                total_count = self.collection.count()
                logger.info(f"   ✓ Successfully indexed {len(ids)} chunks. Total documents in store: {total_count}")
                logger.warning(f"   ⚠️  NOTE: All previous documents were deleted during collection recreation.")
                logger.warning(f"   ⚠️  You may need to re-upload any other documents you had indexed.")
            else:
                logger.error(f"ChromaDB upsert error: {e}")
                raise

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
        # Ensure query_embedding is 1D or 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_list = query_embedding.tolist()

        try:
            results = self.collection.query(
                query_embeddings=query_list, n_results=top_k, where=filter_dict
            )

            # Parse results
            search_results = []
            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    search_results.append(
                        SearchResult(
                            id=results["ids"][0][i],
                            content=results["documents"][0][i],
                            metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                            score=1.0
                            - results["distances"][0][i],  # Convert distance to similarity
                        )
                    )

            return search_results
        except Exception as e:
            logger.error(f"ChromaDB query error: {e}")
            raise

    def delete(self, ids: Optional[List[str]] = None, filter_dict: Optional[Dict] = None) -> None:
        """Delete documents.

        Args:
            ids: Document IDs to delete
            filter_dict: Metadata filter
        """
        try:
            if ids:
                self.collection.delete(ids=ids)
                logger.debug(f"Deleted {len(ids)} documents from ChromaDB")
            elif filter_dict:
                self.collection.delete(where=filter_dict)
                logger.debug(f"Deleted documents matching filter: {filter_dict}")
        except Exception as e:
            logger.error(f"ChromaDB delete error: {e}")
            raise

    def persist(self) -> None:
        """Persist to disk (automatic with PersistentClient)."""
        # ChromaDB PersistentClient auto-persists
        logger.debug("ChromaDB persisted")

    def count(self) -> int:
        """Get document count.

        Returns:
            Number of documents
        """
        return self.collection.count()

    def get_by_ids(self, ids: List[str]) -> List[SearchResult]:
        """Get documents by IDs.

        Args:
            ids: Document IDs

        Returns:
            List of search results
        """
        try:
            results = self.collection.get(ids=ids)

            search_results = []
            for i in range(len(results["ids"])):
                search_results.append(
                    SearchResult(
                        id=results["ids"][i],
                        content=results["documents"][i],
                        metadata=results["metadatas"][i] if results["metadatas"] else {},
                        score=1.0,
                    )
                )

            return search_results
        except Exception as e:
            logger.error(f"ChromaDB get_by_ids error: {e}")
            raise

