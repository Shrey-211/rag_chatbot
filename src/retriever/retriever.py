"""Document retriever implementation."""

import logging
from typing import Dict, List, Optional

from src.adapters.embedding.base import EmbeddingAdapter
from src.vectorstore.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retriever for finding relevant documents.

    Handles the retrieval pipeline: query -> embed -> search -> results.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_adapter: EmbeddingAdapter,
        top_k: int = 5,
        min_score: float = 0.0,
    ):
        """Initialize retriever.

        Args:
            vector_store: Vector store to search
            embedding_adapter: Embedding adapter for queries
            top_k: Number of results to return
            min_score: Minimum relevance score threshold
        """
        self.vector_store = vector_store
        self.embedding_adapter = embedding_adapter
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Retrieve relevant documents for query.

        Args:
            query: Search query text
            top_k: Number of results (overrides default)
            filter_dict: Optional metadata filter

        Returns:
            List of search results
        """
        if top_k is None:
            top_k = self.top_k

        try:
            # Embed query
            query_embedding = self.embedding_adapter.embed_texts([query])[0]

            # Search vector store
            results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict,
            )

            # Filter by minimum score
            filtered_results = [r for r in results if r.score >= self.min_score]

            logger.debug(
                f"Retrieved {len(filtered_results)} results for query (filtered from {len(results)})"
            )
            return filtered_results

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

    def format_context(self, results: List[SearchResult], include_metadata: bool = False) -> str:
        """Format search results into context string.

        Args:
            results: Search results
            include_metadata: Whether to include metadata in context

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            if include_metadata:
                meta_str = ", ".join(f"{k}={v}" for k, v in result.metadata.items())
                context_parts.append(f"[{i}] ({meta_str})\n{result.content}")
            else:
                context_parts.append(f"[{i}] {result.content}")

        return "\n\n".join(context_parts)

    def get_statistics(self) -> Dict[str, any]:
        """Get retriever statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": self.vector_store.count(),
            "top_k": self.top_k,
            "min_score": self.min_score,
            "embedding_dim": self.embedding_adapter.get_embedding_dimension(),
        }

