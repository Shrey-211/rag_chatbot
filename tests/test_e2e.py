"""End-to-end integration tests."""

import numpy as np
import pytest

from src.adapters.embedding.local import LocalTextEmbeddingAdapter
from src.adapters.llm.mock import MockLLMAdapter
from src.retriever.retriever import Retriever
from src.utils.chunking import chunk_text
from src.vectorstore.memory import InMemoryVectorStore


class TestE2E:
    """End-to-end RAG pipeline tests."""

    @pytest.fixture
    def setup_rag_pipeline(self):
        """Set up complete RAG pipeline."""
        # Initialize components
        llm_adapter = MockLLMAdapter(
            response_text="Based on the context, the answer is that RAG combines retrieval and generation."
        )
        embedding_adapter = LocalTextEmbeddingAdapter(
            model_name="all-MiniLM-L6-v2", device="cpu"
        )
        vector_store = InMemoryVectorStore()
        retriever = Retriever(
            vector_store=vector_store, embedding_adapter=embedding_adapter, top_k=3
        )

        return {
            "llm": llm_adapter,
            "embedding": embedding_adapter,
            "vectorstore": vector_store,
            "retriever": retriever,
        }

    def test_index_and_retrieve(self, setup_rag_pipeline):
        """Test full indexing and retrieval pipeline."""
        components = setup_rag_pipeline

        # Index documents
        documents = [
            "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with text generation.",
            "Vector databases store embeddings for efficient similarity search.",
            "Python is a popular programming language for AI and machine learning.",
        ]

        # Chunk and embed
        all_chunks = []
        all_embeddings = []
        all_ids = []

        for i, doc in enumerate(documents):
            chunks = chunk_text(doc, chunk_size=100, chunk_overlap=20)
            embeddings = components["embedding"].embed_texts(chunks)

            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_embeddings.append(embeddings[j])
                all_ids.append(f"doc{i}_chunk{j}")

        # Upsert to vector store
        embeddings_array = np.array(all_embeddings)
        components["vectorstore"].upsert(all_ids, embeddings_array, all_chunks)

        assert components["vectorstore"].count() > 0

        # Query
        query = "What is RAG?"
        results = components["retriever"].retrieve(query, top_k=2)

        assert len(results) > 0
        assert any("RAG" in r.content or "Retrieval" in r.content for r in results)

    def test_rag_query_with_llm(self, setup_rag_pipeline):
        """Test complete RAG query with LLM generation."""
        components = setup_rag_pipeline

        # Index a simple document
        document = "The capital of France is Paris. Paris is known for the Eiffel Tower."
        chunks = chunk_text(document, chunk_size=50, chunk_overlap=10)
        embeddings = components["embedding"].embed_texts(chunks)
        ids = [f"chunk{i}" for i in range(len(chunks))]

        components["vectorstore"].upsert(ids, embeddings, chunks)

        # Query
        query = "What is the capital of France?"
        results = components["retriever"].retrieve(query, top_k=2)

        # Format context
        context = components["retriever"].format_context(results)

        # Generate answer
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        response = components["llm"].generate(prompt)

        assert response.text is not None
        assert len(response.text) > 0

    def test_retriever_statistics(self, setup_rag_pipeline):
        """Test retriever statistics."""
        components = setup_rag_pipeline

        # Add some documents
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        embeddings = components["embedding"].embed_texts(documents)
        ids = [f"doc{i}" for i in range(len(documents))]

        components["vectorstore"].upsert(ids, embeddings, documents)

        # Get statistics
        stats = components["retriever"].get_statistics()

        assert stats["total_documents"] == 3
        assert stats["embedding_dim"] == 384  # all-MiniLM-L6-v2
        assert "top_k" in stats

