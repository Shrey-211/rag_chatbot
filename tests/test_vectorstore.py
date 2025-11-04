"""Tests for vector stores."""

import numpy as np
import pytest

from src.vectorstore.memory import InMemoryVectorStore


class TestInMemoryVectorStore:
    """Tests for in-memory vector store."""

    @pytest.fixture
    def store(self):
        """Create test store."""
        return InMemoryVectorStore()

    def test_upsert_and_count(self, store):
        """Test upserting documents."""
        ids = ["doc1", "doc2"]
        embeddings = np.random.rand(2, 128)
        documents = ["First document", "Second document"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]

        store.upsert(ids, embeddings, documents, metadatas)

        assert store.count() == 2

    def test_query(self, store):
        """Test querying for similar documents."""
        ids = ["doc1", "doc2", "doc3"]
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        documents = ["First", "Second", "Third"]

        store.upsert(ids, embeddings, documents)

        # Query with vector similar to first document
        query_emb = np.array([1.0, 0.0])
        results = store.query(query_emb, top_k=2)

        assert len(results) == 2
        assert results[0].id == "doc1"  # Most similar
        assert results[0].score > results[1].score

    def test_query_with_filter(self, store):
        """Test querying with metadata filter."""
        ids = ["doc1", "doc2"]
        embeddings = np.random.rand(2, 128)
        documents = ["First", "Second"]
        metadatas = [{"type": "A"}, {"type": "B"}]

        store.upsert(ids, embeddings, documents, metadatas)

        query_emb = np.random.rand(128)
        results = store.query(query_emb, top_k=10, filter_dict={"type": "A"})

        assert len(results) == 1
        assert results[0].id == "doc1"

    def test_delete(self, store):
        """Test deleting documents."""
        ids = ["doc1", "doc2"]
        embeddings = np.random.rand(2, 128)
        documents = ["First", "Second"]

        store.upsert(ids, embeddings, documents)
        assert store.count() == 2

        store.delete(ids=["doc1"])
        assert store.count() == 1

    def test_get_by_ids(self, store):
        """Test retrieving documents by IDs."""
        ids = ["doc1", "doc2"]
        embeddings = np.random.rand(2, 128)
        documents = ["First", "Second"]

        store.upsert(ids, embeddings, documents)

        results = store.get_by_ids(["doc1"])
        assert len(results) == 1
        assert results[0].content == "First"

    def test_persist(self, store):
        """Test persist (no-op for in-memory)."""
        store.persist()  # Should not raise

