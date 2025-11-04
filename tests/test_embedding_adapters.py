"""Tests for embedding adapters."""

import numpy as np
import pytest

from src.adapters.embedding.local import LocalTextEmbeddingAdapter


class TestLocalEmbeddingAdapter:
    """Tests for local embedding adapter."""

    @pytest.fixture
    def adapter(self):
        """Create test adapter."""
        return LocalTextEmbeddingAdapter(model_name="all-MiniLM-L6-v2", device="cpu")

    def test_embed_texts(self, adapter):
        """Test text embedding."""
        texts = ["Hello world", "This is a test"]
        embeddings = adapter.embed_texts(texts)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == adapter.get_embedding_dimension()
        assert isinstance(embeddings, np.ndarray)

    def test_embed_empty_list(self, adapter):
        """Test embedding empty list."""
        embeddings = adapter.embed_texts([])
        assert embeddings.shape[0] == 0

    def test_embed_batch(self, adapter):
        """Test batch embedding."""
        texts = [f"Text {i}" for i in range(10)]
        batches = list(adapter.embed_batch(iter(texts), batch_size=3))

        assert len(batches) > 0
        total_embeddings = sum(batch.shape[0] for batch in batches)
        assert total_embeddings == 10

    def test_get_embedding_dimension(self, adapter):
        """Test dimension retrieval."""
        dim = adapter.get_embedding_dimension()
        assert dim == 384  # all-MiniLM-L6-v2 dimension

    def test_health_check(self, adapter):
        """Test health check."""
        assert adapter.health_check() is True

