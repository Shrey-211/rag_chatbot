"""Tests for LLM adapters."""

import pytest

from src.adapters.llm.mock import MockLLMAdapter


class TestMockLLMAdapter:
    """Tests for mock LLM adapter."""

    def test_generate(self):
        """Test basic generation."""
        adapter = MockLLMAdapter(response_text="Test response")
        response = adapter.generate("What is AI?")

        assert response.text is not None
        assert "Test response" in response.text
        assert response.model == "mock-model"
        assert response.usage is not None

    def test_stream_generate(self):
        """Test streaming generation."""
        adapter = MockLLMAdapter(response_text="Hello world")
        chunks = []

        def callback(chunk):
            chunks.append(chunk)

        response = adapter.stream_generate("Test prompt", callback)

        assert len(chunks) > 0
        assert response.text == "Hello world"
        assert response.metadata.get("streaming") is True

    def test_token_usage(self):
        """Test token usage extraction."""
        adapter = MockLLMAdapter()
        response = adapter.generate("Test prompt")
        usage = adapter.token_usage(response)

        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_health_check(self):
        """Test health check."""
        adapter = MockLLMAdapter()
        assert adapter.health_check() is True

        adapter_fail = MockLLMAdapter(fail_health_check=True)
        assert adapter_fail.health_check() is False

    def test_get_model_info(self):
        """Test model info retrieval."""
        adapter = MockLLMAdapter()
        info = adapter.get_model_info()

        assert info["model"] == "mock-model"
        assert info["provider"] == "mock"

