"""Embedding adapter implementations."""

from src.adapters.embedding.base import EmbeddingAdapter
from src.adapters.embedding.local import LocalTextEmbeddingAdapter
from src.adapters.embedding.openai import OpenAIEmbeddingAdapter

__all__ = ["EmbeddingAdapter", "LocalTextEmbeddingAdapter", "OpenAIEmbeddingAdapter"]

