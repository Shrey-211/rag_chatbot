"""Tests for document extractors."""

import tempfile
from pathlib import Path

import pytest

from src.extractors.base import ExtractorFactory
from src.extractors.txt import TextExtractor


class TestTextExtractor:
    """Tests for text extractor."""

    def test_extract_text_file(self):
        """Test extracting from text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document.\nWith multiple lines.")
            temp_path = f.name

        try:
            extractor = TextExtractor()
            result = extractor.extract(temp_path)

            assert "This is a test document" in result.content
            assert result.metadata["extractor"] == "text"
            assert result.source == str(Path(temp_path).absolute())
        finally:
            Path(temp_path).unlink()

    def test_supports(self):
        """Test file support check."""
        extractor = TextExtractor()

        assert extractor.supports("test.txt") is True
        assert extractor.supports("test.md") is True
        assert extractor.supports("test.pdf") is False


class TestExtractorFactory:
    """Tests for extractor factory."""

    def test_get_extractor(self):
        """Test extractor selection."""
        factory = ExtractorFactory()

        txt_extractor = factory.get_extractor("test.txt")
        assert txt_extractor is not None
        assert isinstance(txt_extractor, TextExtractor)

        pdf_extractor = factory.get_extractor("test.pdf")
        assert pdf_extractor is not None

    def test_extract(self):
        """Test extraction through factory."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Factory test content")
            temp_path = f.name

        try:
            factory = ExtractorFactory()
            result = factory.extract(temp_path)

            assert "Factory test content" in result.content
        finally:
            Path(temp_path).unlink()

    def test_unsupported_file(self):
        """Test unsupported file type."""
        factory = ExtractorFactory()

        with pytest.raises(ValueError):
            factory.extract("test.xyz")

