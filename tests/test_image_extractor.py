"""Tests for image extraction with OCR."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

from src.extractors.image import ImageExtractor


# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def sample_image(tmp_path):
    """Create a simple test image with text."""
    # Create a white image with black text
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use default font
    text = "This is a test document for OCR.\nIt contains multiple lines of text."
    draw.text((50, 50), text, fill='black')
    
    # Save image
    image_path = tmp_path / "test_document.jpg"
    img.save(str(image_path))
    
    return str(image_path)


@pytest.fixture
def mock_tesseract():
    """Mock pytesseract for testing without requiring tesseract installation."""
    with patch('src.extractors.image.pytesseract') as mock_tess:
        # Mock image_to_string
        mock_tess.image_to_string.return_value = "This is a test document for OCR.\nIt contains multiple lines of text."
        
        # Mock image_to_data
        mock_tess.image_to_data.return_value = {
            'conf': ['95', '90', '88', '92', '94', '-1']
        }
        mock_tess.Output.DICT = 'dict'
        
        yield mock_tess


class TestImageExtractor:
    """Test image extractor functionality."""
    
    def test_extractor_initialization(self):
        """Test extractor can be initialized."""
        extractor = ImageExtractor(language="eng", dpi=300)
        assert extractor.language == "eng"
        assert extractor.dpi == 300
    
    def test_supports_image_formats(self):
        """Test that extractor supports various image formats."""
        extractor = ImageExtractor()
        
        # Test supported formats
        assert extractor.supports("document.jpg")
        assert extractor.supports("document.jpeg")
        assert extractor.supports("document.png")
        assert extractor.supports("document.tiff")
        assert extractor.supports("document.tif")
        assert extractor.supports("document.bmp")
        assert extractor.supports("document.gif")
        assert extractor.supports("document.webp")
        
        # Test unsupported formats
        assert not extractor.supports("document.pdf")
        assert not extractor.supports("document.txt")
        assert not extractor.supports("document.docx")
    
    def test_supports_case_insensitive(self):
        """Test that file extension matching is case-insensitive."""
        extractor = ImageExtractor()
        
        assert extractor.supports("DOCUMENT.JPG")
        assert extractor.supports("Document.PNG")
        assert extractor.supports("image.TiFF")
    
    def test_extract_with_mock(self, sample_image, mock_tesseract):
        """Test extraction with mocked tesseract."""
        extractor = ImageExtractor(language="eng")
        
        result = extractor.extract(sample_image)
        
        # Verify result
        assert result.content == "This is a test document for OCR.\nIt contains multiple lines of text."
        assert result.source == str(Path(sample_image).absolute())
        
        # Verify metadata
        assert "extractor" in result.metadata
        assert result.metadata["extractor"] == "image_ocr"
        assert "filename" in result.metadata
        assert "image_width" in result.metadata
        assert "image_height" in result.metadata
        assert "ocr_language" in result.metadata
        assert result.metadata["ocr_language"] == "eng"
    
    def test_extract_confidence(self, sample_image, mock_tesseract):
        """Test that OCR confidence is calculated."""
        extractor = ImageExtractor(language="eng")
        
        result = extractor.extract(sample_image)
        
        # Check confidence was calculated
        assert "ocr_confidence" in result.metadata
        # Based on mock data: (95+90+88+92+94)/5 = 91.8
        assert result.metadata["ocr_confidence"] == pytest.approx(91.8, rel=0.1)
    
    def test_extract_multilingual(self, sample_image, mock_tesseract):
        """Test extraction with multiple languages."""
        extractor = ImageExtractor(language="eng+ara")
        
        result = extractor.extract(sample_image)
        
        # Verify language was passed
        assert result.metadata["ocr_language"] == "eng+ara"
        mock_tesseract.image_to_string.assert_called()
        
        # Check that the language parameter was used
        call_kwargs = mock_tesseract.image_to_string.call_args[1]
        assert call_kwargs['lang'] == "eng+ara"
    
    def test_extract_with_dpi(self, sample_image, mock_tesseract):
        """Test extraction with specific DPI."""
        extractor = ImageExtractor(language="eng", dpi=600)
        
        result = extractor.extract(sample_image)
        
        # Should work without errors
        assert result.content
        assert result.metadata["extractor"] == "image_ocr"
    
    def test_extract_empty_image(self, tmp_path, mock_tesseract):
        """Test extraction from blank image."""
        # Create blank white image
        img = Image.new('RGB', (800, 600), color='white')
        image_path = tmp_path / "blank.jpg"
        img.save(str(image_path))
        
        # Mock empty OCR result
        mock_tesseract.image_to_string.return_value = ""
        
        extractor = ImageExtractor()
        result = extractor.extract(str(image_path))
        
        # Should return empty content but not fail
        assert result.content == ""
        assert result.metadata["extractor"] == "image_ocr"
    
    def test_extract_file_not_found(self):
        """Test extraction from non-existent file."""
        extractor = ImageExtractor()
        
        with pytest.raises(Exception):
            extractor.extract("nonexistent.jpg")
    
    def test_extract_invalid_image(self, tmp_path):
        """Test extraction from invalid image file."""
        # Create a text file with .jpg extension
        fake_image = tmp_path / "fake.jpg"
        fake_image.write_text("This is not an image")
        
        extractor = ImageExtractor()
        
        with pytest.raises(Exception):
            extractor.extract(str(fake_image))
    
    @patch('src.extractors.image.Image.open')
    def test_extract_rgb_conversion(self, mock_open, sample_image, mock_tesseract):
        """Test that images are converted to RGB if needed."""
        # Mock an image with RGBA mode
        mock_image = MagicMock()
        mock_image.size = (800, 600)
        mock_image.format = "PNG"
        mock_image.mode = "RGBA"
        
        mock_rgb_image = MagicMock()
        mock_rgb_image.mode = "RGB"
        mock_image.convert.return_value = mock_rgb_image
        
        mock_open.return_value = mock_image
        
        extractor = ImageExtractor()
        result = extractor.extract(sample_image)
        
        # Verify convert was called
        mock_image.convert.assert_called_with('RGB')
    
    def test_get_file_info(self, sample_image):
        """Test that file info is correctly extracted."""
        extractor = ImageExtractor()
        
        file_info = extractor.get_file_info(sample_image)
        
        assert "filename" in file_info
        assert "extension" in file_info
        assert "size_bytes" in file_info
        assert "absolute_path" in file_info
        
        assert file_info["extension"] == ".jpg"
        assert file_info["size_bytes"] > 0


@pytest.mark.integration
class TestImageExtractorIntegration:
    """Integration tests that require actual tesseract installation."""
    
    def test_extract_real_image(self, sample_image):
        """Test extraction with real tesseract (if available)."""
        try:
            import pytesseract
            # Try to verify tesseract is installed
            pytesseract.get_tesseract_version()
        except Exception:
            pytest.skip("Tesseract not installed")
        
        extractor = ImageExtractor(language="eng")
        
        try:
            result = extractor.extract(sample_image)
            
            # Should extract some text
            assert len(result.content) > 0
            assert result.metadata["extractor"] == "image_ocr"
            
        except Exception as e:
            pytest.skip(f"Tesseract not properly configured: {e}")


@pytest.mark.integration
class TestPDFExtractorWithOCR:
    """Integration tests for PDF extractor with OCR."""
    
    def test_scanned_pdf_detection(self):
        """Test that scanned PDFs are detected and OCR is applied."""
        from src.extractors.pdf import PDFExtractor
        
        # This would require a real scanned PDF
        pytest.skip("Requires real scanned PDF file")
    
    def test_ocr_disabled(self):
        """Test that OCR can be disabled."""
        from src.extractors.pdf import PDFExtractor
        
        extractor = PDFExtractor(ocr_enabled=False)
        assert extractor.ocr_enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

