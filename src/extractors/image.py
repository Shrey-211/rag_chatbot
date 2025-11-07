"""Image file extractor with OCR support."""

import logging
from pathlib import Path
from typing import Optional

from PIL import Image
import pytesseract

from src.extractors.base import DocumentExtractor, ExtractedDocument

logger = logging.getLogger(__name__)


class ImageExtractor(DocumentExtractor):
    """Extractor for image files using OCR (pytesseract).
    
    Supports common image formats like JPG, PNG, TIFF, BMP, etc.
    Uses Tesseract OCR to extract text from images.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp"}

    def __init__(self, language: str = "eng", dpi: Optional[int] = None):
        """Initialize image extractor.
        
        Args:
            language: Tesseract language code (default: 'eng' for English).
                     Can be 'eng+ara' for multiple languages.
            dpi: DPI for image processing. If None, uses image's native DPI.
        """
        self.language = language
        self.dpi = dpi

    def extract(self, file_path: str) -> ExtractedDocument:
        """Extract text from image file using OCR.
        
        Args:
            file_path: Path to image file
            
        Returns:
            ExtractedDocument with OCR-extracted text
        """
        path = Path(file_path)
        
        try:
            # Open image
            image = Image.open(str(path))
            
            # Get image info
            width, height = image.size
            image_format = image.format
            mode = image.mode
            
            # Convert to RGB if necessary (some formats like PNG with transparency)
            if mode not in ('RGB', 'L'):
                logger.info(f"Converting image from {mode} to RGB")
                image = image.convert('RGB')
            
            # Perform OCR
            logger.info(f"Performing OCR on {path.name} with language '{self.language}'")
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 1'  # OEM 3: Default, PSM 1: Automatic page segmentation with OSD
            
            if self.dpi:
                # Set DPI if specified
                text = pytesseract.image_to_string(
                    image,
                    lang=self.language,
                    config=custom_config
                )
            else:
                text = pytesseract.image_to_string(
                    image,
                    lang=self.language,
                    config=custom_config
                )
            
            # Get OCR confidence data
            try:
                ocr_data = pytesseract.image_to_data(image, lang=self.language, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in ocr_data['conf'] if conf != '-1']
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            except Exception as e:
                logger.warning(f"Could not get OCR confidence: {e}")
                avg_confidence = None
            
            # Clean up extracted text
            content = text.strip()
            
            if not content:
                logger.warning(f"No text extracted from image {path.name}. Image might be blank or contain no text.")
            
            # Prepare metadata
            metadata = self.get_file_info(file_path)
            metadata.update({
                "extractor": "image_ocr",
                "image_format": image_format,
                "image_width": width,
                "image_height": height,
                "image_mode": mode,
                "ocr_language": self.language,
                "ocr_confidence": avg_confidence,
            })
            
            return ExtractedDocument(
                content=content,
                metadata=metadata,
                source=str(path.absolute())
            )
            
        except Exception as e:
            logger.error(f"Error extracting image {file_path}: {e}")
            raise
        finally:
            # Clean up
            try:
                image.close()
            except:
                pass

    def supports(self, file_path: str) -> bool:
        """Check if file is a supported image format.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is a supported image format
        """
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

