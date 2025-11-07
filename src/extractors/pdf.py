"""PDF file extractor with OCR support for scanned documents."""

import logging
import os
from pathlib import Path
from typing import Optional

from PyPDF2 import PdfReader

from src.extractors.base import DocumentExtractor, ExtractedDocument

logger = logging.getLogger(__name__)

# Import OCR dependencies with fallback
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not installed. Image processing will be disabled.")

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed. OCR will be disabled.")

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not installed. PDF to image conversion will be disabled.")


class PDFExtractor(DocumentExtractor):
    """Extractor for PDF files with automatic OCR fallback for scanned documents.
    
    This extractor:
    1. First attempts to extract text normally from the PDF
    2. If text extraction yields little/no content, uses OCR on page images
    3. Automatically detects if a PDF is scanned and applies OCR accordingly
    """

    def __init__(
        self,
        ocr_enabled: bool = True,
        ocr_language: str = "eng",
        dpi: int = 300,
        min_text_threshold: int = 50,
        poppler_path: Optional[str] = None,
        tesseract_path: Optional[str] = None
    ):
        """Initialize PDF extractor.
        
        Args:
            ocr_enabled: Whether to enable OCR fallback for scanned PDFs
            ocr_language: Tesseract language code (default: 'eng' for English)
            dpi: DPI for converting PDF pages to images for OCR (default: 300)
            min_text_threshold: Minimum characters per page to consider it has text.
                               If below this, OCR is used.
            poppler_path: Optional path to poppler binaries (Windows only)
            tesseract_path: Optional path to tesseract executable
        """
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.dpi = dpi
        self.min_text_threshold = min_text_threshold
        self.poppler_path = poppler_path
        self.tesseract_path = tesseract_path
        
        # Configure tesseract path if provided
        if tesseract_path and PYTESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Check if OCR dependencies are available
        self._check_ocr_availability()

    def extract(self, file_path: str) -> ExtractedDocument:
        """Extract text from PDF file with OCR fallback.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ExtractedDocument with extracted text
        """
        path = Path(file_path)
        
        try:
            reader = PdfReader(str(path))
            num_pages = len(reader.pages)
            
            # First attempt: Try normal text extraction
            text_parts = []
            pages_with_text = 0
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > self.min_text_threshold:
                    text_parts.append(text)
                    pages_with_text += 1
                else:
                    text_parts.append("")  # Placeholder for OCR later
            
            # Determine if OCR is needed
            ocr_used = False
            pages_ocr = 0
            
            if self.ocr_enabled and pages_with_text < num_pages * 0.5:
                # If less than 50% of pages have text, this might be a scanned PDF
                logger.info(f"PDF appears to be scanned ({pages_with_text}/{num_pages} pages with text). Using OCR...")
                
                # Convert PDF to images and perform OCR
                ocr_text_parts = self._ocr_pdf(file_path)
                
                # Replace empty pages with OCR text
                for i, (original_text, ocr_text) in enumerate(zip(text_parts, ocr_text_parts)):
                    if len(original_text.strip()) <= self.min_text_threshold and ocr_text:
                        text_parts[i] = ocr_text
                        pages_ocr += 1
                        ocr_used = True
            
            content = "\n\n".join(filter(None, text_parts))
            
            if not content:
                logger.warning(f"No text extracted from PDF {path.name}. It might be blank or corrupted.")
            
            # Extract metadata
            metadata = self.get_file_info(file_path)
            metadata["extractor"] = "pdf"
            metadata["num_pages"] = num_pages
            metadata["ocr_enabled"] = self.ocr_enabled
            metadata["ocr_used"] = ocr_used
            metadata["pages_with_ocr"] = pages_ocr
            
            # Add PDF metadata if available
            if reader.metadata:
                pdf_meta = reader.metadata
                metadata["title"] = pdf_meta.get("/Title", "")
                metadata["author"] = pdf_meta.get("/Author", "")
                metadata["subject"] = pdf_meta.get("/Subject", "")
                metadata["creator"] = pdf_meta.get("/Creator", "")
            
            return ExtractedDocument(
                content=content,
                metadata=metadata,
                source=str(path.absolute())
            )
            
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {e}")
            raise

    def _check_ocr_availability(self):
        """Check if OCR dependencies are available and log status."""
        if not self.ocr_enabled:
            return
        
        missing_deps = []
        
        if not PIL_AVAILABLE:
            missing_deps.append("Pillow")
        if not PYTESSERACT_AVAILABLE:
            missing_deps.append("pytesseract")
        if not PDF2IMAGE_AVAILABLE:
            missing_deps.append("pdf2image")
        
        if missing_deps:
            logger.warning(f"OCR dependencies missing: {', '.join(missing_deps)}")
            logger.warning("Scanned PDFs will not be processed. Install with: pip install Pillow pytesseract pdf2image")
            self.ocr_enabled = False
        else:
            # Check if Tesseract and Poppler are actually installed
            try:
                pytesseract.get_tesseract_version()
            except Exception as e:
                logger.warning(f"Tesseract not found in PATH: {e}")
                logger.warning("Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
                self.ocr_enabled = False
    
    def _ocr_pdf(self, file_path: str) -> list:
        """Convert PDF pages to images and perform OCR.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of OCR text for each page
        """
        if not (PDF2IMAGE_AVAILABLE and PYTESSERACT_AVAILABLE and PIL_AVAILABLE):
            logger.error("OCR dependencies not available")
            return []
        
        ocr_texts = []
        
        try:
            # Convert PDF to images
            logger.info(f"Converting PDF pages to images at {self.dpi} DPI...")
            
            # Prepare kwargs for pdf2image
            convert_kwargs = {
                'pdf_path': file_path,
                'dpi': self.dpi,
                'fmt': 'png'
            }
            
            # Add poppler_path if provided (Windows)
            if self.poppler_path:
                convert_kwargs['poppler_path'] = self.poppler_path
            
            images = pdf2image.convert_from_path(**convert_kwargs)
            
            logger.info(f"Performing OCR on {len(images)} pages...")
            
            # Perform OCR on each page
            for page_num, image in enumerate(images, 1):
                try:
                    logger.debug(f"OCR on page {page_num}/{len(images)}")
                    
                    # Configure Tesseract
                    custom_config = r'--oem 3 --psm 1'
                    
                    text = pytesseract.image_to_string(
                        image,
                        lang=self.ocr_language,
                        config=custom_config
                    )
                    
                    ocr_texts.append(text.strip())
                    
                except Exception as e:
                    logger.warning(f"Error during OCR on page {page_num}: {e}")
                    ocr_texts.append("")
            
            return ocr_texts
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error during PDF to image conversion: {error_msg}")
            
            # Provide helpful error messages
            if "Unable to get page count" in error_msg or "poppler" in error_msg.lower():
                logger.error("╔══════════════════════════════════════════════════════════════╗")
                logger.error("║ Poppler is not installed or not in PATH                      ║")
                logger.error("╚══════════════════════════════════════════════════════════════╝")
                logger.error("")
                logger.error("Windows: Run scripts/setup_windows_ocr.ps1 as Administrator")
                logger.error("Linux:   sudo apt-get install poppler-utils")
                logger.error("macOS:   brew install poppler")
                logger.error("")
                logger.error("See: docs/windows_setup_guide.md for details")
            
            # Return empty list if OCR fails
            return []

    def supports(self, file_path: str) -> bool:
        """Check if file is a PDF.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is PDF
        """
        return Path(file_path).suffix.lower() == ".pdf"
