"""PDF file extractor."""

import logging
from pathlib import Path

from PyPDF2 import PdfReader

from src.extractors.base import DocumentExtractor, ExtractedDocument

logger = logging.getLogger(__name__)


class PDFExtractor(DocumentExtractor):
    """Extractor for PDF files using PyPDF2."""

    def extract(self, file_path: str) -> ExtractedDocument:
        """Extract text from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            ExtractedDocument with extracted text
        """
        path = Path(file_path)

        try:
            reader = PdfReader(str(path))

            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            content = "\n\n".join(text_parts)

            # Extract metadata
            metadata = self.get_file_info(file_path)
            metadata["extractor"] = "pdf"
            metadata["num_pages"] = len(reader.pages)

            # Add PDF metadata if available
            if reader.metadata:
                pdf_meta = reader.metadata
                metadata["title"] = pdf_meta.get("/Title", "")
                metadata["author"] = pdf_meta.get("/Author", "")
                metadata["subject"] = pdf_meta.get("/Subject", "")
                metadata["creator"] = pdf_meta.get("/Creator", "")

            return ExtractedDocument(
                content=content, metadata=metadata, source=str(path.absolute())
            )
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {e}")
            raise

    def supports(self, file_path: str) -> bool:
        """Check if file is a PDF.

        Args:
            file_path: Path to file

        Returns:
            True if file is PDF
        """
        return Path(file_path).suffix.lower() == ".pdf"

