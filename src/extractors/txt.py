"""Text file extractor."""

import logging
from pathlib import Path

from src.extractors.base import DocumentExtractor, ExtractedDocument

logger = logging.getLogger(__name__)


class TextExtractor(DocumentExtractor):
    """Extractor for plain text files (.txt, .md, .log, etc.)."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".log", ".text", ".rst"}

    def extract(self, file_path: str) -> ExtractedDocument:
        """Extract text from plain text file.

        Args:
            file_path: Path to text file

        Returns:
            ExtractedDocument with file content
        """
        path = Path(file_path)

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            metadata = self.get_file_info(file_path)
            metadata["extractor"] = "text"

            return ExtractedDocument(
                content=content, metadata=metadata, source=str(path.absolute())
            )
        except UnicodeDecodeError:
            # Try with different encoding
            logger.warning(f"UTF-8 failed for {file_path}, trying latin-1")
            with open(path, "r", encoding="latin-1") as f:
                content = f.read()

            metadata = self.get_file_info(file_path)
            metadata["extractor"] = "text"
            metadata["encoding"] = "latin-1"

            return ExtractedDocument(
                content=content, metadata=metadata, source=str(path.absolute())
            )
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise

    def supports(self, file_path: str) -> bool:
        """Check if file is a supported text file.

        Args:
            file_path: Path to file

        Returns:
            True if file extension is supported
        """
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

