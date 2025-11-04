"""Base interface for document extractors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExtractedDocument:
    """Standardized document extraction result."""

    content: str
    metadata: Dict[str, any]
    source: str


class DocumentExtractor(ABC):
    """Abstract base class for document extractors.

    Each extractor handles specific file types and converts them to plain text.
    """

    @abstractmethod
    def extract(self, file_path: str) -> ExtractedDocument:
        """Extract text content from a file.

        Args:
            file_path: Path to the file

        Returns:
            ExtractedDocument with content and metadata

        Raises:
            Exception: If extraction fails
        """
        pass

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Check if this extractor supports the given file.

        Args:
            file_path: Path to the file

        Returns:
            True if this extractor can handle the file
        """
        pass

    def get_file_info(self, file_path: str) -> Dict[str, any]:
        """Get basic file information.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file metadata
        """
        path = Path(file_path)
        return {
            "filename": path.name,
            "extension": path.suffix,
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "absolute_path": str(path.absolute()),
        }


class ExtractorFactory:
    """Factory for selecting appropriate extractor based on file type."""

    def __init__(self, extractors: Optional[List[DocumentExtractor]] = None):
        """Initialize factory with extractors.

        Args:
            extractors: List of extractor instances. If None, uses defaults.
        """
        if extractors is None:
            # Avoid circular import by importing here
            from src.extractors.docx import DocxExtractor
            from src.extractors.pdf import PDFExtractor
            from src.extractors.table import TableExtractor
            from src.extractors.txt import TextExtractor

            extractors = [
                TextExtractor(),
                PDFExtractor(),
                DocxExtractor(),
                TableExtractor(),
            ]
        self.extractors = extractors

    def get_extractor(self, file_path: str) -> Optional[DocumentExtractor]:
        """Get appropriate extractor for file.

        Args:
            file_path: Path to the file

        Returns:
            Matching extractor or None
        """
        for extractor in self.extractors:
            if extractor.supports(file_path):
                return extractor
        return None

    def extract(self, file_path: str) -> ExtractedDocument:
        """Extract content using appropriate extractor.

        Args:
            file_path: Path to the file

        Returns:
            ExtractedDocument

        Raises:
            ValueError: If no extractor supports the file
        """
        extractor = self.get_extractor(file_path)
        if extractor is None:
            raise ValueError(f"No extractor found for file: {file_path}")
        return extractor.extract(file_path)

