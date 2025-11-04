"""Table file extractor for CSV and JSON."""

import json
import logging
from pathlib import Path

import pandas as pd

from src.extractors.base import DocumentExtractor, ExtractedDocument

logger = logging.getLogger(__name__)


class TableExtractor(DocumentExtractor):
    """Extractor for tabular data (CSV, JSON)."""

    SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl"}

    def extract(self, file_path: str) -> ExtractedDocument:
        """Extract text from tabular data files.

        Args:
            file_path: Path to CSV or JSON file

        Returns:
            ExtractedDocument with formatted table content
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        try:
            # Read into pandas DataFrame
            if extension == ".csv":
                df = pd.read_csv(path)
            elif extension == ".json":
                df = pd.read_json(path)
            elif extension == ".jsonl":
                df = pd.read_json(path, lines=True)
            else:
                raise ValueError(f"Unsupported extension: {extension}")

            # Convert to readable text format
            # Include column names and row data
            content_parts = [f"Table with {len(df)} rows and {len(df.columns)} columns"]
            content_parts.append(f"Columns: {', '.join(df.columns)}")
            content_parts.append("")

            # Convert each row to text
            for idx, row in df.iterrows():
                row_text = " | ".join(f"{col}: {val}" for col, val in row.items())
                content_parts.append(row_text)

            content = "\n".join(content_parts)

            # Build metadata
            metadata = self.get_file_info(file_path)
            metadata["extractor"] = "table"
            metadata["num_rows"] = len(df)
            metadata["num_columns"] = len(df.columns)
            metadata["columns"] = list(df.columns)

            return ExtractedDocument(
                content=content, metadata=metadata, source=str(path.absolute())
            )
        except Exception as e:
            logger.error(f"Error extracting table from {file_path}: {e}")
            raise

    def supports(self, file_path: str) -> bool:
        """Check if file is a supported table format.

        Args:
            file_path: Path to file

        Returns:
            True if file extension is supported
        """
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

