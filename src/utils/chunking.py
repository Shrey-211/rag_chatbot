"""Text chunking utilities."""

import logging
from typing import List

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separator: str = "\n\n",
) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
        separator: Separator to use for initial split

    Returns:
        List of text chunks
    """
    if not text or chunk_size <= 0:
        return []

    # First split by separator (paragraphs, sections, etc.)
    splits = text.split(separator)

    chunks = []
    current_chunk = []
    current_size = 0

    for split in splits:
        split = split.strip()
        if not split:
            continue

        split_size = len(split)

        # If this split alone is larger than chunk_size, split it further
        if split_size > chunk_size:
            # Add current chunk if it exists
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

            # Split large text by sentences or words
            sub_chunks = _split_large_text(split, chunk_size, chunk_overlap)
            chunks.extend(sub_chunks)
            continue

        # If adding this split exceeds chunk_size, finalize current chunk
        if current_size + split_size > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Add overlap from previous chunk
            if chunk_overlap > 0 and current_chunk:
                overlap_text = " ".join(current_chunk)
                if len(overlap_text) > chunk_overlap:
                    overlap_text = overlap_text[-chunk_overlap:]
                current_chunk = [overlap_text]
                current_size = len(overlap_text)
            else:
                current_chunk = []
                current_size = 0

        # Add split to current chunk
        current_chunk.append(split)
        current_size += split_size + 1  # +1 for space

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.debug(f"Chunked text into {len(chunks)} chunks")
    return chunks


def _split_large_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split large text that exceeds chunk_size.

    Args:
        text: Text to split
        chunk_size: Target chunk size
        chunk_overlap: Overlap size

    Returns:
        List of chunks
    """
    chunks = []
    words = text.split()

    current_chunk = []
    current_size = 0

    for word in words:
        word_size = len(word) + 1  # +1 for space

        if current_size + word_size > chunk_size and current_chunk:
            # Finalize current chunk
            chunks.append(" ".join(current_chunk))

            # Create overlap
            if chunk_overlap > 0:
                overlap_text = " ".join(current_chunk)
                if len(overlap_text) > chunk_overlap:
                    # Take last N characters
                    overlap_words = []
                    overlap_size = 0
                    for w in reversed(current_chunk):
                        if overlap_size + len(w) + 1 > chunk_overlap:
                            break
                        overlap_words.insert(0, w)
                        overlap_size += len(w) + 1
                    current_chunk = overlap_words
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
            else:
                current_chunk = []
                current_size = 0

        current_chunk.append(word)
        current_size += word_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

