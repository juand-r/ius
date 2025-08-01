"""
Simple chunking functions that respect text boundaries.

All chunking functions use delimiter-aware splitting to avoid breaking
words or sentences. Content preservation is guaranteed.
"""

from typing import List

from .utils import validate_chunks


def chunk_fixed_size(text: str, chunk_size: int, delimiter: str = "\n") -> List[str]:
    """
    Split text into chunks of approximately fixed size, respecting delimiter boundaries.

    Args:
        text: Input text to chunk
        chunk_size: Target size for each chunk (in characters)
        delimiter: Boundary delimiter to respect (default: newline)

    Returns:
        List of text chunks, each roughly chunk_size characters

    Note:
        Chunks may be slightly larger or smaller than chunk_size to respect
        delimiter boundaries and avoid splitting meaningful units.
    """
    if not text:
        return []

    if delimiter not in text:
        # No delimiters found, return whole text as single chunk
        return [text]

    # Split text into units separated by delimiter
    units = text.split(delimiter)
    chunks = []
    current_chunk = []
    current_size = 0

    for unit in units:
        unit_size = len(unit)
        # Add delimiter length except for first unit in chunk
        total_size = current_size + unit_size + (len(delimiter) if current_chunk else 0)

        if total_size <= chunk_size or not current_chunk:
            # Add unit to current chunk (always add at least one unit)
            current_chunk.append(unit)
            current_size = total_size
        else:
            # Current chunk is full, start new chunk
            chunks.append(delimiter.join(current_chunk))
            current_chunk = [unit]
            current_size = unit_size

    # Add remaining chunk if any
    if current_chunk:
        chunks.append(delimiter.join(current_chunk))

    # Validate content preservation
    if not validate_chunks(text, chunks, delimiter):
        raise ValueError("Content validation failed: chunks do not preserve original text")

    return chunks


def chunk_fixed_count(text: str, num_chunks: int, delimiter: str = "\n") -> List[str]:
    """
    Split text into a fixed number of chunks, respecting delimiter boundaries.

    Args:
        text: Input text to chunk
        num_chunks: Target number of chunks
        delimiter: Boundary delimiter to respect (default: newline)

    Returns:
        List of text chunks (may be fewer than num_chunks if not enough delimiters)
    """
    if not text:
        return []

    if num_chunks <= 1:
        return [text]

    if delimiter not in text:
        # No delimiters found, return whole text as single chunk
        return [text]

    # Split text into units separated by delimiter
    units = text.split(delimiter)

    if len(units) < num_chunks:
        # Not enough units to create requested number of chunks
        # Return each unit as separate chunk
        return [delimiter.join([unit]) if unit else unit for unit in units]

    # Calculate target units per chunk
    units_per_chunk = len(units) // num_chunks
    remainder = len(units) % num_chunks

    chunks = []
    start_idx = 0

    for i in range(num_chunks):
        # Distribute remainder units across first chunks
        chunk_size = units_per_chunk + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size

        chunk_units = units[start_idx:end_idx]
        chunks.append(delimiter.join(chunk_units))
        start_idx = end_idx

    # Validate content preservation
    if not validate_chunks(text, chunks, delimiter):
        raise ValueError("Content validation failed: chunks do not preserve original text")

    return chunks


def chunk_custom(
    text: str, strategy: str, delimiter: str = "\n", **kwargs
) -> List[str]:
    """
    Split text using custom strategy (placeholder for dataset-specific approaches).

    Args:
        text: Input text to chunk
        strategy: Custom strategy name (to be implemented later)
        delimiter: Boundary delimiter to respect (default: newline)
        **kwargs: Additional strategy-specific parameters

    Returns:
        List of text chunks

    Note:
        This is a placeholder for future dataset-specific chunking strategies.
        Currently falls back to fixed_size chunking.
    """
    raise NotImplementedError("Custom chunking strategy not implemented")
