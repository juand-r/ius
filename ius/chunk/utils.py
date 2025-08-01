"""
Utilities for chunking validation and analysis.
"""



def validate_chunks(
    original_text: str, chunks: list[str], delimiter: str = "\n"
) -> bool:
    """
    Verify that chunks preserve all content when joined with delimiter.

    Args:
        original_text: Original text before chunking
        chunks: List of text chunks
        delimiter: Delimiter used for chunking

    Returns:
        True if chunks preserve all original content, False otherwise
    """
    if not chunks:
        return not original_text

    reconstructed = delimiter.join(chunks)
    return reconstructed == original_text


def analyze_chunks(chunks: list[str], delimiter: str = "\n") -> dict:
    """
    Analyze chunk statistics for debugging and optimization.

    Args:
        chunks: List of text chunks
        delimiter: Delimiter used for chunking

    Returns:
        Dictionary with chunk analysis statistics
    """
    if not chunks:
        return {
            "num_chunks": 0,
            "total_chars": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "size_std": 0,
        }

    chunk_sizes = [len(chunk) for chunk in chunks]
    total_chars = sum(chunk_sizes)
    avg_size = total_chars / len(chunks)

    # Calculate standard deviation
    variance = sum((size - avg_size) ** 2 for size in chunk_sizes) / len(chunks)
    std_dev = variance**0.5

    return {
        "num_chunks": len(chunks),
        "total_chars": total_chars,
        "avg_chunk_size": round(avg_size, 1),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes),
        "size_std": round(std_dev, 1),
        "delimiter": repr(delimiter),
    }


def preview_chunks(chunks: list[str], max_preview: int = 100) -> list[str]:
    """
    Create preview of chunks for debugging (truncated for readability).

    Args:
        chunks: List of text chunks
        max_preview: Maximum characters to show per chunk

    Returns:
        List of truncated chunk previews
    """
    previews = []
    for i, chunk in enumerate(chunks):
        preview = chunk if len(chunk) <= max_preview else chunk[:max_preview] + "..."

        # Replace newlines with visible characters for debugging
        preview = preview.replace("\n", "\\n").replace("\t", "\\t")
        previews.append(f"Chunk {i + 1}: {preview}")

    return previews
