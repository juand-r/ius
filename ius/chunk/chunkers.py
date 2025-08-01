"""
Simple chunking functions that respect text boundaries.

All chunking functions use delimiter-aware splitting to avoid breaking
words or sentences. Content preservation is guaranteed.
"""

from typing import Any, Dict, List, Optional

from .utils import analyze_chunks, validate_chunks


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


def _apply_chunking_strategy(
    text: str,
    strategy: str,
    chunk_size: Optional[int],
    num_chunks: Optional[int],
    delimiter: str,
) -> List[str]:
    """
    Apply the specified chunking strategy to text.

    Args:
        text: Text to chunk
        strategy: Chunking strategy
        chunk_size: Target chunk size (for fixed_size)
        num_chunks: Target number of chunks (for fixed_count)
        delimiter: Delimiter for chunking

    Returns:
        List of text chunks

    Raises:
        ValueError: If strategy parameters are invalid
    """
    if strategy == "fixed_size":
        if not chunk_size:
            raise ValueError("chunk_size required for fixed_size strategy")
        return chunk_fixed_size(text, chunk_size, delimiter)

    elif strategy == "fixed_count":
        if not num_chunks:
            raise ValueError("num_chunks required for fixed_count strategy")
        return chunk_fixed_count(text, num_chunks, delimiter)

    elif strategy == "custom":
        return chunk_custom(text, "default", delimiter)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def process_dataset_items(
    items: Dict[str, Any],
    strategy: str,
    document_handling: str = "chunk-individual-docs",
    chunk_size: Optional[int] = None,
    num_chunks: Optional[int] = None,
    delimiter: str = "\n",
) -> Dict[str, Any]:
    """
    Core logic to chunk all items in a dataset.

    Args:
        items: Dictionary of item_id -> item_data
        strategy: Chunking strategy ('fixed_size', 'fixed_count', 'custom')
        document_handling: How to handle multiple documents per item:
            - "chunk-individual-docs": Chunk each document separately
            - "chunk-concatenated-docs": Concatenate documents then chunk
        chunk_size: Target chunk size (for fixed_size strategy)
        num_chunks: Number of chunks (for fixed_count strategy)
        delimiter: Boundary delimiter for splitting

    Returns:
        Dictionary of item_id -> chunking results

    Raises:
        ValueError: If strategy parameters are invalid
    """
    results = {}
    errors = {}

    # Validate document_handling parameter
    if document_handling not in ["chunk-individual-docs", "chunk-concatenated-docs"]:
        raise ValueError(f"Invalid document_handling: {document_handling}")

    for item_id, item_data in items.items():
        try:
            # Extract documents
            if not isinstance(item_data, dict) or 'documents' not in item_data:
                raise ValueError("Item missing required 'documents' field")

            documents = item_data['documents']
            if not documents:
                raise ValueError("No documents found")

            if document_handling == "chunk-individual-docs":
                # Chunk each document separately
                chunks_array = []
                total_original_length = 0

                for doc in documents:
                    doc_id = doc['doc_id']
                    doc_text = doc['content']
                    if not doc_text:
                        raise ValueError("Document content is empty")

                    # Apply chunking strategy to this document
                    doc_chunks = _apply_chunking_strategy(
                        doc_text, strategy, chunk_size, num_chunks, delimiter
                    )

                    # Analyze chunks for this document
                    doc_stats = analyze_chunks(doc_chunks, delimiter)

                    chunks_array.append({
                        "document_id": doc_id,
                        "chunks": doc_chunks,
                        "stats": doc_stats,
                    })
                    total_original_length += len(doc_text)

                # Create overall stats
                total_chunks = sum(chunk_group["stats"]["num_chunks"] for chunk_group in chunks_array)
                avg_chunk_size = sum(chunk_group["stats"]["avg_chunk_size"] * chunk_group["stats"]["num_chunks"]
                                   for chunk_group in chunks_array) / total_chunks if total_chunks > 0 else 0

                overall_stats = {
                    "num_documents": len(chunks_array),
                    "total_chunks": total_chunks,
                    "avg_chunk_size": round(avg_chunk_size, 1),
                }

                results[item_id] = {
                    "item_id": item_id,
                    "original_length": total_original_length,
                    "chunks": chunks_array,
                    "overall_stats": overall_stats,
                    "validation_passed": True,  # Validation happens in chunking functions
                    "strategy": strategy,
                    "document_handling": document_handling,
                    "parameters": {
                        "delimiter": delimiter,
                        **({"chunk_size": chunk_size} if chunk_size else {}),
                        **({"num_chunks": num_chunks} if num_chunks else {}),
                    }
                }

            elif document_handling == "chunk-concatenated-docs":
                # Concatenate all documents then chunk
                doc_texts = [doc['content'] for doc in documents]
                text = delimiter.join(doc_texts)

                if not text:
                    raise ValueError("No text content found")

                # Apply chunking strategy to concatenated text
                concatenated_chunks = _apply_chunking_strategy(
                    text, strategy, chunk_size, num_chunks, delimiter
                )

                # Analyze chunks
                stats = analyze_chunks(concatenated_chunks, delimiter)

                # Create unified chunks structure (single pseudo-document)
                chunks_array = [{
                    "document_id": "concatenated",
                    "chunks": concatenated_chunks,
                    "stats": stats,
                }]

                # Overall stats (same as single document stats)
                overall_stats = {
                    "num_documents": len(documents),  # Original number of documents
                    "total_chunks": stats["num_chunks"],
                    "avg_chunk_size": stats["avg_chunk_size"],
                }

                results[item_id] = {
                    "item_id": item_id,
                    "original_length": len(text),
                    "chunks": chunks_array,
                    "overall_stats": overall_stats,
                    "validation_passed": True,  # Validation happens in chunking functions
                    "strategy": strategy,
                    "document_handling": document_handling,
                    "parameters": {
                        "delimiter": delimiter,
                        **({"chunk_size": chunk_size} if chunk_size else {}),
                        **({"num_chunks": num_chunks} if num_chunks else {}),
                    }
                }

        except Exception as e:
            errors[item_id] = str(e)

    return {"results": results, "errors": errors}
