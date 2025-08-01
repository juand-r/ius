"""
Simple chunking functions that respect text boundaries.

All chunking functions use delimiter-aware splitting to avoid breaking
words or sentences. Content preservation is guaranteed.
"""

import logging
from typing import Any

from tqdm import tqdm

from ..exceptions import ChunkingError, ValidationError
from .utils import analyze_chunks, validate_chunks


# Set up logger for this module
logger = logging.getLogger(__name__)


def chunk_fixed_size(text: str, chunk_size: int, delimiter: str = "\n") -> list[str]:
    """
    Split text into chunks of approximately fixed size, respecting delimiter boundaries.

    Args:
        text: Input text to chunk
        chunk_size: Target size for each chunk (in characters)
        delimiter: Boundary delimiter to respect (default: newline)

    Returns:
        List of text chunks, each roughly chunk_size characters

    Raises:
        ChunkingError: If input parameters are invalid
        ValidationError: If content preservation fails

    Note:
        Chunks may be slightly larger or smaller than chunk_size to respect
        delimiter boundaries and avoid splitting meaningful units.
    """
    # Input validation
    if not isinstance(text, str):
        raise ChunkingError(f"text must be a string, got {type(text).__name__}")

    if not isinstance(chunk_size, int):
        raise ChunkingError(
            f"chunk_size must be an integer, got {type(chunk_size).__name__}"
        )

    if chunk_size <= 0:
        raise ChunkingError(f"chunk_size must be positive, got {chunk_size}")

    if not isinstance(delimiter, str):
        raise ChunkingError(
            f"delimiter must be a string, got {type(delimiter).__name__}"
        )

    if len(delimiter) == 0:
        raise ChunkingError("delimiter cannot be empty")

    # Handle empty text - this should be an error, not silent failure
    if not text:
        raise ChunkingError("Cannot chunk empty text")

    # Warn about potentially inefficient chunk sizes
    if chunk_size > len(text) * 2:
        logger.warning(
            f"chunk_size ({chunk_size}) is much larger than text length ({len(text)}). "
            "Consider using a smaller chunk_size for better results."
        )

    if delimiter not in text:
        raise ChunkingError(
            f"Delimiter '{delimiter}' not found in text. "
            "Cannot split text that doesn't contain the specified delimiter."
        )

    # Split text into units separated by delimiter
    units = text.split(delimiter)
    chunks = []
    current_chunk = []
    current_size = 0

    # Add progress bar for large texts (many units to process)
    units_progress = tqdm(
        units,
        desc="Chunking text",
        unit="unit",
        disable=len(units) < 100,  # Only show for texts with many units
        leave=False,  # Remove bar when done
    )

    for unit in units_progress:
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
        raise ValidationError(
            "Content validation failed: chunks do not preserve original text"
        )

    return chunks


def chunk_fixed_count(text: str, num_chunks: int, delimiter: str = "\n") -> list[str]:
    """
    Split text into a fixed number of chunks, respecting delimiter boundaries.

    Args:
        text: Input text to chunk
        num_chunks: Target number of chunks
        delimiter: Boundary delimiter to respect (default: newline)

    Returns:
        List of text chunks (may be fewer than num_chunks if not enough delimiters)

    Raises:
        ChunkingError: If input parameters are invalid
        ValidationError: If content preservation fails
    """
    # Input validation
    if not isinstance(text, str):
        raise ChunkingError(f"text must be a string, got {type(text).__name__}")

    if not isinstance(num_chunks, int):
        raise ChunkingError(
            f"num_chunks must be an integer, got {type(num_chunks).__name__}"
        )

    if num_chunks <= 0:
        raise ChunkingError(f"num_chunks must be positive, got {num_chunks}")

    if not isinstance(delimiter, str):
        raise ChunkingError(
            f"delimiter must be a string, got {type(delimiter).__name__}"
        )

    if len(delimiter) == 0:
        raise ChunkingError("delimiter cannot be empty")

    # Handle empty text - this should be an error, not silent failure
    if not text:
        raise ChunkingError("Cannot chunk empty text")

    # Handle single chunk case
    if num_chunks == 1:
        return [text]

    # Warn about potentially inefficient chunk counts
    estimated_avg_chunk_size = len(text) // num_chunks
    if estimated_avg_chunk_size < 50:
        logger.warning(
            f"num_chunks ({num_chunks}) may create very small chunks "
            f"(estimated avg size: {estimated_avg_chunk_size} chars). "
            "Consider using fewer chunks for better results."
        )

    if delimiter not in text:
        raise ChunkingError(
            f"Delimiter '{delimiter}' not found in text. "
            "Cannot split text that doesn't contain the specified delimiter."
        )

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

    # Add progress bar for many chunks
    chunk_progress = tqdm(
        range(num_chunks),
        desc="Creating chunks",
        unit="chunk",
        disable=num_chunks < 10,  # Only show for many chunks
        leave=False,  # Remove bar when done
    )

    for i in chunk_progress:
        # Distribute remainder units across first chunks
        chunk_size = units_per_chunk + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size

        chunk_units = units[start_idx:end_idx]
        chunks.append(delimiter.join(chunk_units))
        start_idx = end_idx

    # Validate content preservation
    if not validate_chunks(text, chunks, delimiter):
        raise ValidationError(
            "Content validation failed: chunks do not preserve original text"
        )

    return chunks


def chunk_custom(
    text: str, strategy: str, delimiter: str = "\n", **kwargs
) -> list[str]:
    """
    Split text using custom strategy (placeholder for dataset-specific approaches).

    Args:
        text: Input text to chunk
        strategy: Custom strategy name (to be implemented later)
        delimiter: Boundary delimiter to respect (default: newline)
        **kwargs: Additional strategy-specific parameters

    Returns:
        List of text chunks

    Raises:
        ChunkingError: If input parameters are invalid
        NotImplementedError: Custom strategies are not yet implemented

    Note:
        This is a placeholder for future dataset-specific chunking strategies.
    """
    # Input validation
    if not isinstance(text, str):
        raise ChunkingError(f"text must be a string, got {type(text).__name__}")

    if not isinstance(strategy, str):
        raise ChunkingError(f"strategy must be a string, got {type(strategy).__name__}")

    if not isinstance(delimiter, str):
        raise ChunkingError(
            f"delimiter must be a string, got {type(delimiter).__name__}"
        )

    if len(delimiter) == 0:
        raise ChunkingError("delimiter cannot be empty")

    raise NotImplementedError(f"Custom chunking strategy '{strategy}' not implemented")


def _apply_chunking_strategy(
    text: str,
    strategy: str,
    chunk_size: int | None,
    num_chunks: int | None,
    delimiter: str,
) -> list[str]:
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
        ChunkingError: If strategy parameters are invalid
    """
    if strategy == "fixed_size":
        if not chunk_size or chunk_size <= 0:
            raise ChunkingError(
                "chunk_size required and must be positive for fixed_size strategy"
            )
        return chunk_fixed_size(text, chunk_size, delimiter)

    elif strategy == "fixed_count":
        if not num_chunks or num_chunks <= 0:
            raise ChunkingError(
                "num_chunks required and must be positive for fixed_count strategy"
            )
        return chunk_fixed_count(text, num_chunks, delimiter)

    elif strategy == "custom":
        return chunk_custom(text, "default", delimiter)

    else:
        raise ChunkingError(
            f"Unknown chunking strategy: {strategy}. Available strategies: fixed_size, fixed_count, custom"
        )


def process_dataset_items(
    items: dict[str, Any],
    strategy: str,
    document_handling: str = "chunk-individual-docs",
    chunk_size: int | None = None,
    num_chunks: int | None = None,
    delimiter: str = "\n",
) -> dict[str, Any]:
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
        ChunkingError: If input parameters are invalid
    """
    results = {}
    errors = {}

    # Input validation
    if not isinstance(items, dict):
        raise ChunkingError(f"items must be a dictionary, got {type(items).__name__}")

    if not isinstance(strategy, str):
        raise ChunkingError(f"strategy must be a string, got {type(strategy).__name__}")

    # Validate document_handling parameter
    valid_handling = ["chunk-individual-docs", "chunk-concatenated-docs"]
    if document_handling not in valid_handling:
        raise ChunkingError(
            f"Invalid document_handling: {document_handling}. Must be one of: {valid_handling}"
        )

    # Add progress bar for item processing
    items_progress = tqdm(
        items.items(),
        desc="Processing items",
        unit="item",
        disable=len(items) < 2,  # Don't show for single items
    )

    for item_id, item_data in items_progress:
        try:
            # Extract documents
            if not isinstance(item_data, dict) or "documents" not in item_data:
                raise ValidationError("Item missing required 'documents' field")

            documents = item_data["documents"]
            if not documents:
                raise ValidationError("No documents found in item")

            if document_handling == "chunk-individual-docs":
                # Chunk each document separately
                chunks_array = []
                total_original_length = 0

                # Add progress bar for document processing (only if multiple docs)
                docs_progress = tqdm(
                    documents,
                    desc=f"Processing {item_id} docs",
                    unit="doc",
                    disable=len(documents) < 2,  # Don't show for single documents
                    leave=False,  # Remove bar when done
                )

                for doc in docs_progress:
                    doc_id = doc["doc_id"]
                    doc_text = doc["content"]
                    if not doc_text:
                        raise ValidationError("Document content is empty")

                    # Apply chunking strategy to this document
                    doc_chunks = _apply_chunking_strategy(
                        doc_text, strategy, chunk_size, num_chunks, delimiter
                    )

                    # Analyze chunks for this document
                    doc_stats = analyze_chunks(doc_chunks, delimiter)

                    chunks_array.append(
                        {
                            "document_id": doc_id,
                            "chunks": doc_chunks,
                            "stats": doc_stats,
                        }
                    )
                    total_original_length += len(doc_text)

                # Create overall stats
                total_chunks = sum(
                    chunk_group["stats"]["num_chunks"] for chunk_group in chunks_array
                )
                avg_chunk_size = (
                    sum(
                        chunk_group["stats"]["avg_chunk_size"]
                        * chunk_group["stats"]["num_chunks"]
                        for chunk_group in chunks_array
                    )
                    / total_chunks
                    if total_chunks > 0
                    else 0
                )

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
                    },
                }

            elif document_handling == "chunk-concatenated-docs":
                raise ValueError("Currently untested, TODO Test before use!!")
                # Concatenate all documents then chunk
                doc_texts = [doc["content"] for doc in documents]
                text = delimiter.join(doc_texts)

                if not text:
                    raise ValidationError(
                        "No text content found in concatenated documents"
                    )

                # Apply chunking strategy to concatenated text
                concatenated_chunks = _apply_chunking_strategy(
                    text, strategy, chunk_size, num_chunks, delimiter
                )

                # Analyze chunks
                stats = analyze_chunks(concatenated_chunks, delimiter)

                # Create unified chunks structure (single pseudo-document)
                chunks_array = [
                    {
                        "document_id": "concatenated",
                        "chunks": concatenated_chunks,
                        "stats": stats,
                    }
                ]

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
                    },
                }

        except (ChunkingError, ValidationError) as e:
            # Handle validation and content errors
            error_msg = f"Validation error: {str(e)}"
            logger.warning(f"Processing item '{item_id}' failed: {error_msg}")
            errors[item_id] = error_msg

        except KeyError as e:
            # Handle missing required fields
            error_msg = f"Missing required field: {str(e)}"
            logger.error(f"Processing item '{item_id}' failed: {error_msg}")
            errors[item_id] = error_msg

        except TypeError as e:
            # Handle type-related errors (e.g., operations on wrong types)
            error_msg = f"Type error: {str(e)}"
            logger.error(f"Processing item '{item_id}' failed: {error_msg}")
            errors[item_id] = error_msg

        except NotImplementedError as e:
            # Handle custom chunking strategy not implemented
            error_msg = f"Feature not implemented: {str(e)}"
            logger.error(f"Processing item '{item_id}' failed: {error_msg}")
            errors[item_id] = error_msg

        except Exception as e:
            # Catch any unexpected exceptions with detailed logging
            error_msg = f"Unexpected error: {str(e)}"
            logger.exception(
                f"Processing item '{item_id}' failed with unexpected error: {error_msg}"
            )
            errors[item_id] = error_msg

    return {"results": results, "errors": errors}
