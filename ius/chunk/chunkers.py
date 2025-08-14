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

# NLTK imports for sentence segmentation
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


# Set up logger for this module
logger = logging.getLogger(__name__)


def _get_text_units(text: str, sentence_mode: bool = False, delimiter: str = "\n") -> list[str]:
    """
    Get text units for chunking - either sentences (via NLTK) or delimiter-split units.
    
    Args:
        text: Text to split into units
        sentence_mode: Whether to use NLTK sentence segmentation
        delimiter: Delimiter to use if not in sentence mode
        
    Returns:
        List of text units
        
    Raises:
        ChunkingError: If sentence mode is requested but NLTK is not available
    """
    if sentence_mode:
        if not NLTK_AVAILABLE:
            raise ChunkingError(
                "NLTK is required for sentence mode but is not installed. "
                "Please install it with: pip install nltk"
            )
        
        # Ensure punkt tokenizer is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        return sent_tokenize(text)
    else:
        return text.split(delimiter)


def _extract_reveal_segment(item_data: dict[str, Any], dataset_name: str) -> str | None:
    """
    Extract reveal segment from dataset item based on dataset type.
    
    Args:
        item_data: Item data dictionary
        dataset_name: Name of the dataset (e.g., "bmds", "true-detective")
        
    Returns:
        Reveal segment text or None if not found
    """
    try:
        documents = item_data.get("documents", [])
        if not documents:
            return None
            
        metadata = documents[0].get("metadata", {})
        
        if dataset_name == "bmds":
            # BMDS path: metadata.detection.reveal_segment
            detection = metadata.get("detection", {})
            return detection.get("reveal_segment", "") or None
            
        elif dataset_name == "true-detective":
            # True Detective path: metadata.original_metadata.puzzle_data.outcome
            original_metadata = metadata.get("original_metadata", {})
            puzzle_data = original_metadata.get("puzzle_data", {})
            return puzzle_data.get("outcome", "") or None
            
        else:
            logger.warning(f"Unknown dataset '{dataset_name}' for reveal segment extraction")
            return None
        
    except (KeyError, IndexError, AttributeError):
        return None


def chunk_fixed_size(text: str, chunk_size: int, delimiter: str = "\n", _is_retry: bool = False, sentence_mode: bool = False) -> list[str]:
    """
    Split text into chunks of approximately fixed size, respecting delimiter boundaries.
    
    Automatically optimizes chunk sizes to minimize very small final chunks by either
    merging tiny chunks or redistributing content across all chunks for better uniformity.

    Args:
        text: Input text to chunk
        chunk_size: Target size for each chunk (in characters)
        delimiter: Boundary delimiter to respect (default: newline)
        _is_retry: Internal flag to prevent infinite recursion (do not use)
        sentence_mode: Whether to use NLTK sentence segmentation instead of delimiter

    Returns:
        List of text chunks with optimized sizes:
        - Tiny final chunks (<500 chars) are merged with previous chunk
        - Small final chunks (500-50% of chunk_size) trigger redistribution
        - Reasonable final chunks (>50% of chunk_size) are kept as-is

    Raises:
        ChunkingError: If input parameters are invalid
        ValidationError: If content preservation fails

    Note:
        Chunks may be slightly larger or smaller than chunk_size to respect
        delimiter boundaries and avoid splitting meaningful units. The algorithm
        performs intelligent post-processing to ensure better size uniformity.
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

    if not sentence_mode and delimiter not in text:
        raise ChunkingError(
            f"Delimiter '{delimiter}' not found in text. "
            "Cannot split text that doesn't contain the specified delimiter."
        )

    # Split text into units (sentences or delimiter-separated)
    units = _get_text_units(text, sentence_mode, delimiter)
    
    # Determine the joiner for combining units back into chunks
    joiner = " " if sentence_mode else delimiter
    
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
        # Add joiner length except for first unit in chunk
        total_size = current_size + unit_size + (len(joiner) if current_chunk else 0)

        if total_size <= chunk_size or not current_chunk:
            # Add unit to current chunk (always add at least one unit)
            current_chunk.append(unit)
            current_size = total_size
        else:
            # Current chunk is full, start new chunk
            chunks.append(joiner.join(current_chunk))
            current_chunk = [unit]
            current_size = unit_size

    # Add remaining chunk if any
    if current_chunk:
        chunks.append(joiner.join(current_chunk))

    # Validate content preservation (use flexible validation for sentence mode)
    if not validate_chunks(text, chunks, joiner, normalize_whitespace=sentence_mode):
        raise ValidationError(
            "Content validation failed: chunks do not preserve original text"
        )

    # Optimize chunk sizes to avoid very small final chunks (only on first pass)
    if not _is_retry and len(chunks) > 1:
        last_chunk_size = len(chunks[-1])
        min_len = 400  # Minimum acceptable chunk size
        small_threshold = 0.5 * chunk_size  # 50% of target chunk size
        
        # Case 1: Tiny final chunk - merge with previous chunk
        if last_chunk_size < min_len:
            logger.info(f"Merging tiny final chunk ({last_chunk_size} chars) with previous chunk")
            chunks[-2] = chunks[-2] + joiner + chunks[-1]
            chunks.pop()
            
        # Case 2: Small final chunk - redistribute across all chunks
        elif min_len <= last_chunk_size <= small_threshold:
            num_complete_chunks = len(chunks) - 1  # Exclude the small final chunk
            total_content_size = chunk_size * num_complete_chunks + last_chunk_size
            new_chunk_size = int(total_content_size / num_complete_chunks)
            
            logger.info(f"Redistributing small final chunk ({last_chunk_size} chars) "
                       f"across {num_complete_chunks} chunks. New target size: {new_chunk_size}")
            
            # Re-chunk with optimized size
            optimized_chunks = chunk_fixed_size(text, new_chunk_size, delimiter, _is_retry=True, sentence_mode=sentence_mode)
            
            # Final safety check: if we still have a tiny final chunk, merge it
            if len(optimized_chunks) > 1 and len(optimized_chunks[-1]) < 1.5*min_len:
                logger.info(f"Final safety check: merging remaining tiny chunk ({len(optimized_chunks[-1])} chars)")
                # Use the same joiner logic as the optimized chunks were created with
                opt_joiner = " " if sentence_mode else delimiter
                optimized_chunks[-2] = optimized_chunks[-2] + opt_joiner + optimized_chunks[-1]
                optimized_chunks.pop()
            
            return optimized_chunks
            
        # Case 3: Reasonable final chunk - keep as-is
        else:
            logger.debug(f"Final chunk size ({last_chunk_size} chars) is reasonable, keeping as-is")

    return chunks


def chunk_fixed_count(text: str, num_chunks: int, delimiter: str = "\n", sentence_mode: bool = False) -> list[str]:
    """
    Split text into a fixed number of chunks, respecting delimiter boundaries.

    Args:
        text: Input text to chunk
        num_chunks: Target number of chunks
        delimiter: Boundary delimiter to respect (default: newline)
        sentence_mode: Whether to use NLTK sentence segmentation instead of delimiter

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

    if not sentence_mode and delimiter not in text:
        raise ChunkingError(
            f"Delimiter '{delimiter}' not found in text. "
            "Cannot split text that doesn't contain the specified delimiter."
        )

    # Split text into units (sentences or delimiter-separated)
    units = _get_text_units(text, sentence_mode, delimiter)
    
    # Determine the joiner for combining units back into chunks
    joiner = " " if sentence_mode else delimiter

    if len(units) < num_chunks:
        # Not enough units to create requested number of chunks
        # Return each unit as separate chunk
        return [joiner.join([unit]) if unit else unit for unit in units]

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
        chunks.append(joiner.join(chunk_units))
        start_idx = end_idx

    # Validate content preservation (use flexible validation for sentence mode)
    if not validate_chunks(text, chunks, joiner, normalize_whitespace=sentence_mode):
        raise ValidationError(
            "Content validation failed: chunks do not preserve original text"
        )

    return chunks


def chunk_custom(
    text: str, strategy: str, delimiter: str = "\n", sentence_mode: bool = False, **kwargs
) -> list[str]:
    """
    Split text using custom strategy (placeholder for dataset-specific approaches).

    Args:
        text: Input text to chunk
        strategy: Custom strategy name (to be implemented later)
        delimiter: Boundary delimiter to respect (default: newline)
        sentence_mode: Whether to use NLTK sentence segmentation instead of delimiter
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
    sentence_mode: bool = False,
) -> list[str]:
    """
    Apply the specified chunking strategy to text.

    Args:
        text: Text to chunk
        strategy: Chunking strategy
        chunk_size: Target chunk size (for fixed_size)
        num_chunks: Target number of chunks (for fixed_count)
        delimiter: Delimiter for chunking
        sentence_mode: Whether to use NLTK sentence segmentation instead of delimiter

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
        return chunk_fixed_size(text, chunk_size, delimiter, sentence_mode=sentence_mode)

    elif strategy == "fixed_count":
        if not num_chunks or num_chunks <= 0:
            raise ChunkingError(
                "num_chunks required and must be positive for fixed_count strategy"
            )
        return chunk_fixed_count(text, num_chunks, delimiter, sentence_mode=sentence_mode)

    elif strategy == "custom":
        return chunk_custom(text, "default", delimiter, sentence_mode=sentence_mode)

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
    reveal_add_on: bool = False,
    dataset_name: str | None = None,
    sentence_mode: bool = False,
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
        reveal_add_on: Whether to add reveal segment as final chunk
        dataset_name: Name of the dataset (for reveal segment extraction)
        sentence_mode: Whether to use NLTK sentence segmentation instead of delimiter

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
                        doc_text, strategy, chunk_size, num_chunks, delimiter, sentence_mode
                    )

                    # Analyze chunks for this document
                    joiner = " " if sentence_mode else delimiter
                    doc_stats = analyze_chunks(doc_chunks, joiner)

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

                # Add reveal segment as final chunk if requested
                if reveal_add_on:
                    reveal_segment = _extract_reveal_segment(item_data, dataset_name)
                    if reveal_segment:
                        logger.info(f"Adding reveal segment as final chunk for {item_id} ({len(reveal_segment)} chars)")
                        
                        # Add reveal segment to all documents' chunks
                        for chunk_group in chunks_array:
                            chunk_group["chunks"].append(reveal_segment)
                    else:
                        logger.warning(f"Reveal segment not found for {item_id}, skipping reveal add-on")

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
                        **({"reveal_add_on": reveal_add_on} if reveal_add_on else {}),
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
                    text, strategy, chunk_size, num_chunks, delimiter, sentence_mode
                )

                # Analyze chunks
                joiner = " " if sentence_mode else delimiter
                stats = analyze_chunks(concatenated_chunks, joiner)

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
                
                original_length = len(text)

                # Add reveal segment as final chunk if requested
                if reveal_add_on:
                    reveal_segment = _extract_reveal_segment(item_data, dataset_name)
                    if reveal_segment:
                        logger.info(f"Adding reveal segment as final chunk for {item_id} ({len(reveal_segment)} chars)")
                        
                        # Add reveal segment to all documents' chunks
                        for chunk_group in chunks_array:
                            chunk_group["chunks"].append(reveal_segment)
                    else:
                        logger.warning(f"Reveal segment not found for {item_id}, skipping reveal add-on")

                results[item_id] = {
                    "item_id": item_id,
                    "original_length": original_length,
                    "chunks": chunks_array,
                    "overall_stats": overall_stats,
                    "validation_passed": True,  # Validation happens in chunking functions
                    "strategy": strategy,
                    "document_handling": document_handling,
                    "parameters": {
                        "delimiter": delimiter,
                        **({"chunk_size": chunk_size} if chunk_size else {}),
                        **({"num_chunks": num_chunks} if num_chunks else {}),
                        **({"reveal_add_on": reveal_add_on} if reveal_add_on else {}),
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
