"""
Command-line interface for text chunking operations.

Usage:
    python -m ius.chunk --dataset bmds --strategy fixed_size --size 2048
    python -m ius.chunk --dataset true-detective --strategy fixed_count --count 10
    python -m ius.chunk --dataset bmds --strategy custom --delimiter "\\n\\n"
"""

import argparse
import sys
import time
from typing import Any

from ius.chunk import process_dataset_items
from ius.data import Dataset, list_datasets
from ius.exceptions import ChunkingError, DatasetError, ValidationError
from ius.logging_config import get_logger, setup_logging

from .common import (
    print_summary_stats,
    save_json_output,
    save_chunked_collection_and_items,
)


# Set up logger for this module
logger = get_logger(__name__)


def chunk_dataset(
    dataset_name: str,
    strategy: str,
    chunk_size: int | None = None,
    num_chunks: int | None = None,
    delimiter: str = "\n",
    output_path: str | None = None,
    preview: bool = False,
) -> dict[str, Any]:
    """
    CLI wrapper for chunking datasets with progress printing and file I/O.

    Args:
        dataset_name: Name of the dataset to chunk
        strategy: Chunking strategy ('fixed_size', 'fixed_count', 'custom')
        chunk_size: Target chunk size (for fixed_size strategy)
        num_chunks: Number of chunks (for fixed_count strategy)
        delimiter: Boundary delimiter for splitting
        output_path: Path to save chunked results
        preview: Whether to show chunk previews

    Returns:
        Dictionary with chunking results and metadata
    """
    # Capture the command that generated this chunked dataset for reproducibility
    # Replace the full path to __main__.py with user-friendly "python -m ius" format
    argv_copy = sys.argv.copy()
    if argv_copy[0].endswith("__main__.py"):
        argv_copy[0] = "python -m ius"
    command_run = " ".join(argv_copy)
    
    # Load and validate dataset
    dataset = _load_and_validate_dataset(dataset_name)
    if not dataset:
        return {}

    # Convert Dataset object to items dictionary for processing
    items_dict = _dataset_to_items_dict(dataset)

    # Print strategy information
    _print_strategy_info(strategy, chunk_size, num_chunks, delimiter)

    # Process items with chunking
    results, errors = _process_items_with_chunking(
        items_dict, strategy, chunk_size, num_chunks, delimiter, preview
    )
    if results is None:  # Processing failed
        return {}

    # Calculate and display statistics
    overall_stats = _calculate_overall_statistics(results, errors)
    print_summary_stats(overall_stats)

    # Prepare chunked items and collection in new format
    chunked_items, chunked_collection = _prepare_chunked_data_for_saving(
        dataset_name, strategy, overall_stats, results, items_dict, dataset.metadata, command_run
    )

    # Save chunked collection and items if path specified
    if output_path:
        save_chunked_collection_and_items(chunked_collection, chunked_items, output_path)

    # Prepare legacy output format for return value (compatibility)
    output_data = _prepare_output_data(
        dataset_name, strategy, overall_stats, results, errors
    )

    return output_data


# Helper functions for chunk_dataset breakdown


def _load_and_validate_dataset(dataset_name: str) -> Dataset | None:
    """
    Load dataset using Dataset class with comprehensive error handling.

    Args:
        dataset_name: Name of the dataset to load

    Returns:
        Dataset object or None if loading failed
    """
    logger.info(f"Loading dataset: {dataset_name}")

    try:
        # Load dataset using Dataset class
        dataset = Dataset(f"datasets/{dataset_name}")
        logger.info(f"Loaded {len(dataset)} items from {dataset_name}")

        return dataset

    except DatasetError as e:
        logger.error(f"Dataset error: {e}")
        logger.info(
            "Run 'python -m ius chunk --list-datasets' to see available datasets"
        )
        return None
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info(
            "Check that the datasets directory exists and contains the specified dataset"
        )
        return None
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        logger.info("Check file permissions for the datasets directory")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading dataset {dataset_name}: {e}")
        logger.info("This may be a bug. Please check the dataset format and try again.")
        return None


def _dataset_to_items_dict(dataset: Dataset) -> dict[str, Any]:
    """
    Convert Dataset object to items dictionary for compatibility with existing processing functions.
    
    Args:
        dataset: Dataset object
        
    Returns:
        Dictionary mapping item_id -> item_data
    """
    items_dict = {}
    for item_id in dataset.item_ids:
        items_dict[item_id] = dataset.load_item(item_id)
    return items_dict


def _print_strategy_info(
    strategy: str, chunk_size: int | None, num_chunks: int | None, delimiter: str
) -> None:
    """Print chunking strategy information to console."""
    logger.info(f"Chunking strategy: {strategy}")
    if strategy == "fixed_size" and chunk_size:
        logger.info(f"Target chunk size: {chunk_size} characters")
    elif strategy == "fixed_count" and num_chunks:
        logger.info(f"Target number of chunks: {num_chunks}")
    logger.info(f"Delimiter: {repr(delimiter)}")


def _process_items_with_chunking(
    items: dict,
    strategy: str,
    chunk_size: int | None,
    num_chunks: int | None,
    delimiter: str,
    preview: bool,
) -> tuple[dict, dict] | tuple[None, None]:
    """
    Process items with chunking and handle progress display.

    Returns:
        (results, errors) tuple or (None, None) if processing failed
    """
    logger.info(f"Processing {len(items)} items...")

    try:
        processing_results = process_dataset_items(
            items=items,
            strategy=strategy,
            document_handling="chunk-individual-docs",
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            delimiter=delimiter,
        )

        results = processing_results["results"]
        errors = processing_results["errors"]

        # Log progress for each item
        for i, (item_id, item_result) in enumerate(results.items(), 1):
            total_items = len(results)
            logger.info(f"[{i}/{total_items}] Processing: {item_id}")

            # Get overall stats
            overall_stats = item_result["overall_stats"]
            logger.info(
                f"Created {overall_stats['total_chunks']} chunks, avg size: {overall_stats['avg_chunk_size']}"
            )

            # Show document breakdown if individual docs
            if (
                item_result["document_handling"] == "chunk-individual-docs"
                and len(item_result["chunks"]) > 1
            ):
                logger.info(f"{len(item_result['chunks'])} documents processed")

            # Show previews if requested
            if preview and item_result.get("chunks"):
                from ius.chunk.utils import preview_chunks

                # Get all chunks flattened for preview
                all_chunks = []
                for chunk_group in item_result["chunks"]:
                    all_chunks.extend(chunk_group["chunks"])

                previews = preview_chunks(all_chunks[:3])
                for prev in previews:
                    logger.info(f"Preview: {prev}")
                if len(all_chunks) > 3:
                    logger.info(f"... and {len(all_chunks) - 3} more chunks")

        # Log errors if any
        if errors:
            for item_id, error in errors.items():
                logger.error(f"Error processing {item_id}: {error}")

        return results, errors

    except ChunkingError as e:
        logger.error(f"Chunking configuration error: {e}")
        logger.info(
            "Check your chunking parameters (strategy, chunk_size, num_chunks, delimiter)"
        )
        return None, None
    except ValidationError as e:
        logger.error(f"Data validation error: {e}")
        logger.info("Check that your dataset items have the required structure")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
        logger.info("This may be a bug. Please check your input data and try again.")
        return None, None


def _calculate_overall_statistics(results: dict, errors: dict) -> dict[str, Any]:
    """
    Calculate overall statistics from chunking results.

    Args:
        results: Dictionary of successful chunking results
        errors: Dictionary of processing errors

    Returns:
        Dictionary with overall statistics
    """
    if results:
        total_chunks = sum(r["overall_stats"]["total_chunks"] for r in results.values())
        total_chars = sum(r["original_length"] for r in results.values())
        avg_chunks_per_item = total_chunks / len(results) if results else 0

        return {
            "total_items": len(results),
            "total_chunks": total_chunks,
            "total_characters": total_chars,
            "avg_chunks_per_item": round(avg_chunks_per_item, 1),
            "processing_time_seconds": 0.0,  # Simple for now
            "error_count": len(errors),
        }
    else:
        return {
            "total_items": 0,
            "total_chunks": 0,
            "total_characters": 0,
            "avg_chunks_per_item": 0,
            "processing_time_seconds": 0.0,
            "error_count": len(errors),
        }


def _prepare_chunked_data_for_saving(
    dataset_name: str, strategy: str, overall_stats: dict, results: dict, 
    original_items: dict, original_collection_metadata: dict, command_run: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Prepare chunked collection and items in the new format that matches original dataset structure.
    
    Args:
        dataset_name: Name of the dataset
        strategy: Chunking strategy used
        overall_stats: Overall statistics from chunking
        results: Chunking results per item
        original_items: Original item data to preserve metadata
        original_collection_metadata: Original collection metadata
        command_run: Command that generated this chunked dataset (for reproducibility)
        
    Returns:
        Tuple of (chunked_items, chunked_collection)
    """
    # Prepare collection-level data
    chunked_collection = {
        **original_collection_metadata,  # Preserve original collection metadata
        "chunking_info": {
            "strategy": strategy,
            "overall_stats": overall_stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "command_run": command_run
        }
    }
    
    # Prepare individual items
    chunked_items = {}
    
    for item_id, chunk_result in results.items():
        # Get original item data for metadata preservation
        original_item = original_items[item_id]
        original_metadata = original_item.get("item_metadata", {})
        
        # Create new item_metadata with ONLY item-specific chunking info
        item_metadata = {
            **original_metadata,  # Preserve original metadata
            "chunking_method": strategy,
            "chunking_params": chunk_result["parameters"],
            "chunking_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Convert chunked documents to new format
        documents = []
        for i, chunk_group in enumerate(chunk_result["chunks"]):
            # Get original document metadata if available
            original_doc = original_item["documents"][i] if i < len(original_item["documents"]) else {}
            original_doc_metadata = original_doc.get("metadata", {})
            
            document = {
                "chunks": chunk_group["chunks"],  # List of text chunks
                "metadata": {
                    "original_metadata": original_doc_metadata,
                    "chunking_stats": {
                        **chunk_group["stats"],
                        "original_length": len(original_doc.get("content", "")),
                        "command_run": command_run
                    }
                }
            }
            documents.append(document)
        
        # Create the new chunked item structure
        chunked_items[item_id] = {
            "item_metadata": item_metadata,
            "documents": documents
        }
    
    return chunked_items, chunked_collection


def _prepare_output_data(
    dataset_name: str, strategy: str, overall_stats: dict, results: dict, errors: dict
) -> dict[str, Any]:
    """
    Prepare final output data structure (legacy format for compatibility).

    Returns:
        Complete output data dictionary
    """
    return {
        "dataset": dataset_name,
        "strategy": strategy,
        "overall_stats": overall_stats,
        "items": results,
        "errors": errors,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> None:
    """Main entry point for chunking CLI."""
    # Parse args early to get verbose flag for logging setup
    if "--verbose" in sys.argv or "-v" in sys.argv:
        setup_logging(log_level="INFO", verbose=True)
    else:
        setup_logging(log_level="INFO", verbose=False)

    # Handle --list-datasets early to avoid requiring other arguments
    if "--list-datasets" in sys.argv:
        datasets = list_datasets()
        if datasets:
            logger.info("Available datasets:")
            for dataset in sorted(datasets):
                logger.info(f"  - {dataset}")
        else:
            logger.info("No datasets found in datasets/ directory")
        return

    parser = argparse.ArgumentParser(
        description="Chunk documents for incremental summarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python -m ius.chunk --list-datasets

  # Fixed size chunking
  python -m ius.chunk --dataset bmds --strategy fixed_size --size 2048

  # Fixed count chunking with verbose output
  python -m ius.chunk --dataset true-detective --strategy fixed_count --count 10 --verbose

  # Dry run to preview what would be processed
  python -m ius.chunk --dataset bmds --strategy fixed_size --size 1000 --dry-run

  # Custom delimiter with verbose logging
  python -m ius.chunk --dataset bmds --strategy fixed_size --size 1000 --delimiter "\\n\\n" -v

  # Save output and show previews
  python -m ius.chunk --dataset bmds --strategy fixed_size --size 2048 \\
    --output outputs/bmds_chunks_2048.json --preview
        """,
    )

    # Required arguments
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (must exist in datasets/ directory)",
    )

    parser.add_argument(
        "--strategy",
        choices=["fixed_size", "fixed_count", "custom"],
        required=True,
        help="Chunking strategy to use",
    )

    # Strategy-specific arguments
    parser.add_argument(
        "--size",
        type=int,
        help="Target chunk size in characters (required for fixed_size)",
    )

    parser.add_argument(
        "--count",
        type=int,
        help="Number of chunks to create (required for fixed_count)",
    )

    # Optional arguments
    parser.add_argument(
        "--delimiter",
        default="\n",
        help="Boundary delimiter for splitting (default: newline)",
    )

    parser.add_argument(
        "--output",
        help="Output file path (JSON format)",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show chunk previews during processing",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging with timestamps and module names",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually doing it",
    )

    args = parser.parse_args()

    # Validate required arguments for each strategy
    if args.strategy == "fixed_size" and not args.size:
        parser.error("--size is required for fixed_size strategy")

    if args.strategy == "fixed_count" and not args.count:
        parser.error("--count is required for fixed_count strategy")

    # Validate dataset exists
    available_datasets = list_datasets()
    if args.dataset not in available_datasets:
        logger.error(f"Dataset '{args.dataset}' not found")
        if available_datasets:
            logger.info(f"Available datasets: {', '.join(sorted(available_datasets))}")
        else:
            logger.info("No datasets found in datasets/ directory")
        sys.exit(1)

    # Set up output path if not specified
    if not args.output:
        strategy_suffix = f"{args.strategy}_{args.size or args.count}"
        args.output = f"outputs/chunks/{args.dataset}_{strategy_suffix}"

    # Handle dry-run mode
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No actual processing will be performed")

        # Load dataset to show what would be processed
        try:
            dataset = _load_and_validate_dataset(args.dataset)
            if dataset:
                logger.info(
                    f"📋 Would process {len(dataset)} items from dataset '{args.dataset}'"
                )
                logger.info(
                    f"📋 Items: {', '.join(sorted(dataset.item_ids)[:5])}{'...' if len(dataset) > 5 else ''}"
                )

                # Show strategy that would be used
                logger.info(f"🔧 Would use chunking strategy: {args.strategy}")
                if args.strategy == "fixed_size" and args.size:
                    logger.info(f"🔧 Target chunk size: {args.size} characters")
                elif args.strategy == "fixed_count" and args.count:
                    logger.info(f"🔧 Target number of chunks: {args.count}")
                logger.info(f"🔧 Delimiter: {repr(args.delimiter)}")

                # Show output path
                logger.info(f"💾 Would save results to: {args.output}")

                logger.info("✨ Dry run completed - no files were modified")
                return
            else:
                logger.error("Cannot show dry run preview - dataset loading failed")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error during dry run: {e}")
            sys.exit(1)

    # Run chunking
    try:
        results = chunk_dataset(
            dataset_name=args.dataset,
            strategy=args.strategy,
            chunk_size=args.size,
            num_chunks=args.count,
            delimiter=args.delimiter,
            output_path=args.output,
            preview=args.preview,
        )

        if results:
            logger.info("Chunking completed successfully!")
            if args.output:
                logger.info(f"Results saved to: {args.output}")
        else:
            logger.error("Chunking failed or produced no results")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Chunking interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
