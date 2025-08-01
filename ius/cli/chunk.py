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

from ius.chunk import process_dataset_items
from ius.data import list_datasets, load_data

from .common import (
    print_summary_stats,
    save_json_output,
)


def chunk_dataset(
    dataset_name: str,
    strategy: str,
    chunk_size: int | None = None,
    num_chunks: int | None = None,
    delimiter: str = "\n",
    output_path: str | None = None,
    preview: bool = False,
) -> dict[str, any]:
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
    print(f"🔍 Loading dataset: {dataset_name}")

    try:
        # Load dataset - returns {"items": {...}, "collection_metadata": {...}, "num_items_loaded": N}
        dataset = load_data(dataset_name)

        # Extract the actual item data dictionary
        items = dataset["items"]
        print(f"📚 Loaded {len(items)} items from {dataset_name}")

    except Exception as e:
        print(f"❌ Error loading dataset {dataset_name}: {e}", file=sys.stderr)
        return {}

    # Print strategy info
    print(f"\n🔧 Chunking strategy: {strategy}")
    if strategy == "fixed_size" and chunk_size:
        print(f"   Target chunk size: {chunk_size} characters")
    elif strategy == "fixed_count" and num_chunks:
        print(f"   Target number of chunks: {num_chunks}")
    print(f"   Delimiter: {repr(delimiter)}")

    print(f"\n📝 Processing {len(items)} items...")

    # Call core processing function
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

        # Print progress for each item
        for i, (item_id, item_result) in enumerate(results.items(), 1):
            total_items = len(results)
            print(f"  [{i}/{total_items}] Processing: {item_id}")

            # Get overall stats
            overall_stats = item_result['overall_stats']
            print(f"    ✅ Created {overall_stats['total_chunks']} chunks, avg size: {overall_stats['avg_chunk_size']}")

            # Show document breakdown if individual docs
            if item_result['document_handling'] == 'chunk-individual-docs' and len(item_result['chunks']) > 1:
                print(f"    📄 {len(item_result['chunks'])} documents processed")

            # Show previews if requested
            if preview and item_result.get("chunks"):
                from ius.chunk.utils import preview_chunks
                # Get all chunks flattened for preview
                all_chunks = []
                for chunk_group in item_result["chunks"]:
                    all_chunks.extend(chunk_group["chunks"])

                previews = preview_chunks(all_chunks[:3])
                for prev in previews:
                    print(f"      {prev}")
                if len(all_chunks) > 3:
                    print(f"      ... and {len(all_chunks) - 3} more chunks")

        # Print errors if any
        if errors:
            for item_id, error in errors.items():
                print(f"  ❌ Error processing {item_id}: {error}")

    except Exception as e:
        print(f"❌ Error during processing: {e}", file=sys.stderr)
        return {}

    # Calculate overall statistics
    if results:
        total_chunks = sum(r["overall_stats"]["total_chunks"] for r in results.values())
        total_chars = sum(r["original_length"] for r in results.values())
        avg_chunks_per_item = total_chunks / len(results) if results else 0

        overall_stats = {
            "total_items": len(results),
            "total_chunks": total_chunks,
            "total_characters": total_chars,
            "avg_chunks_per_item": round(avg_chunks_per_item, 1),
            "processing_time_seconds": 0.0,  # Simple for now
            "validation_failures": sum(1 for r in results.values() if not r["validation_passed"]),
            "error_count": len(errors),
        }
    else:
        overall_stats = {
            "total_items": 0,
            "total_chunks": 0,
            "total_characters": 0,
            "avg_chunks_per_item": 0,
            "processing_time_seconds": 0.0,
            "validation_failures": 0,
            "error_count": len(errors),
        }

    # Print summary statistics
    print_summary_stats(overall_stats)

    # Prepare final output
    output_data = {
        "dataset": dataset_name,
        "strategy": strategy,
        "overall_stats": overall_stats,
        "items": results,
        "errors": errors,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save output if path specified
    if output_path:
        save_json_output(output_data, output_path)

    return output_data


def main() -> None:
    """Main entry point for chunking CLI."""
    # Handle --list-datasets early to avoid requiring other arguments
    if "--list-datasets" in sys.argv:
        datasets = list_datasets()
        if datasets:
            print("📚 Available datasets:")
            for dataset in sorted(datasets):
                print(f"  - {dataset}")
        else:
            print("📚 No datasets found in datasets/ directory")
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

  # Fixed count chunking
  python -m ius.chunk --dataset true-detective --strategy fixed_count --count 10

  # Custom delimiter
  python -m ius.chunk --dataset bmds --strategy fixed_size --size 1000 --delimiter "\\n\\n"

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

    # Note: Validation is now built into chunking functions
    # parser.add_argument(
    #     "--no-validate",
    #     action="store_true",
    #     help="Skip content preservation validation (deprecated)",
    # )

    args = parser.parse_args()

    # Validate required arguments for each strategy
    if args.strategy == "fixed_size" and not args.size:
        parser.error("--size is required for fixed_size strategy")

    if args.strategy == "fixed_count" and not args.count:
        parser.error("--count is required for fixed_count strategy")

    # Validate dataset exists
    available_datasets = list_datasets()
    if args.dataset not in available_datasets:
        print(f"❌ Dataset '{args.dataset}' not found", file=sys.stderr)
        if available_datasets:
            print(f"Available datasets: {', '.join(sorted(available_datasets))}", file=sys.stderr)
        else:
            print("No datasets found in datasets/ directory", file=sys.stderr)
        sys.exit(1)

    # Set up output path if not specified
    if not args.output:
        strategy_suffix = f"{args.strategy}_{args.size or args.count}"
        args.output = f"outputs/chunks/{args.dataset}_{strategy_suffix}.json"

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
            # validate=not args.no_validate,  # Validation now in chunking functions
        )

        if results:
            print("\n🎉 Chunking completed successfully!")
            if args.output:
                print(f"Results saved to: {args.output}")
        else:
            print("\n❌ Chunking failed or produced no results")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Chunking interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
