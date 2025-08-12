"""
Command-line interface for claim extraction operations.

Usage:
    python -m ius.cli.claim_extract --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --output outputs/claims/bmds_claims_default
    python -m ius.cli.claim_extract --input outputs/summaries/squality_summaries --scope item --item-ids 23942 24192
"""

import argparse
import sys
import time
from typing import Any
from pathlib import Path

from ius.claim_extract import process_dataset_summaries
from ius.exceptions import ClaimExtractionError, ValidationError
from ius.logging_config import get_logger, setup_logging

from .common import (
    print_summary_stats,
    save_json_output,
)

# Set up logger for this module
logger = get_logger(__name__)


def extract_claims_from_dataset(
    input_path: str,
    output_path: str | None = None,
    model: str = "gpt-4o-mini",
    prompt_name: str = "default-claim-extraction",
    scope: str = "all",
    item_ids: list[str] | None = None,
    domain: str = "story",
    ask_user_confirmation: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    CLI wrapper for extracting claims from summary datasets.
    
    Args:
        input_path: Path to the summary collection directory
        output_path: Path to save claim extraction results
        model: LLM model to use
        prompt_name: Name of the prompt directory to use
        scope: Processing scope ("all" or "item")
        item_ids: List of specific item IDs to process
        domain: Domain context for the summaries
        ask_user_confirmation: Whether to ask for confirmation before API calls
        verbose: Enable verbose logging
        
    Returns:
        Dictionary containing processing results and metadata
    """
    start_time = time.time()
    
    if verbose:
        setup_logging(log_level="DEBUG")
    
    logger.info(f"Starting claim extraction from: {input_path}")
    logger.info(f"Model: {model}, Prompt: {prompt_name}, Scope: {scope}")
    
    # Generate output path if not provided
    if not output_path:
        input_name = Path(input_path).name
        output_path = f"outputs/summaries-claims/{input_name}_claims_{prompt_name}"
        logger.info(f"Auto-generated output path: {output_path}")
    
    try:
        # Process the dataset
        results = process_dataset_summaries(
            summary_collection_path=input_path,
            output_path=output_path,
            model=model,
            prompt_name=prompt_name,
            ask_user_confirmation=ask_user_confirmation,
            domain=domain,
            scope=scope,
            item_ids=item_ids,
        )
        
        # Print processing statistics
        processing_time = time.time() - start_time
        stats = results["metadata"]["claim_extraction_info"]["processing_stats"]
        
        print(f"\n{'='*60}")
        print("CLAIM EXTRACTION COMPLETED")
        print(f"{'='*60}")
        print(f"Items processed: {stats['successful_items']}/{stats['total_items']}")
        if stats['failed_items'] > 0:
            print(f"Failed items: {stats['failed_items']}")
        print(f"Total cost: ${stats['total_cost']:.4f}")
        print(f"Total tokens: {stats['total_tokens']:,}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Output saved to: {output_path}")
        
        if results.get("errors"):
            print(f"\nErrors encountered:")
            for item_id, error in results["errors"].items():
                print(f"  {item_id}: {error}")
        
        return results
        
    except (ClaimExtractionError, ValidationError) as e:
        logger.error(f"Claim extraction failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the claim extraction CLI."""
    parser = argparse.ArgumentParser(
        description="Extract claims from summary datasets using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract claims from all summaries in a collection
  python -m ius.cli.claim_extract --input outputs/summaries/bmds_summaries --output outputs/summaries-claims/bmds_claims

  # Extract claims from specific items only
  python -m ius.cli.claim_extract --input outputs/summaries/squality_summaries --scope item --item-ids 23942 24192

  # Use a different model and prompt
  python -m ius.cli.claim_extract --input outputs/summaries/detective_summaries --model gpt-4 --prompt custom-claims

  # Enable verbose logging and user confirmation
  python -m ius.cli.claim_extract --input outputs/summaries/bmds_summaries --verbose --confirm
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the summary collection directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        help="Output directory path (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        default="default-claim-extraction",
        dest="prompt_name",
        help="Prompt directory name (default: default-claim-extraction)"
    )
    
    parser.add_argument(
        "--scope", "-s",
        choices=["all", "item"],
        default="all",
        help="Processing scope: 'all' for all items, 'item' for specific items (default: all)"
    )
    
    parser.add_argument(
        "--item-ids",
        nargs="+",
        help="Specific item IDs to process (required when scope is 'item')"
    )
    
    parser.add_argument(
        "--domain", "-d",
        default="story",
        help="Domain context for the summaries (default: story)"
    )
    
    parser.add_argument(
        "--confirm", "-c",
        action="store_true",
        help="Ask for user confirmation before making API calls"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.scope == "item" and not args.item_ids:
        parser.error("--item-ids is required when --scope is 'item'")
    
    if not Path(args.input).exists():
        parser.error(f"Input path does not exist: {args.input}")
    
    # Execute claim extraction
    extract_claims_from_dataset(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        prompt_name=args.prompt_name,
        scope=args.scope,
        item_ids=args.item_ids,
        domain=args.domain,
        ask_user_confirmation=args.confirm,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()