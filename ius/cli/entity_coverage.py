"""
Command-line interface for entity coverage evaluation operations.

Usage:
    python -m ius entity-coverage --input outputs/summaries/bmds_fixed_size2_8000_all_concat_5e8bbe --range penultimate
    python -m ius entity-coverage --input outputs/summaries/bmds_fixed_size2_8000_all_concat_5e8bbe --range all-but-last --model gpt-5-mini
"""

import argparse
import sys
import time
from typing import Any
from pathlib import Path

from ius.eval.intrinsic.entity_coverage import run_entity_coverage_evaluation
from ius.exceptions import ValidationError
from ius.logging_config import get_logger, setup_logging

from .common import (
    print_summary_stats,
    save_json_output,
)

# Set up logger for this module
logger = get_logger(__name__)


def _validate_input_path(input_path: str) -> None:
    """Validate that input path exists and is a directory."""
    path = Path(input_path)
    if not path.exists():
        raise ValidationError(f"Input path does not exist: {input_path}")
    if not path.is_dir():
        raise ValidationError(f"Input path must be a directory: {input_path}")


def evaluate_entity_coverage_dataset(
    input_path: str,
    range_spec: str = "penultimate",
    model: str = "gpt-5-mini",
    prompt_name: str = "default-entity-matching",
    output_dir: str | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    stop_after: int | None = None,
    add_reveal: bool = False,
    reveal_only: bool = False,
) -> dict[str, Any]:
    """
    Run entity coverage evaluation on a summary dataset.

    Args:
        input_path: Path to summary collection (must start with "outputs/summaries")
        range_spec: Which summary parts to use ("penultimate", "all-but-last", etc.)
        model: LLM model for entity matching and deduplication
        prompt_name: Name of matching prompt template
        output_dir: Optional custom output directory name
        overwrite: Whether to overwrite existing results
        verbose: Enable verbose logging
        stop_after: Optional limit on number of items to process (for testing)
        add_reveal: Whether to append reveal text to source documents

    Returns:
        Dictionary with evaluation results and statistics
    """
    # Validate input path
    _validate_input_path(input_path)
    
    logger.info(f"Starting entity coverage evaluation")
    logger.info(f"Input: {input_path}")
    logger.info(f"Range: {range_spec}")
    logger.info(f"Model: {model}")
    logger.info(f"Prompt: {prompt_name}")
    if output_dir:
        logger.info(f"Output directory: {output_dir}")
    logger.info(f"Overwrite: {overwrite}")
    if stop_after:
        logger.info(f"Stop after: {stop_after} items")

    start_time = time.time()

    try:
        # Run the evaluation
        output_path = run_entity_coverage_evaluation(
            input_path=input_path,
            range_spec=range_spec,
            model=model,
            prompt_name=prompt_name,
            output_dir=output_dir,
            overwrite=overwrite,
            stop_after=stop_after,
            add_reveal=add_reveal,
            reveal_only=reveal_only
        )

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Entity coverage evaluation completed in {duration:.2f} seconds")
        logger.info(f"Results saved to: {output_path}")

        # Load and return results summary
        collection_file = Path(output_path) / "collection.json"
        if collection_file.exists():
            import json
            with open(collection_file, 'r') as f:
                results = json.load(f)
                
            # Extract key statistics
            eval_info = results.get("entity_coverage_evaluation_info", {})
            stats = eval_info.get("processing_stats", {})
            
            summary = {
                "output_path": output_path,
                "duration": duration,
                "total_items": stats.get("total_items", 0),
                "successful_items": stats.get("successful_items", 0),
                "skipped_items": stats.get("skipped_items", 0),
                "failed_items": stats.get("failed_items", 0),
                "total_matching_calls": stats.get("total_matching_calls", 0),
                "estimated_cost": stats.get("total_cost", 0.0)
            }
            
            logger.info(f"Summary - Processed: {summary['successful_items']}/{summary['total_items']} items")
            logger.info(f"Skipped: {summary['skipped_items']}, Failed: {summary['failed_items']}")
            logger.info(f"Total LLM calls: {summary['total_matching_calls']}")
            logger.info(f"Estimated cost: ${summary['estimated_cost']:.3f}")
            
            return summary
        else:
            return {"output_path": output_path, "duration": duration}

    except Exception as e:
        logger.error(f"Entity coverage evaluation failed: {e}")
        raise


def main():
    """Main entry point for entity coverage CLI."""
    parser = argparse.ArgumentParser(
        description="Run entity coverage evaluation on summary datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with default settings
  python -m ius entity-coverage --input outputs/summaries/bmds_fixed_size2_8000_all_concat_5e8bbe

  # Use different summary range and model
  python -m ius entity-coverage --input outputs/summaries/bmds_fixed_size2_8000_all_concat_5e8bbe --range all-but-last --model gpt-5-mini

  # Custom output directory and overwrite existing results
  python -m ius entity-coverage --input outputs/summaries/bmds_fixed_size2_8000_all_concat_5e8bbe --output-dir my_test --overwrite

  # Process only first 10 items for testing
  python -m ius entity-coverage --input outputs/summaries/bmds_fixed_size2_8000_all_concat_5e8bbe --stop 10

  # Include reveal text in source documents (for detective stories)
  python -m ius entity-coverage --input outputs/summaries/bmds_fixed_size2_8000_all_concat_5e8bbe --add-reveal

  # Use only reveal text as source documents (for detective stories)
  python -m ius entity-coverage --input outputs/summaries/bmds_fixed_size2_8000_all_concat_5e8bbe --reveal-only

Range specifications:
  penultimate     Use second-to-last summary chunk (default)
  all-but-last    Use all chunks except the last one
  last            Use only the last summary chunk
  all             Use all summary chunks
  1-3             Use chunks 1 through 3
  2               Use only chunk 2

Supported models:

  gpt-5-mini      OpenAI GPT-5-mini (default)
  gpt-4o          OpenAI GPT-4o 
  gpt-4.1-mini    OpenAI GPT-4.1-mini
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        required=True,
        help="Path to summary collection (must start with 'outputs/summaries')",
    )

    # Optional arguments
    parser.add_argument(
        "--range",
        default="last",
        help="Which summary parts to evaluate (default: last)",
    )

    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="LLM model for entity matching and deduplication (default: gpt-5-mini)",
    )

    parser.add_argument(
        "--prompt",
        default="default-entity-matching",
        help="Name of matching prompt template (default: default-entity-matching)",
    )

    parser.add_argument(
        "--output-dir",
        help="Custom output directory name (auto-generated if not specified)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--stop",
        type=int,
        help="Stop processing after this many items (for testing/preview)",
    )

    parser.add_argument(
        "--add-reveal",
        action="store_true",
        help="Append reveal text to source documents (for bmds, true-detective, and detectiveqa datasets)",
    )

    parser.add_argument(
        "--reveal-only",
        action="store_true",
        help="Use only reveal text as source documents (for bmds, true-detective, and detectiveqa datasets)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)

    try:
        # Run evaluation
        results = evaluate_entity_coverage_dataset(
            input_path=args.input,
            range_spec=args.range,
            model=args.model,
            prompt_name=args.prompt,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            verbose=args.verbose,
            stop_after=args.stop,
            add_reveal=args.add_reveal,
            reveal_only=args.reveal_only,
        )

        logger.info("Entity coverage evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Entity coverage evaluation interrupted by user")
        sys.exit(1)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Entity coverage evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()