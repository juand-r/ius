"""
Command-line interface for whodunit evaluation operations.

Usage:
    python -m ius whodunit --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --range 1-3
    python -m ius whodunit --input outputs/chunks/bmds_fixed_size2_8000 --range last --scope item --item-ids ADP02
"""

import argparse
import sys
import time
from typing import Any
from pathlib import Path

from ius.eval.extrinsic.whodunit import run_whodunit_evaluation
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


def evaluate_whodunit_dataset(
    input_path: str,
    range_spec: str = "all",
    prompt_name: str = "default-whodunit-culprits-and-accomplices",
    scoring_prompt_name: str | None = None,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.1,
    max_completion_tokens: int = 2000,
    scope: str = "all",
    item_ids: list[str] | None = None,
    output_path: str | None = None,
    overwrite: bool = False,
    rescore: bool = False,
    ask_user_confirmation: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    CLI wrapper for whodunit evaluation on detective stories.
    
    Args:
        input_path: Path to the input collection directory (chunks or summaries)
        range_spec: Range specification for text selection
        prompt_name: Name of the prompt directory to use
        model: LLM model to use
        temperature: LLM temperature
        max_completion_tokens: Maximum tokens for LLM response
        scope: Processing scope ("all" or "item")
        item_ids: List of specific item IDs to process
        output_path: Path to save evaluation results
        overwrite: Whether to overwrite existing item results
        ask_user_confirmation: Whether to ask for confirmation before API calls
        verbose: Enable verbose logging
        
    Returns:
        Dictionary with evaluation results and metadata
    """
    # Logging is set up by the main() function
    
    # Show auto-generated output name if not provided
    if output_path is None:
        print(f"🎯 Auto-generated output name will be created")
    
    print(f"🕵️ Starting whodunit evaluation...")
    print(f"📥 Input: {input_path}")
    print(f"📊 Range: {range_spec}")
    print(f"🧠 Model: {model}")
    print(f"📝 Prompt: {prompt_name}")
    print(f"🎯 Scope: {scope}")
    
    if scope == "item" and not item_ids:
        raise ValidationError("--item-ids is required when scope is 'item'")
    
    if scope == "item":
        print(f"📋 Processing {len(item_ids)} specific items: {', '.join(item_ids)}")
    
    logger.info(f"Starting whodunit evaluation from: {input_path}")
    logger.info(f"Range: {range_spec}, Model: {model}, Prompt: {prompt_name}, Scope: {scope}")
    
    start_time = time.time()
    
    try:
        # Capture the actual command that was run
        argv_copy = sys.argv.copy()
        if argv_copy[0].endswith("__main__.py"):
            argv_copy[0] = "python -m ius"
        command_run = " ".join(argv_copy)
        
        # Run the evaluation
        result = run_whodunit_evaluation(
            input_dir=input_path,
            range_spec=range_spec,
            prompt_name=prompt_name,
            scoring_prompt_name=scoring_prompt_name,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            item_ids=item_ids if scope == "item" else None,
            output_dir=output_path,
            overwrite=overwrite,
            rescore=rescore,
            command_run=command_run,
            ask_user_confirmation=ask_user_confirmation,
            verbose=verbose
        )
        
        # Show the actual auto-generated output name
        if output_path is None:
            output_dir_name = result['output_dir'].split('/')[-1]  # Get just the directory name
            print(f"🎯 Auto-generated output name: {output_dir_name}")
        
        processing_time = time.time() - start_time
        
        # Extract statistics
        stats = result['collection_metadata']['whodunit_evaluation_info']['processing_stats']
        successful_items = stats['successful_items']
        total_items = stats['total_items']
        failed_items = stats['failed_items']
        skipped_items = stats.get('skipped_items', 0)  # Get actual skipped count
        
        # Rich final statistics display
        print(f"\n🎉 Whodunit evaluation completed!")
        print(f"✅ Processed: {successful_items} items")
        if skipped_items > 0:
            print(f"⏭️  Skipped: {skipped_items} items (already existed)")
        if failed_items > 0:
            print(f"❌ Failed: {failed_items} items")
        print(f"⏱️  Total time: {processing_time:.2f}s")
        print(f"💰 Total cost: ${result['total_cost']:.6f}")
        print(f"🔢 Total tokens: {result['total_tokens']:,}")
        print(f"📁 Results saved to: {result['output_dir']}")
        
        # Keep the logger messages for detailed logging
        logger.info("=" * 60)
        logger.info("WHODUNIT EVALUATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Items processed: {successful_items}/{total_items}")
        logger.info(f"Failed items: {failed_items}")
        logger.info(f"Total cost: ${result['total_cost']:.4f}")
        logger.info(f"Total tokens: {result['total_tokens']}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Output saved to: {result['output_dir']}")
        
        if failed_items > 0:
            logger.warning("Some items failed processing. Check logs for details.")
        
        return result
        
    except Exception as e:
        logger.error(f"Whodunit evaluation failed: {e}")
        raise


def main():
    """Main CLI entry point for whodunit evaluation."""
    parser = argparse.ArgumentParser(
        description="Extract whodunit analysis from detective story summaries or chunks using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all summaries using all available text
  python -m ius whodunit --input outputs/summaries/bmds_summaries

  # Evaluate using specific range of chunks/summaries
  python -m ius whodunit --input outputs/chunks/bmds_fixed_size2_8000 --range 1-3

  # Evaluate specific items only using the last chunk/summary
  python -m ius whodunit --input outputs/summaries/squality_summaries --scope item --item-ids 23942 24192 --range last

  # Use a different model and prompt
  python -m ius whodunit --input outputs/summaries/detective_summaries --model gpt-4 --prompt custom-whodunit

  # Enable verbose logging and user confirmation
  python -m ius whodunit --input outputs/summaries/bmds_summaries --verbose --confirm

  # Custom output directory
  python -m ius whodunit --input outputs/summaries/bmds_summaries --output outputs/eval/custom_whodunit
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input collection directory (chunks or summaries)"
    )
    
    # Range specification
    parser.add_argument(
        "--range", "-r",
        default="all",
        help="Range specification for text selection: 'all', 'last', 'penultimate', 'all-but-last', '1', '1-3', etc. (default: all)"
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
        default="default-whodunit-culprits-and-accomplices",
        help="Prompt directory name (default: default-whodunit-culprits-and-accomplices)"
    )
    
    parser.add_argument(
        "--scoring-prompt",
        help="Scoring prompt directory name (if not provided, Phase 2 scoring will be skipped)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Maximum tokens for LLM response (default: 2000)"
    )
    
    # Scope and item selection
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
    
    # Control flags
    parser.add_argument(
        "--confirm", "-c",
        action="store_true",
        help="Ask for user confirmation before making API calls"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing item results (default: skip existing items)"
    )
    
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Force re-run Phase 2 scoring even if already completed"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (equivalent to --log-level DEBUG)"
    )
    
    args = parser.parse_args()
    
    # Set up logging (verbose flag overrides log-level)
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(log_level=log_level)
    
    try:
        # Validate input path
        _validate_input_path(args.input)
        
        # Run the evaluation
        kwargs = {
            "input_path": args.input,
            "range_spec": args.range,
            "prompt_name": args.prompt,
            "scoring_prompt_name": args.scoring_prompt,
            "model": args.model,
            "temperature": args.temperature,
            "max_completion_tokens": args.max_tokens,
            "scope": args.scope,
            "item_ids": args.item_ids,
            "output_path": args.output,
            "overwrite": args.overwrite,
            "rescore": args.rescore,
            "ask_user_confirmation": args.confirm,
            "verbose": args.verbose
        }
        
        result = evaluate_whodunit_dataset(**kwargs)
        
        logger.info(f"Whodunit evaluation completed successfully: {result['collection_metadata']['whodunit_evaluation_info']['processing_stats']['successful_items']} items processed")
        
    except (ValidationError) as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {e}")
        if args.verbose or log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()