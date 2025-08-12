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

from eval.extrinsic.whodunit import run_whodunit_evaluation
from ius.exceptions import ValidationError
from ius.logging_config import get_logger, setup_logging

from .common import (
    print_summary_stats,
    save_json_output,
)

# Set up logger for this module
logger = get_logger(__name__)


def evaluate_whodunit_dataset(
    input_path: str,
    range_spec: str = "all",
    prompt_name: str = "default-whodunit-culprits-and-accomplices",
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    scope: str = "all",
    item_ids: list[str] | None = None,
    output_path: str | None = None,
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
        max_tokens: Maximum tokens for LLM response
        scope: Processing scope ("all" or "item")
        item_ids: List of specific item IDs to process
        output_path: Path to save evaluation results
        ask_user_confirmation: Whether to ask for confirmation before API calls
        verbose: Enable verbose logging
        
    Returns:
        Dictionary with evaluation results and metadata
    """
    if verbose:
        setup_logging(log_level="DEBUG")
    else:
        setup_logging(log_level="INFO")
    
    logger.info(f"Starting whodunit evaluation from: {input_path}")
    logger.info(f"Range: {range_spec}, Model: {model}, Prompt: {prompt_name}, Scope: {scope}")
    
    if scope == "item" and not item_ids:
        raise ValidationError("--item-ids is required when scope is 'item'")
    
    if scope == "item":
        logger.info(f"Processing {len(item_ids)} items for whodunit evaluation")
    
    start_time = time.time()
    
    try:
        # Build the command that was run
        command_parts = ["python -m ius whodunit", f"--input {input_path}", f"--range {range_spec}"]
        if prompt_name != "default-whodunit-culprits-and-accomplices":
            command_parts.append(f"--prompt {prompt_name}")
        if model != "gpt-4o-mini":
            command_parts.append(f"--model {model}")
        if temperature != 0.1:
            command_parts.append(f"--temperature {temperature}")
        if max_tokens != 2000:
            command_parts.append(f"--max-tokens {max_tokens}")
        if scope == "item":
            command_parts.extend(["--scope item", f"--item-ids {' '.join(item_ids)}"])
        if output_path:
            command_parts.append(f"--output {output_path}")
        if ask_user_confirmation:
            command_parts.append("--confirm")
        if verbose:
            command_parts.append("--verbose")
        
        command_run = " ".join(command_parts)
        
        # Run the evaluation
        result = run_whodunit_evaluation(
            input_dir=input_path,
            range_spec=range_spec,
            prompt_name=prompt_name,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            item_ids=item_ids if scope == "item" else None,
            output_dir=output_path,
            ask_user_confirmation=ask_user_confirmation,
            verbose=verbose
        )
        
        # Add command_run to the results
        if "collection_metadata" in result:
            result["collection_metadata"]["whodunit_evaluation_info"]["collection_metadata"]["command_run"] = command_run
        
        # Add command_run to individual item results
        for item_result in result.get("results", []):
            if "evaluation_metadata" in item_result:
                item_result["evaluation_metadata"]["command_run"] = command_run
        
        processing_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("WHODUNIT EVALUATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Items processed: {result['collection_metadata']['whodunit_evaluation_info']['processing_stats']['successful_items']}/{result['collection_metadata']['whodunit_evaluation_info']['processing_stats']['total_items']}")
        logger.info(f"Failed items: {result['collection_metadata']['whodunit_evaluation_info']['processing_stats']['failed_items']}")
        logger.info(f"Total cost: ${result['total_cost']:.4f}")
        logger.info(f"Total tokens: {result['total_tokens']}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Output saved to: {result['output_dir']}")
        
        if result['collection_metadata']['whodunit_evaluation_info']['processing_stats']['failed_items'] > 0:
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
        help="Range specification for text selection: 'all', 'last', 'penultimate', '1', '1-3', etc. (default: all)"
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
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    if not input_path.is_dir():
        logger.error(f"Input path must be a directory: {args.input}")
        sys.exit(1)
    
    # Validate scope and item_ids
    if args.scope == "item" and not args.item_ids:
        logger.error("--item-ids is required when scope is 'item'")
        sys.exit(1)
    
    try:
        # Run the evaluation
        result = evaluate_whodunit_dataset(
            input_path=str(input_path),
            range_spec=args.range,
            prompt_name=args.prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            scope=args.scope,
            item_ids=args.item_ids,
            output_path=args.output,
            ask_user_confirmation=args.confirm,
            verbose=args.verbose
        )
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()