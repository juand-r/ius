#!/usr/bin/env python3
"""
Multi-range entity coverage evaluation CLI.

This script runs entity coverage evaluation for ranges 1, 2, 3, ..., up to max_range
(or all available ranges if max_range not specified) for each item, creating a 
comprehensive evaluation across all summary lengths.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add the parent directory to the path so we can import from ius
sys.path.append(str(Path(__file__).parent.parent.parent))

from ius.eval.intrinsic.entity_coverage import run_entity_coverage_evaluation
from ius.exceptions import ValidationError
from ius.logging_config import setup_logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scan_item_max_ranges(input_path: str) -> dict:
    """Scan all items to determine the maximum range for each item."""
    input_path_obj = Path(input_path)
    items_dir = input_path_obj / "items"
    
    if not items_dir.exists():
        raise FileNotFoundError(f"Items directory not found: {items_dir}")
    
    item_max_ranges = {}
    item_files = list(items_dir.glob("*.json"))
    
    logger.info(f"Scanning {len(item_files)} items to determine max ranges...")
    
    for item_file in item_files:
        item_id = item_file.stem
        try:
            with open(item_file, 'r') as f:
                data = json.load(f)
            
            # Get summaries from documents[0]['summaries']
            documents = data.get("documents", [])
            if not documents:
                logger.warning(f"No documents found for {item_id}")
                continue
            
            summaries = documents[0].get("summaries", [])
            if not summaries:
                logger.warning(f"No summaries found for {item_id}")
                continue
            
            max_range = len(summaries)
            item_max_ranges[item_id] = max_range
            logger.debug(f"{item_id}: {max_range} summaries")
            
        except Exception as e:
            logger.error(f"Error processing {item_id}: {e}")
            continue
    
    logger.info(f"Found max ranges for {len(item_max_ranges)} items")
    return item_max_ranges

def run_multi_range_evaluation(
    input_path: str,
    max_range: int = None,
    model: str = "gpt-5-mini",
    prompt_name: str = "default-entity-matching",
    output_dir: str = None,
    overwrite: bool = False,
    verbose: bool = False,
    add_reveal: bool = False,
    reveal_only: bool = False,
    stop: int = None
) -> str:
    """
    Run entity coverage evaluation for multiple ranges (1, 2, 3, ..., max_range).
    
    Args:
        input_path: Path to summary collection
        max_range: Maximum range to evaluate up to (None to use all available ranges)
        model: LLM model for entity matching
        prompt_name: Name of matching prompt
        output_dir: Optional custom output directory name
        overwrite: Whether to overwrite existing results
        verbose: Enable verbose logging
        add_reveal: Whether to append reveal text to source documents
        stop: Stop after processing this many items (None to process all)
        
    Returns:
        Path to output directory
    """
    
    if verbose:
        setup_logging(log_level="DEBUG")
    
    # Validate input
    if not input_path.startswith("outputs/summaries"):
        raise ValueError("Can only be used to evaluate summaries, input must be in 'outputs/summaries'")
    
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # Scan items to get per-item max ranges
    item_max_ranges = scan_item_max_ranges(input_path)
    if not item_max_ranges:
        raise ValueError("No valid items found for processing")
    
    # Generate output directory name
    if output_dir is None:
        import hashlib
        hash_parameters = {
            "extraction_method": "spacy",
            "extraction_model": "en_core_web_lg",
            "model": model,
            "prompt_name": prompt_name,
            "temperature": 1.0,
            "max_completion_tokens": 10000,
            "multi_range": True,
            "add_reveal": add_reveal,
            "reveal_only": reveal_only
            # Note: max_range and stop are execution parameters, not evaluation parameters
            # They don't affect the quality of results, just which ranges/items to compute
        }
        param_str = json.dumps(hash_parameters, sort_keys=True)
        hash_value = hashlib.md5(param_str.encode()).hexdigest()[:6]
        input_basename = os.path.basename(input_path.rstrip('/'))
        output_dir = f"{input_basename}_entity_coverage_multi_{hash_value}"
    
    # Create main output directory
    output_path = Path(f"outputs/eval/intrinsic/entity-coverage/{output_dir}")
    output_path.mkdir(parents=True, exist_ok=True)
    items_path = output_path / "items"
    items_path.mkdir(exist_ok=True)
    
    # Efficient processing: process each item once for all its ranges
    total_evaluations = 0
    successful_evaluations = 0
    skipped_evaluations = 0
    failed_evaluations = 0
    processed_items = 0
    
    for item_id, item_max_range in item_max_ranges.items():
        # Check if we should stop after processing a certain number of items
        if stop is not None and processed_items >= stop:
            logger.info(f"Stopped after processing {processed_items} items (--stop {stop})")
            break
        
        processed_items += 1
        # Limit to user-specified max_range (if provided)
        if max_range is None:
            effective_max_range = item_max_range
        else:
            effective_max_range = min(item_max_range, max_range)
        
        # Create item subdirectory
        item_dir = items_path / item_id
        item_dir.mkdir(exist_ok=True)
        
        logger.info(f"Processing {item_id}: ranges 1-{effective_max_range}")
        
        for range_num in range(1, effective_max_range + 1):
            total_evaluations += 1
            range_output_file = item_dir / f"{range_num}.json"
            
            # Skip if already exists and not overwriting
            if range_output_file.exists() and not overwrite:
                logger.debug(f"Skipping {item_id} range {range_num} (already exists)")
                skipped_evaluations += 1
                continue
            
            try:
                logger.debug(f"Evaluating {item_id} range {range_num}")
                
                # Create a temporary single-item evaluation in ~/ius-temp/
                temp_base_dir = Path.home() / "ius-temp" / "entity-coverage-single"
                temp_base_dir.mkdir(parents=True, exist_ok=True)
                
                # Calculate relative path to ~/ius-temp/entity-coverage-single/
                current_dir = Path.cwd() / "outputs" / "eval" / "intrinsic" / "entity-coverage"
                rel_path = os.path.relpath(temp_base_dir, current_dir)
                single_item_output_dir = f"{rel_path}/single_{item_id}_range_{range_num}_{hash_value}"
                
                # Run single-item, single-range evaluation 
                temp_output_path = run_entity_coverage_evaluation(
                    input_path=input_path,
                    range_spec=str(range_num),
                    model=model,
                    prompt_name=prompt_name,
                    output_dir=single_item_output_dir,
                    overwrite=True,
                    add_reveal=add_reveal,
                    reveal_only=reveal_only,
                    item_id=item_id  # Process only this item!
                )
                
                # Move the result file to our structure
                temp_items_dir = Path(temp_output_path) / "items"
                temp_item_file = temp_items_dir / f"{item_id}.json"
                
                if temp_item_file.exists():
                    import shutil
                    shutil.move(str(temp_item_file), str(range_output_file))
                    successful_evaluations += 1
                    logger.debug(f"✓ {item_id} range {range_num} completed")
                else:
                    logger.error(f"Expected output file not found: {temp_item_file}")
                    failed_evaluations += 1
                
                # Clean up temporary directory
                import shutil
                temp_output_path_obj = Path(temp_output_path)
                if temp_output_path_obj.exists():
                    shutil.rmtree(temp_output_path_obj)
                    
            except Exception as e:
                logger.error(f"Failed to process {item_id} range {range_num}: {e}")
                failed_evaluations += 1
                continue
    
    # Create aggregate collection.json
    collection_data = {
        "entity_coverage_multi_evaluation_info": {
            "collection_metadata": {
                "evaluation_function": "run_multi_range_evaluation",
                "content_type": "entity_coverage_multi_analysis",
                "input_type": "summaries",
                "model": model,
                "prompt_name": prompt_name,
                "max_range": max_range,
                "temperature": 1.0,
                "max_completion_tokens": 10000,
                "extraction_method": "spacy",
                "extraction_model": "en_core_web_lg",
                "source_collection": input_path,
                "command_run": f"python -m ius entity-coverage-multi --input {input_path}" + (f" --max-range {max_range}" if max_range is not None else "") + f" --model {model}",
                "hash_parameters": hash_parameters,
                "hash_note": "Directory name contains 6-char MD5 hash of these parameters",
                "hash_value": hash_value,
                "multi_range": True
            },
            "timestamp": "2025-08-17T16:00:00.000000",  # TODO: Use actual timestamp
            "item_max_ranges": item_max_ranges,
            "processing_stats": {
                "total_items": len(item_max_ranges),
                "total_evaluations": total_evaluations,
                "successful_evaluations": successful_evaluations,
                "skipped_evaluations": skipped_evaluations,
                "failed_evaluations": failed_evaluations
            }
        }
    }
    
    collection_output_file = output_path / "collection.json"
    with open(collection_output_file, 'w') as f:
        json.dump(collection_data, f, indent=2)
    
    logger.info(f"Multi-range entity coverage evaluation complete!")
    logger.info(f"Processed {successful_evaluations}/{total_evaluations} range evaluations successfully")
    logger.info(f"Skipped {skipped_evaluations} existing evaluations")
    logger.info(f"Results saved to: {output_path}")
    
    return str(output_path)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run entity coverage evaluation for multiple ranges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate ranges 1-5 for all items
  python -m ius entity-coverage-multi --input outputs/summaries/bmds_summaries --max-range 5
  
  # Use specific model and prompt
  python -m ius entity-coverage-multi --input outputs/summaries/bmds_summaries --max-range 3 --model gpt-4o --prompt custom-prompt
  
  # Overwrite existing results
  python -m ius entity-coverage-multi --input outputs/summaries/bmds_summaries --max-range 4 --overwrite
  
  # Include reveal text in source documents (for detective stories)
  python -m ius entity-coverage-multi --input outputs/summaries/bmds_summaries --max-range 3 --add-reveal
  
  # Use only reveal text as source documents (for detective stories)  
  python -m ius entity-coverage-multi --input outputs/summaries/bmds_summaries --max-range 3 --reveal-only
  
  # Stop after processing 2 items per range (useful for testing)
  python -m ius entity-coverage-multi --input outputs/summaries/bmds_summaries --max-range 3 --stop 2
        """
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Path to summary collection (must be in 'outputs/summaries')"
    )
    
    parser.add_argument(
        "--max-range",
        type=int,
        help="Maximum range to evaluate (will process ranges 1, 2, 3, ..., max-range). If not provided, uses all available summary ranges."
    )
    
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="LLM model for entity matching (default: gpt-5-mini)"
    )
    
    parser.add_argument(
        "--prompt",
        default="default-entity-matching",
        help="Entity matching prompt name (default: default-entity-matching)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Custom output directory name (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--add-reveal",
        action="store_true",
        help="Append reveal text to source documents (for bmds, true-detective, and detectiveqa datasets)"
    )

    parser.add_argument(
        "--reveal-only",
        action="store_true",
        help="Use only reveal text as source documents (for bmds, true-detective, and detectiveqa datasets)"
    )
    
    parser.add_argument(
        "--stop",
        type=int,
        help="Stop after processing this many items per range (useful for testing)"
    )
    
    args = parser.parse_args()
    
    try:
        output_dir = run_multi_range_evaluation(
            input_path=args.input,
            max_range=args.max_range,
            model=args.model,
            prompt_name=args.prompt,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            verbose=args.verbose,
            add_reveal=args.add_reveal,
            reveal_only=args.reveal_only,
            stop=args.stop
        )
        print(f"Results saved to: {output_dir}")
        
    except (ValidationError, ValueError, FileNotFoundError) as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()