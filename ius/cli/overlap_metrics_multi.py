#!/usr/bin/env python3
"""
Multi-range overlap metrics evaluation CLI.

This script runs overlap metrics evaluation (ROUGE or SUPERT) for ranges 1, 2, 3, ..., up to max_range
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

from ius.cli.overlap_metrics import evaluate_dataset
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

def run_multi_range_overlap_evaluation(
    input_path: str,
    metric_type: str,        # "rouge" or "supert"
    max_range: int = None,
    conda_env: str = "supert",
    output_dir: str = None,
    overwrite: bool = False,
    verbose: bool = False,
    add_reveal: bool = False,
    reveal_only: bool = False,
    stop: int = None
) -> str:
    """
    Run overlap metrics evaluation for multiple ranges (1, 2, 3, ..., max_range).
    
    Args:
        input_path: Path to summary collection
        metric_type: Type of metric to evaluate ("rouge" or "supert")
        max_range: Maximum range to evaluate up to (None to use all available ranges)
        conda_env: Conda environment for SUPERT (ignored for ROUGE)
        output_dir: Optional custom output directory name
        overwrite: Whether to overwrite existing results
        verbose: Enable verbose logging
        add_reveal: Whether to append reveal text to source documents
        reveal_only: Whether to use only reveal text as source documents
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
    
    # Validate metric type
    if metric_type not in ["rouge", "supert"]:
        raise ValueError(f"Invalid metric_type: {metric_type}. Must be 'rouge' or 'supert'")
    
    # Scan items to get per-item max ranges
    item_max_ranges = scan_item_max_ranges(input_path)
    if not item_max_ranges:
        raise ValueError("No valid items found for processing")
    
    # Generate output directory name
    if output_dir is None:
        import hashlib
        hash_parameters = {
            "metric": metric_type,
            "multi_range": True,
            "add_reveal": add_reveal,
            "reveal_only": reveal_only
        }
        
        # Add stemming parameter to hash for ROUGE
        if metric_type == "rouge":
            hash_parameters["use_stemmer"] = True  # Current default
        else:  # supert
            hash_parameters["conda_env"] = conda_env
            
        param_str = json.dumps(hash_parameters, sort_keys=True)
        hash_value = hashlib.md5(param_str.encode()).hexdigest()[:6]
        input_basename = os.path.basename(input_path.rstrip('/'))
        output_dir = f"{input_basename}_{metric_type}_multi_{hash_value}"
    
    # Create main output directory
    output_path = Path(f"outputs/eval/intrinsic/{metric_type}/{output_dir}")
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
                temp_base_dir = Path.home() / "ius-temp" / f"{metric_type}-single"
                temp_base_dir.mkdir(parents=True, exist_ok=True)
                
                # Calculate relative path to ~/ius-temp/{metric_type}-single/
                current_dir = Path.cwd() / "outputs" / "eval" / "intrinsic" / metric_type
                rel_path = os.path.relpath(temp_base_dir, current_dir)
                single_item_output_dir = f"{rel_path}/single_{item_id}_range_{range_num}_{hash_value}"
                
                # Run single-item, single-range evaluation 
                temp_results = evaluate_dataset(
                    input_path=input_path,
                    metric_type=metric_type,
                    range_spec=str(range_num),
                    conda_env=conda_env,
                    output_dir=single_item_output_dir,
                    overwrite=True,
                    verbose=False,  # Keep individual evaluations quiet
                    add_reveal=add_reveal,
                    reveal_only=reveal_only,
                    item_id=item_id  # Process only this item!
                )
                
                # Extract output path from results dict
                temp_output_path = temp_results["output_path"]
                
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
        f"{metric_type}_multi_evaluation_info": {
            "collection_metadata": {
                "evaluation_function": "run_multi_range_overlap_evaluation",
                "content_type": f"{metric_type}_multi_analysis",
                "input_type": "summaries",
                "metric": metric_type,
                "max_range": max_range,
                "source_collection": input_path,
                "command_run": f"python -m ius overlap-metrics-multi --input {input_path} --{metric_type}" + (f" --max-range {max_range}" if max_range is not None else ""),
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
    
    # Add metric-specific parameters to collection metadata
    if metric_type == "rouge":
        collection_data[f"{metric_type}_multi_evaluation_info"]["collection_metadata"]["use_stemmer"] = True
    else:  # supert
        collection_data[f"{metric_type}_multi_evaluation_info"]["collection_metadata"]["conda_env"] = conda_env
    
    collection_output_file = output_path / "collection.json"
    with open(collection_output_file, 'w') as f:
        json.dump(collection_data, f, indent=2)
    
    logger.info(f"Multi-range {metric_type.upper()} evaluation complete!")
    logger.info(f"Processed {successful_evaluations}/{total_evaluations} range evaluations successfully")
    logger.info(f"Skipped {skipped_evaluations} existing evaluations")
    logger.info(f"Results saved to: {output_path}")
    
    return str(output_path)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run overlap metrics evaluation (ROUGE or SUPERT) for multiple ranges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate ROUGE for ranges 1-5 for all items
  python -m ius overlap-metrics-multi --input outputs/summaries/bmds_summaries --rouge --max-range 5
  
  # Evaluate SUPERT for ranges 1-3 for all items
  python -m ius overlap-metrics-multi --input outputs/summaries/bmds_summaries --supert --max-range 3
  
  # Use specific conda environment for SUPERT
  python -m ius overlap-metrics-multi --input outputs/summaries/bmds_summaries --supert --max-range 3 --conda-env my_supert
  
  # Overwrite existing results
  python -m ius overlap-metrics-multi --input outputs/summaries/bmds_summaries --rouge --max-range 4 --overwrite
  
  # Include reveal text in source documents (for detective stories)
  python -m ius overlap-metrics-multi --input outputs/summaries/bmds_summaries --rouge --max-range 3 --add-reveal
  
  # Use only reveal text as source documents (for detective stories)  
  python -m ius overlap-metrics-multi --input outputs/summaries/bmds_summaries --supert --max-range 3 --reveal-only
  
  # Stop after processing 2 items per range (useful for testing)
  python -m ius overlap-metrics-multi --input outputs/summaries/bmds_summaries --rouge --max-range 3 --stop 2
        """
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Path to summary collection (must be in 'outputs/summaries')"
    )
    
    # Metric type selection (exactly one required)
    metric_group = parser.add_mutually_exclusive_group(required=True)
    metric_group.add_argument(
        "--supert",
        action="store_true",
        help="Use SUPERT metric (requires conda environment)"
    )
    metric_group.add_argument(
        "--rouge",
        action="store_true", 
        help="Use ROUGE metric (runs in current environment)"
    )
    
    parser.add_argument(
        "--max-range",
        type=int,
        help="Maximum range to evaluate (will process ranges 1, 2, 3, ..., max-range). If not provided, uses all available summary ranges."
    )
    
    parser.add_argument(
        "--conda-env",
        default="supert",
        help="Name of conda environment with SacreROUGE (for SUPERT only, default: supert)"
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
        help="Append reveal text to source documents (for bmds and true-detective datasets)"
    )
    
    parser.add_argument(
        "--reveal-only",
        action="store_true",
        help="Use only reveal text as source documents (for bmds and true-detective datasets)"
    )
    
    parser.add_argument(
        "--stop",
        type=int,
        help="Stop after processing this many items per range (useful for testing)"
    )
    
    args = parser.parse_args()
    
    # Determine metric type from flags
    if args.rouge:
        metric_type = "rouge"
    elif args.supert:
        metric_type = "supert"
    else:
        raise ValueError("Must specify either --rouge or --supert")
    
    try:
        output_dir = run_multi_range_overlap_evaluation(
            input_path=args.input,
            metric_type=metric_type,
            max_range=args.max_range,
            conda_env=args.conda_env,
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