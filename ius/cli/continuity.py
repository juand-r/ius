"""
Command-line interface for continuity evaluation metrics (ROUGE and BERTScore).

Usage:
    python -m ius continuity --rouge --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --range all
    python -m ius continuity --rouge --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --range all-but-last
    python -m ius continuity --bertscore --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --range all
"""

import argparse
import sys
import time
import json
import os
import hashlib
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from ius.exceptions import ValidationError
from ius.logging_config import get_logger, setup_logging

# Set up logger for this module
logger = get_logger(__name__)


def _validate_input_path(input_path: str) -> None:
    """Validate that input path exists and is a directory."""
    path = Path(input_path)
    if not path.exists():
        raise ValidationError(f"Input path does not exist: {input_path}")
    if not path.is_dir():
        raise ValidationError(f"Input path must be a directory: {input_path}")


def extract_dataset_name(input_path: str) -> str:
    """Extract dataset name from input path."""
    input_path = input_path.rstrip('/')
    last_dir = os.path.basename(input_path)
    dataset_name = last_dir.split("_")[0]
    return dataset_name


def load_summaries_for_continuity(summary_path: str, range_spec: str, item_id: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Load summaries for continuity evaluation.
    
    Args:
        summary_path: Path to the summary collection
        range_spec: Which summaries to include ('all' or 'all-but-last')
        item_id: If provided, only process this specific item (for single-item processing)
    
    Returns:
        Dict mapping item_id to list of summary texts
    """
    summary_path_obj = Path(summary_path)
    items_dir = summary_path_obj / "items"
    
    if not items_dir.exists():
        raise ValidationError(f"Summary items directory not found: {items_dir}")
    
    # Extract dataset name
    dataset_name = extract_dataset_name(summary_path)
    
    logger.info(f"Loading summaries from: {summary_path}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Range specification: {range_spec}")
    
    # Validate range specification
    if range_spec not in ['all', 'all-but-last']:
        raise ValueError(f"Unsupported range specification: {range_spec}. Supported: 'all', 'all-but-last'")
    
    data = {}
    
    for item_file in items_dir.glob("*.json"):
        with open(item_file) as f:
            item_data = json.load(f)
        
        current_item_id = item_data["item_metadata"]["item_id"]
        
        # Filter to specific item if requested (for single-item processing)
        if item_id and item_id != current_item_id:
            continue
        
        # Extract summaries
        documents = item_data.get("documents", [])
        if not documents:
            logger.warning(f"No documents found in summary file: {item_file}")
            continue
            
        summaries = documents[0].get("summaries", [])
        if not summaries:
            logger.warning(f"No summaries found in: {item_file}")
            continue
            
        # Apply range specification
        if range_spec == 'all':
            selected_summaries = summaries
        elif range_spec == 'all-but-last':
            if len(summaries) <= 1:
                logger.warning(f"Item {current_item_id} has only {len(summaries)} summaries, skipping for 'all-but-last' range")
                continue
            selected_summaries = summaries[:-1]
        
        # Need at least 2 summaries for continuity evaluation
        if len(selected_summaries) < 2:
            logger.warning(f"Item {current_item_id} has only {len(selected_summaries)} selected summaries, need at least 2 for continuity")
            continue
            
        data[current_item_id] = selected_summaries
    
    logger.info(f"Loaded {len(data)} items with sufficient summaries for continuity evaluation")
    return data


def run_continuity_evaluation_subprocess(data: Dict[str, List[str]], metric_type: str) -> Dict[str, Dict[str, Any]]:
    """Run continuity evaluation using rouge-score or BERTScore."""
    
    if metric_type == "bertscore":
        raise NotImplementedError("BERTScore continuity evaluation is not yet implemented")
    
    # Write data to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f, indent=2)
        temp_file = f.name
    
    try:
        # Create Python script for ROUGE continuity evaluation
        if metric_type == "rouge":
            script_content = f'''
import sys
import json
import numpy as np
from rouge_score import rouge_scorer

# Initialize rouge-score for ROUGE-2, ROUGE-L, and ROUGE-Lsum
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge2', 'rougeL', 'rougeLsum'], 
                                           use_stemmer=True, split_summaries=True)

# Read data from file
with open("{temp_file}", "r") as f:
    input_data = json.load(f)

results = {{}}

for item_id, summaries in input_data.items():
    try:
        # Compute pairwise ROUGE scores between consecutive summaries
        pairwise_scores = []
        
        for i in range(len(summaries) - 1):
            summary_1 = summaries[i]
            summary_2 = summaries[i + 1]
            
            # Compute ROUGE scores between consecutive summaries
            rouge_result = rouge_scorer_obj.score(summary_1, summary_2)
            
            # Format pair results (multiply by 100 to match overlap_metrics.py scale)
            pair_result = {{}}
            for metric, score in rouge_result.items():
                pair_result[metric] = {{
                    "precision": round(score.precision * 100, 2),
                    "recall": round(score.recall * 100, 2),
                    "f1": round(score.fmeasure * 100, 2)
                }}
            
            pairwise_scores.append({{
                "pair": f"{{i+1}}-{{i+2}}",
                "scores": pair_result
            }})
        
        # Calculate averages across all pairs for each metric
        continuity_averages = {{}}
        
        if pairwise_scores:
            metrics = ['rouge2', 'rougeL', 'rougeLsum']
            score_types = ['precision', 'recall', 'f1']
            
            for metric in metrics:
                continuity_averages[f"{{metric}}-continuity"] = {{}}
                
                for score_type in score_types:
                    scores_for_averaging = []
                    for pair_data in pairwise_scores:
                        pair_scores = pair_data['scores']
                        if metric in pair_scores and score_type in pair_scores[metric]:
                            scores_for_averaging.append(pair_scores[metric][score_type])
                    
                    if scores_for_averaging:
                        avg_score = round(np.mean(scores_for_averaging), 2)
                        continuity_averages[f"{{metric}}-continuity"][score_type] = avg_score
        
        results[item_id] = {{
            "pairwise_scores": pairwise_scores,
            "continuity_averages": continuity_averages,
            "num_pairs": len(pairwise_scores),
            "num_summaries": len(summaries)
        }}
        
    except Exception as e:
        print(f"Error processing {{item_id}}: {{e}}", file=sys.stderr)
        results[item_id] = {{
            "pairwise_scores": [],
            "continuity_averages": {{}},
            "num_pairs": 0,
            "num_summaries": len(summaries),
            "error": str(e)
        }}

print(json.dumps(results))
'''
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        logger.info(f"Running {metric_type.upper()} continuity evaluation on {len(data)} items")
        logger.debug(f"Data file: {temp_file}")
        
        # Run the script in current environment (rouge-score doesn't need conda)
        cmd = ['python', '-c', script_content]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"{metric_type.upper()} subprocess failed: {result.stderr}")
            raise RuntimeError(f"{metric_type.upper()} continuity evaluation failed: {result.stderr}")
        
        scores = json.loads(result.stdout)
        logger.info(f"Successfully evaluated {len(scores)} items with {metric_type.upper()} continuity")
        return scores
        
    except subprocess.TimeoutExpired:
        logger.error(f"{metric_type.upper()} evaluation timed out after 1 hour")
        raise RuntimeError(f"{metric_type.upper()} evaluation timed out")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {metric_type.upper()} output: {e}")
        logger.error(f"Raw output: {result.stdout}")
        raise RuntimeError(f"Failed to parse {metric_type.upper()} output")
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except:
            pass


def generate_output_hash(hash_parameters: Dict[str, Any]) -> str:
    """Generate hash for output directory name."""
    param_str = json.dumps(hash_parameters, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:6]


def save_continuity_results(
    scores: Dict[str, Dict[str, Any]], 
    data: Dict[str, List[str]],
    output_path: Path,
    input_path: str,
    range_spec: str,
    metric_type: str,
    command_run: str
) -> None:
    """Save continuity evaluation results to output directory."""
    
    # Calculate collection-level statistics
    total_items = len(data)
    successful_items = len([item for item, result in scores.items() if result.get("num_pairs", 0) > 0])
    failed_items = total_items - successful_items
    
    # Calculate statistics for each continuity metric
    continuity_stats = {}
    
    if successful_items > 0:
        # Collect all continuity averages across items
        all_continuity_scores = {}
        
        for item_result in scores.values():
            continuity_averages = item_result.get("continuity_averages", {})
            for metric_name, metric_scores in continuity_averages.items():
                if metric_name not in all_continuity_scores:
                    all_continuity_scores[metric_name] = {}
                for score_type, score_value in metric_scores.items():
                    if score_type not in all_continuity_scores[metric_name]:
                        all_continuity_scores[metric_name][score_type] = []
                    all_continuity_scores[metric_name][score_type].append(score_value)
        
        # Calculate statistics for each continuity metric
        for metric_name, metric_data in all_continuity_scores.items():
            continuity_stats[metric_name] = {}
            for score_type, score_values in metric_data.items():
                if score_values:
                    continuity_stats[metric_name][score_type] = {
                        "mean": round(sum(score_values) / len(score_values), 2),
                        "median": round(sorted(score_values)[len(score_values)//2], 2),
                        "min": round(min(score_values), 2),
                        "max": round(max(score_values), 2),
                    }
    
    # Save collection-level results
    collection_data = {
        "experiment_id": output_path.name,
        "created_at": datetime.now().isoformat(),
        "input_path": input_path,
        "dataset_name": extract_dataset_name(input_path),
        f"{metric_type}_continuity_evaluation_info": {
            "processing_stats": {
                "total_items": total_items,
                "successful_items": successful_items,
                "failed_items": failed_items,
            },
            "parameters": {
                "range_spec": range_spec,
                "metric": f"{metric_type}-continuity",
                "implementation": "rouge-score" if metric_type == "rouge" else metric_type
            },
            "command_run": command_run,
            "continuity_statistics": continuity_stats
        },
        "statistics": {
            "total_items": total_items,
            "successful_items": successful_items,
            "failed_items": failed_items,
        },
    }
    
    with open(output_path / "collection.json", 'w') as f:
        json.dump(collection_data, f, indent=2)
    
    # Save individual item results
    items_dir = output_path / "items"
    items_dir.mkdir(exist_ok=True)
    
    for item_id, result in scores.items():
        item_result = {
            "item_id": item_id,
            f"{metric_type}_continuity_results": result,
            "parameters": {
                "metric": f"{metric_type}-continuity",
                "implementation": "rouge-score" if metric_type == "rouge" else metric_type,
                "range_spec": range_spec
            }
        }
        
        with open(items_dir / f"{item_id}.json", 'w') as f:
            json.dump(item_result, f, indent=2)


def evaluate_dataset(
    input_path: str,
    metric_type: str,
    range_spec: str = "all",
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = False,
    stop_after: Optional[int] = None,
    item_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run continuity evaluation (ROUGE or BERTScore) on a summary dataset.

    Args:
        input_path: Path to summary collection (must start with "outputs/summaries")
        metric_type: Type of metric to use ("rouge" or "bertscore")
        range_spec: Which summaries to include ("all" or "all-but-last")
        output_dir: Optional custom output directory name
        overwrite: Whether to overwrite existing results
        verbose: Enable verbose logging
        stop_after: Optional limit on number of items to process (for testing)
        item_id: If provided, only process this specific item (for single-item processing)

    Returns:
        Dictionary with evaluation results and statistics
    """
    # Validate input path
    _validate_input_path(input_path)
    
    if not input_path.startswith("outputs/summaries"):
        raise ValidationError("Input path must start with 'outputs/summaries'")
    
    logger.info(f"Starting {metric_type.upper()} continuity evaluation")
    logger.info(f"Input: {input_path}")
    logger.info(f"Range: {range_spec}")
    if output_dir:
        logger.info(f"Output directory: {output_dir}")
    logger.info(f"Overwrite: {overwrite}")
    if stop_after:
        logger.info(f"Stop after: {stop_after} items")

    start_time = time.time()

    try:
        # Generate output directory name if not provided
        if output_dir is None:
            hash_parameters = {
                "range_spec": range_spec,
                "metric_type": f"{metric_type}-continuity",
                "implementation": "rouge-score" if metric_type == "rouge" else metric_type
            }
            
            hash_value = generate_output_hash(hash_parameters)
            input_basename = os.path.basename(input_path.rstrip('/'))
            output_dir = f"{input_basename}_{metric_type}-continuity_{hash_value}"
        
        # Create output directory - separate by metric type
        output_path = Path("outputs") / "eval" / "intrinsic" / f"{metric_type}-continuity" / output_dir
        
        if output_path.exists() and not overwrite:
            logger.info(f"Output directory already exists: {output_path}")
            logger.info("Use --overwrite to regenerate results")
            
            # Load existing results
            collection_file = output_path / "collection.json"
            if collection_file.exists():
                with open(collection_file, 'r') as f:
                    existing_results = json.load(f)
                return {
                    "output_path": str(output_path), 
                    "duration": 0,
                    "statistics": existing_results.get("statistics", {})
                }
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load summaries for continuity evaluation
        data = load_summaries_for_continuity(input_path, range_spec, item_id)
        
        if not data:
            raise ValidationError("No valid items found with sufficient summaries for continuity evaluation")
        
        # Apply stop_after limit if specified
        if stop_after and stop_after < len(data):
            logger.info(f"Limiting evaluation to first {stop_after} items")
            data = dict(list(data.items())[:stop_after])
        
        # Run continuity evaluation
        scores = run_continuity_evaluation_subprocess(data, metric_type)
        
        # Capture the command that was run
        argv_copy = sys.argv.copy()
        if argv_copy[0].endswith("__main__.py"):
            argv_copy[0] = "python -m ius"
        command_run = " ".join(argv_copy)
        
        # Save results
        save_continuity_results(scores, data, output_path, input_path, range_spec, metric_type, command_run)

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"{metric_type.upper()} continuity evaluation completed in {duration:.2f} seconds")
        logger.info(f"Results saved to: {output_path}")

        # Load and return results summary
        collection_file = output_path / "collection.json"
        with open(collection_file, 'r') as f:
            results = json.load(f)
            
        # Extract key statistics
        stats = results.get("statistics", {})
        
        summary = {
            "output_path": str(output_path),
            "duration": duration,
            "total_items": stats.get("total_items", 0),
            "successful_items": stats.get("successful_items", 0),
            "failed_items": stats.get("failed_items", 0),
            "statistics": stats
        }
        
        logger.info(f"Summary - Processed: {summary['successful_items']}/{summary['total_items']} items")
        logger.info(f"Failed: {summary['failed_items']}")
        
        return summary

    except Exception as e:
        logger.error(f"{metric_type.upper()} continuity evaluation failed: {e}")
        raise


def main():
    """Main entry point for continuity evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Run continuity metrics (ROUGE or BERTScore) on summary datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic ROUGE continuity evaluation with all summaries
  python -m ius continuity --rouge --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --range all

  # ROUGE continuity evaluation excluding last summary (reveal)
  python -m ius continuity --rouge --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --range all-but-last

  # BERTScore continuity evaluation (not yet implemented)
  python -m ius continuity --bertscore --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --range all

  # Process only first 10 items for testing
  python -m ius continuity --rouge --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --range all --stop 10

Range specifications:
  all             Include all summaries (default)
  all-but-last    Include all summaries except the last one (exclude reveal)
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        required=True,
        help="Path to summary collection (must start with 'outputs/summaries')",
    )

    # Metric type selection (mutually exclusive)
    metric_group = parser.add_mutually_exclusive_group(required=True)
    metric_group.add_argument(
        "--rouge",
        action="store_true",
        help="Use ROUGE metric for continuity evaluation",
    )
    metric_group.add_argument(
        "--bertscore",
        action="store_true", 
        help="Use BERTScore metric for continuity evaluation (not yet implemented)",
    )

    # Optional arguments
    parser.add_argument(
        "--range",
        default="all",
        choices=["all", "all-but-last"],
        help="Which summaries to include in continuity evaluation (default: all)",
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

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)

    try:
        # Determine metric type
        if args.rouge:
            metric_type = "rouge"
        elif args.bertscore:
            metric_type = "bertscore"
        else:
            # This shouldn't happen due to required=True in mutually_exclusive_group
            logger.error("No metric type specified. Use --rouge or --bertscore")
            sys.exit(1)
        
        # Run evaluation
        results = evaluate_dataset(
            input_path=args.input,
            metric_type=metric_type,
            range_spec=args.range,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            verbose=args.verbose,
            stop_after=args.stop,
        )

        logger.info(f"{metric_type.upper()} continuity evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info(f"Evaluation interrupted by user")
        sys.exit(1)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()