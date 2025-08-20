"""
Command-line interface for overlap-based evaluation metrics (ROUGE and SUPERT).

Usage:
    python -m ius overlap_metrics --supert --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --range all
    python -m ius overlap_metrics --rouge --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --range penultimate
    python -m ius overlap_metrics --supert --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --conda-env my_supert
"""

import argparse
import sys
import time
import json
import os
import hashlib
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional
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


def get_dataset_source_path(dataset_name: str) -> Path:
    """Get source dataset path for a dataset."""
    dataset_mappings = {
        "bmds": "datasets/bmds",
        "true-detective": "datasets/true-detective", 
        "squality": "datasets/squality",
        "detectiveqa": "datasets/detectiveqa"
    }
    
    if dataset_name not in dataset_mappings:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(dataset_mappings.keys())}")
    
    return Path(dataset_mappings[dataset_name])


def parse_range_spec(range_spec: str, total_summaries: int) -> int:
    """Parse range specification to get summary index."""
    if range_spec == "all" or range_spec == "last":
        return total_summaries - 1
    elif range_spec == "penultimate":
        if total_summaries < 2:
            return 0  # Use first if only one summary
        return total_summaries - 2
    elif range_spec == "all-but-last":
        # Return last available index that's not the final one
        if total_summaries < 2:
            return 0
        return total_summaries - 2
    elif range_spec.isdigit():
        n = int(range_spec)
        if n < 1 or n > total_summaries:
            raise ValueError(f"Range index {n} out of bounds (1 to {total_summaries})")
        return n - 1  # Convert to 0-based indexing
    else:
        # Handle ranges like "1-3" - for now just take the last number
        if "-" in range_spec:
            try:
                parts = range_spec.split("-")
                end_idx = int(parts[1])
                if end_idx < 1 or end_idx > total_summaries:
                    raise ValueError(f"Range end index {end_idx} out of bounds (1 to {total_summaries})")
                return end_idx - 1
            except (ValueError, IndexError):
                raise ValueError(f"Invalid range format: {range_spec}")
        else:
            raise ValueError(f"Unknown range specification: {range_spec}")


def load_summaries_and_source_docs(summary_path: str, range_spec: str, add_reveal: bool = False, reveal_only: bool = False, item_id: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Load summaries and corresponding source documents.
    
    Args:
        summary_path: Path to the summary collection
        range_spec: Which summary to extract
        add_reveal: If True, append the reveal text to the source content
        reveal_only: If True, use only the reveal text as source content
        item_id: If provided, only process this specific item (for single-item processing)
    
    Returns:
        Dict mapping item_id to {"summary": text, "source": text}
    """
    summary_path_obj = Path(summary_path)
    items_dir = summary_path_obj / "items"
    
    if not items_dir.exists():
        raise ValidationError(f"Summary items directory not found: {items_dir}")
    
    # Extract dataset name and get source path
    dataset_name = extract_dataset_name(summary_path)
    source_path = get_dataset_source_path(dataset_name)
    
    logger.info(f"Loading summaries from: {summary_path}")
    logger.info(f"Loading source documents from: {source_path}")
    logger.info(f"Range specification: {range_spec}")
    
    data = {}
    
    for item_file in items_dir.glob("*.json"):
        with open(item_file) as f:
            item_data = json.load(f)
        
        current_item_id = item_data["item_metadata"]["item_id"]
        
        # Filter to specific item if requested (for single-item processing)
        if item_id and item_id != current_item_id:
            continue
        
        # Extract summary using range specification
        documents = item_data.get("documents", [])
        if not documents:
            logger.warning(f"No documents found in summary file: {item_file}")
            continue
            
        summaries = documents[0].get("summaries", [])
        if not summaries:
            logger.warning(f"No summaries found in: {item_file}")
            continue
            
        # Parse range specification
        try:
            summary_idx = parse_range_spec(range_spec, len(summaries))
            summary_text = summaries[summary_idx]
        except (IndexError, ValueError) as e:
            logger.warning(f"Could not extract summary for {current_item_id} with range {range_spec}: {e}")
            continue
        
        # Load corresponding source document
        source_file = source_path / "items" / f"{current_item_id}.json"
        if not source_file.exists():
            logger.warning(f"Source file not found for {current_item_id}: {source_file}")
            continue
            
        with open(source_file) as f:
            source_data = json.load(f)
            
        source_docs = source_data.get("documents", [])
        if not source_docs:
            logger.warning(f"No documents found in source file: {source_file}")
            continue
            
        source_content = source_docs[0].get("content", "")
        if not source_content:
            logger.warning(f"No content found in source file: {source_file}")
            continue
        
        # Check for conflicting flags
        if add_reveal and reveal_only:
            raise ValueError("Cannot use both --add-reveal and --reveal-only flags together")
        
        # Add reveal text if requested
        if add_reveal:
            reveal_text = ""
            try:
                metadata = source_docs[0].get("metadata", {})
                
                if dataset_name == "bmds":
                    # For bmds: documents[0]['metadata']['detection']['reveal_segment']
                    reveal_text = metadata.get("detection", {}).get("reveal_segment", "")
                elif dataset_name == "true-detective":
                    # For true-detective: documents[0]['metadata']['original_metadata']['reveal_text']
                    reveal_text = metadata.get("original_metadata", {}).get("reveal_text", "")
                else:
                    raise ValueError(f"Unknown dataset '{dataset_name}' - cannot extract reveal text")
                
                if reveal_text:
                    source_content = source_content + "\n\n" + reveal_text
                    logger.debug(f"Added reveal text to {current_item_id} (length: {len(reveal_text)} chars)")
                else:
                    logger.warning(f"No reveal text found for {current_item_id}")
                    raise ValueError(f"No reveal text found for {current_item_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to extract reveal text for {current_item_id}: {e}")
                raise ValueError(f"Failed to extract reveal text for {current_item_id}: {e}")
        
        # Use only reveal text if requested
        if reveal_only:
            reveal_text = ""
            try:
                metadata = source_docs[0].get("metadata", {})
                
                if dataset_name == "bmds":
                    # For bmds: documents[0]['metadata']['detection']['reveal_segment']
                    reveal_text = metadata.get("detection", {}).get("reveal_segment", "")
                elif dataset_name == "true-detective":
                    # For true-detective: documents[0]['metadata']['original_metadata']['reveal_text']
                    reveal_text = metadata.get("original_metadata", {}).get("reveal_text", "")
                else:
                    raise ValueError(f"Unknown dataset '{dataset_name}' - cannot extract reveal text")
                
                if reveal_text:
                    source_content = reveal_text
                    logger.debug(f"Using only reveal text for {current_item_id} (length: {len(reveal_text)} chars)")
                else:
                    logger.warning(f"No reveal text found for {current_item_id}")
                    raise ValueError(f"No reveal text found for {current_item_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to extract reveal text for {current_item_id}: {e}")
                raise ValueError(f"Failed to extract reveal text for {current_item_id}: {e}")
        
        data[current_item_id] = {
            "summary": summary_text,
            "source": source_content
        }
    
    logger.info(f"Loaded {len(data)} summary-source pairs")
    return data


def run_metric_evaluation_subprocess(data: Dict[str, Dict[str, str]], metric_type: str, conda_env: str = "supert") -> Dict[str, float]:
    """Run SUPERT or ROUGE evaluation using SacreROUGE."""
    
    # For ROUGE, we don't need conda environment verification
    if metric_type == "supert":
        # Verify conda environment exists for SUPERT
        result = subprocess.run(
            ['conda', 'env', 'list'], 
            capture_output=True, text=True
        )
        
        if conda_env not in result.stdout:
            raise ValueError(f"Conda environment '{conda_env}' not found. Please create it first.")
    
    # Write data to temporary file instead of using stdin
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f, indent=2)
        temp_file = f.name
    
    try:
        # Create Python script based on metric type
        if metric_type == "supert":
            script_content = f'''
import os
import sys
import json
from sacrerouge.metrics import SUPERT

# Set conda init for SUPERT
os.environ["CONDA_INIT"] = "/Users/juandiego/miniconda3/etc/profile.d/conda.sh"

# Initialize SUPERT with conda environment
supert = SUPERT(environment_name="{conda_env}")

# Read data from file
with open("{temp_file}", "r") as f:
    input_data = json.load(f)

results = {{}}

for item_id, item_data in input_data.items():
    summary = item_data["summary"]
    source = item_data["source"]
    
    try:
        # SUPERT expects a list of documents
        result = supert.score(summary, [source])
        results[item_id] = result["supert"]
    except Exception as e:
        print(f"Error scoring {{item_id}}: {{e}}", file=sys.stderr)
        results[item_id] = 0.0

print(json.dumps(results))
'''
        elif metric_type == "rouge":
            script_content = f'''
import sys
import json
from sacrerouge.metrics import Rouge
from rouge_score import rouge_scorer

# Initialize SacreROUGE for ROUGE-1, ROUGE-2, and ROUGE-L (original implementation)
sacrerouge = Rouge(max_ngram=2, compute_rouge_l=True, use_porter_stemmer=True)

# Initialize rouge-score for ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum (Google implementation)
rouge_score_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], 
                                             use_stemmer=True, split_summaries=True)

# Read data from file
with open("{temp_file}", "r") as f:
    input_data = json.load(f)

results = {{}}

for item_id, item_data in input_data.items():
    summary = item_data["summary"]
    reference = item_data["source"]  # For ROUGE, source acts as reference
    
    try:
        # Get SacreROUGE results (ROUGE-1, ROUGE-2)
        sacrerouge_result = sacrerouge.score(summary, [reference])
        
        # Get rouge-score results (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)
        rouge_score_result = rouge_score_scorer.score(reference, summary)
        
        # Combine results - start with SacreROUGE results (rounded to 2 decimal places)
        combined_result = {{}}
        for metric, scores in sacrerouge_result.items():
            combined_result[metric] = {{
                "precision": round(scores["precision"], 2),
                "recall": round(scores["recall"], 2),
                "f1": round(scores["f1"], 2)
            }}
        
        # Add rouge-score results with rs- prefix and proper format (rounded to 2 decimal places)
        # Multiply by 100 to convert from 0-1 scale to 0-100 scale (to match SacreROUGE)
        for metric, score in rouge_score_result.items():
            # Convert rouge-score format to match SacreROUGE format
            metric_name = f"rs-{{metric}}"  # Add rs- prefix (rouge-score)
            combined_result[metric_name] = {{
                "precision": round(score.precision * 100, 2),
                "recall": round(score.recall * 100, 2),
                "f1": round(score.fmeasure * 100, 2)
            }}
        
        results[item_id] = combined_result
    except Exception as e:
        print(f"Error scoring {{item_id}}: {{e}}", file=sys.stderr)
        results[item_id] = {{}}

print(json.dumps(results))
'''
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        logger.info(f"Running {metric_type.upper()} evaluation on {len(data)} items" + (f" using conda environment: {conda_env}" if metric_type == "supert" else ""))
        logger.debug(f"Data file: {temp_file}")
        
        # Run the script
        if metric_type == "supert":
            # SUPERT needs conda environment
            cmd = ['conda', 'run', '-n', conda_env, 'python', '-c', script_content]
        else:
            # ROUGE can run in current environment
            cmd = ['python', '-c', script_content]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"{metric_type.upper()} subprocess failed: {result.stderr}")
            raise RuntimeError(f"{metric_type.upper()} evaluation failed: {result.stderr}")
        
        scores = json.loads(result.stdout)
        logger.info(f"Successfully evaluated {len(scores)} summaries with {metric_type.upper()}")
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


def save_metric_results(
    scores: Dict[str, Any], 
    data: Dict[str, Dict[str, str]],
    output_path: Path,
    input_path: str,
    range_spec: str,
    metric_type: str,
    conda_env: str,
    command_run: str
) -> None:
    """Save metric results (SUPERT or ROUGE) to output directory."""
    
    # Extract numeric scores for statistics
    # For SUPERT: scores are floats directly
    # For ROUGE: scores are dicts, we'll use the main metric (rouge-l f1)
    numeric_scores = []
    for item_id, score in scores.items():
        if score is not None:
            if isinstance(score, dict):  # ROUGE case
                # Use ROUGE-1 F1 score as the primary metric (most common)
                if 'rouge-1' in score and isinstance(score['rouge-1'], dict) and 'f1' in score['rouge-1']:
                    numeric_scores.append(score['rouge-1']['f1'])
                elif 'rouge-1' in score:  # Fallback if rouge-1 is a float directly
                    numeric_scores.append(score['rouge-1'])
                else:  # Fallback to any F1 score in nested dicts
                    for rouge_key, rouge_values in score.items():
                        if isinstance(rouge_values, dict) and 'f1' in rouge_values:
                            numeric_scores.append(rouge_values['f1'])
                            break
            elif isinstance(score, (int, float)):  # SUPERT case
                # SUPERT scores are typically 0-1, so 0.0 is valid
                if score >= 0:
                    numeric_scores.append(score)
    
    # Calculate statistics
    stats = {
        "total_items": len(data),
        "successful_items": len(numeric_scores),
        "failed_items": len(scores) - len(numeric_scores),
        "mean_score": float(sum(numeric_scores) / len(numeric_scores)) if numeric_scores else 0.0,
        "median_score": float(sorted(numeric_scores)[len(numeric_scores)//2]) if numeric_scores else 0.0,
        "min_score": float(min(numeric_scores)) if numeric_scores else 0.0,
        "max_score": float(max(numeric_scores)) if numeric_scores else 0.0,
    }
    
    # Save collection-level results
    collection_data = {
        "experiment_id": output_path.name,
        "created_at": datetime.now().isoformat(),
        "input_path": input_path,
        "dataset_name": extract_dataset_name(input_path),
        f"{metric_type}_evaluation_info": {
            "processing_stats": stats,
            "parameters": {
                "range_spec": range_spec,
                "conda_env_name": conda_env,
                "metric": metric_type,
                "implementation": "sacrerouge"
            },
            "command_run": command_run,
        },
        "statistics": stats,
    }
    
    with open(output_path / "collection.json", 'w') as f:
        json.dump(collection_data, f, indent=2)
    
    # Save individual item results
    items_dir = output_path / "items"
    items_dir.mkdir(exist_ok=True)
    
    for item_id, score in scores.items():
        item_result = {
            "item_id": item_id,
            f"{metric_type}_score": score,  # Store complete score (dict for ROUGE, float for SUPERT)
            "summary_text": data.get(item_id, {}).get("summary", ""),
            "parameters": {
                "conda_env_name": conda_env,
                "metric": metric_type,
                "implementation": "sacrerouge"
            }
        }
        
        with open(items_dir / f"{item_id}.json", 'w') as f:
            json.dump(item_result, f, indent=2)


def evaluate_dataset(
    input_path: str,
    metric_type: str,
    range_spec: str = "all",
    conda_env: str = "supert", 
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = False,
    stop_after: Optional[int] = None,
    add_reveal: bool = False,
    reveal_only: bool = False,
    item_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run SUPERT or ROUGE evaluation on a summary dataset.

    Args:
        input_path: Path to summary collection (must start with "outputs/summaries")
        metric_type: Type of metric to use ("supert" or "rouge")
        range_spec: Which summary parts to use ("all", "penultimate", "last", etc.)
        conda_env: Name of conda environment with SacreROUGE (for SUPERT only)
        output_dir: Optional custom output directory name
        overwrite: Whether to overwrite existing results
        verbose: Enable verbose logging
        stop_after: Optional limit on number of items to process (for testing)
        add_reveal: Whether to append reveal text to source documents
        reveal_only: Whether to use only reveal text as source documents
        item_id: If provided, only process this specific item (for single-item processing)

    Returns:
        Dictionary with evaluation results and statistics
    """
    # Validate input path
    _validate_input_path(input_path)
    
    if not input_path.startswith("outputs/summaries"):
        raise ValidationError("Input path must start with 'outputs/summaries'")
    
    logger.info(f"Starting {metric_type.upper()} evaluation")
    logger.info(f"Input: {input_path}")
    logger.info(f"Range: {range_spec}")
    logger.info(f"Conda environment: {conda_env}")
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
                "metric_type": metric_type,
                "sacrerouge_version": "sacrerouge",
                "add_reveal": add_reveal,
                "reveal_only": reveal_only
            }
            
            # Only include conda_env for SUPERT
            if metric_type == "supert":
                hash_parameters["conda_env"] = conda_env
            
            hash_value = generate_output_hash(hash_parameters)
            input_basename = os.path.basename(input_path.rstrip('/'))
            output_dir = f"{input_basename}_{metric_type}_{hash_value}"
        
        # Create output directory - separate by metric type
        output_path = Path("outputs") / "eval" / "intrinsic" / metric_type / output_dir
        
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
        
        # Load summaries and source documents
        data = load_summaries_and_source_docs(input_path, range_spec, add_reveal, reveal_only, item_id)
        
        if not data:
            raise ValidationError("No valid summary-source pairs found")
        
        # Apply stop_after limit if specified
        if stop_after and stop_after < len(data):
            logger.info(f"Limiting evaluation to first {stop_after} items")
            data = dict(list(data.items())[:stop_after])
        
        # Run metric evaluation
        scores = run_metric_evaluation_subprocess(data, metric_type, conda_env)
        
        # Capture the command that was run
        argv_copy = sys.argv.copy()
        if argv_copy[0].endswith("__main__.py"):
            argv_copy[0] = "python -m ius"
        command_run = " ".join(argv_copy)
        
        # Save results
        save_metric_results(scores, data, output_path, input_path, range_spec, metric_type, conda_env, command_run)

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"{metric_type.upper()} evaluation completed in {duration:.2f} seconds")
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
            "mean_score": stats.get("mean_score", 0.0),
            "statistics": stats
        }
        
        logger.info(f"Summary - Processed: {summary['successful_items']}/{summary['total_items']} items")
        logger.info(f"Failed: {summary['failed_items']}")
        logger.info(f"Mean SUPERT score: {summary['mean_score']:.4f}")
        
        return summary

    except Exception as e:
        logger.error(f"SUPERT evaluation failed: {e}")
        raise


def main():
    """Main entry point for SUPERT/ROUGE CLI."""
    parser = argparse.ArgumentParser(
        description="Run overlap-based metrics (SUPERT or ROUGE) on summary datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic SUPERT evaluation with default settings
  python -m ius overlap_metrics --supert --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac

  # Basic ROUGE evaluation 
  python -m ius overlap_metrics --rouge --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac

  # Use different summary range
  python -m ius overlap_metrics --supert --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --range penultimate

  # Custom conda environment for SUPERT  
  python -m ius overlap_metrics --supert --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --conda-env my_supert

  # Process only first 10 items for testing
  python -m ius overlap_metrics --rouge --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --stop 10

  # Include reveal text in source documents (for detective stories)
  python -m ius overlap_metrics --supert --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --add-reveal

  # Use only reveal text as source documents (for detective stories)
  python -m ius overlap_metrics --rouge --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac --reveal-only

Range specifications:
  all             Use last summary (default)
  last            Use last summary  
  penultimate     Use second-to-last summary
  all-but-last    Use second-to-last summary
  1               Use first summary
  2               Use second summary
  1-3             Use third summary (last number in range)
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
        "--supert",
        action="store_true",
        help="Use SUPERT metric (requires conda environment)",
    )
    metric_group.add_argument(
        "--rouge",
        action="store_true", 
        help="Use ROUGE metric (runs in current environment)",
    )

    # Optional arguments
    parser.add_argument(
        "--range",
        default="all",
        help="Which summary to evaluate (default: all)",
    )

    parser.add_argument(
        "--conda-env",
        default="supert", 
        help="Name of conda environment with SacreROUGE (for SUPERT only, default: supert)",
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
        help="Append reveal text to source documents (for bmds and true-detective datasets)",
    )

    parser.add_argument(
        "--reveal-only",
        action="store_true",
        help="Use only reveal text as source documents (for bmds and true-detective datasets)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)

    try:
        # Determine metric type
        if args.supert:
            metric_type = "supert"
        elif args.rouge:
            metric_type = "rouge"
        else:
            # This shouldn't happen due to required=True in mutually_exclusive_group
            logger.error("No metric type specified. Use --supert or --rouge")
            sys.exit(1)
        
        # Run evaluation
        results = evaluate_dataset(
            input_path=args.input,
            metric_type=metric_type,
            range_spec=args.range,
            conda_env=args.conda_env,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            verbose=args.verbose,
            stop_after=args.stop,
            add_reveal=args.add_reveal,
            reveal_only=args.reveal_only,
        )

        logger.info(f"{metric_type.upper()} evaluation completed successfully!")
        
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