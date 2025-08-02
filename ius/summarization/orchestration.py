"""
Summarization orchestration and experimental tracking for the IUS system.

This module provides high-level functions for organizing summarization experiments,
handling different scope options, and managing output with comprehensive metadata.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from .methods import no_op, concat_and_summarize, iterative_summarize
from ..data import ChunkedDataset, SummaryDataset
from ..logging_config import get_logger

logger = get_logger(__name__)


def summarize(strategy: str, dataset: str, scope: str, 
             chunked_file_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Main summarization orchestration function with experimental tracking.
    
    Args:
        strategy: Summarization strategy ("no_op", "concat_and_summarize", "iterative_summarize")
        dataset: Dataset name (e.g., "bmds", "true-detective")
        scope: Scope of summarization ("item", "dataset", "doc_range")
        chunked_file_path: Path to pre-chunked data file (optional, will use default if None)
        **kwargs: Additional parameters:
            - item_id: Required for "item" and "doc_range" scopes
            - doc_range: Required for "doc_range" scope (e.g., "0:2" or "1")
            - model: LLM model to use
            - system_and_user_prompt: Custom prompts
            - ask_user_confirmation: Whether to ask before API calls
            
    Returns:
        Dict with experiment results and metadata
    """
    # Validate inputs
    if scope in ["item", "doc_range"] and "item_id" not in kwargs:
        raise ValueError(f"item_id required for scope '{scope}'")
    if scope == "doc_range" and "doc_range" not in kwargs:
        raise ValueError("doc_range required for scope 'doc_range'")
    
    # Create experiment directory and metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{timestamp}_{strategy}_{scope}"
    experiment_dir = f"outputs/summaries/{experiment_id}"
    results_dir = f"{experiment_dir}/results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Load chunked data using new ChunkedDataset class
    if chunked_file_path is None:
        # Use default chunked directory - look for most recent in outputs/chunks/
        chunked_file_path = _find_default_chunked_directory(dataset)
    
    chunked_dataset = ChunkedDataset(chunked_file_path)
    
    # Initialize experiment config and metadata
    config = {
        "experiment_id": experiment_id,
        "strategy": strategy,
        "scope": scope,
        "dataset": dataset,
        "chunked_file_path": chunked_file_path,
        "timestamp": datetime.now().isoformat(),
        "kwargs": {k: v for k, v in kwargs.items() if k not in ["system_and_user_prompt"]},
        "model": kwargs.get("model", "gpt-4.1-mini")
    }
    
    summary_metadata = {}
    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "api_calls": 0
    }
    
    # Process based on scope using ChunkedDataset methods
    if scope == "item":
        # Single item processing
        item_id = kwargs["item_id"]
        chunks = chunked_dataset.get_chunks_for_scope(scope, item_id=item_id)
        
        # Apply strategy
        strategy_kwargs = {k: v for k, v in kwargs.items() if k not in ["item_id", "scope", "doc_range"]}
        result = _apply_strategy(strategy, chunks, **strategy_kwargs)
        
        # Save result
        _save_item_result(results_dir, item_id, result["response"])
        
        # Update metadata
        summary_metadata[item_id] = _extract_item_metadata_new(result, chunked_dataset, item_id, scope, kwargs.get("doc_range"))
        _update_total_usage(total_usage, result["usage"])
        config["total_items_processed"] = 1
        
    elif scope == "dataset":
        # Process all items in dataset
        item_ids = chunked_dataset.item_ids
        
        for item_id in item_ids:
            logger.info(f"Processing item {item_id} ({len(summary_metadata)+1}/{len(item_ids)})")
            
            try:
                chunks = chunked_dataset.get_chunks_for_scope("item", item_id=item_id)
                
                # Apply strategy
                strategy_kwargs = {k: v for k, v in kwargs.items() if k not in ["item_id", "scope", "doc_range"]}
                result = _apply_strategy(strategy, chunks, **strategy_kwargs)
                
                # Save result
                _save_item_result(results_dir, item_id, result["response"])
                
                # Update metadata
                summary_metadata[item_id] = _extract_item_metadata_new(result, chunked_dataset, item_id, "item", None)
                _update_total_usage(total_usage, result["usage"])
                
            except Exception as e:
                logger.error(f"Failed to process item {item_id}: {e}")
                summary_metadata[item_id] = {"error": str(e)}
        
        config["total_items_processed"] = len([m for m in summary_metadata.values() if "error" not in m])
        
    elif scope == "doc_range":
        # Document range within an item
        item_id = kwargs["item_id"]
        doc_range = kwargs["doc_range"]
        chunks = chunked_dataset.get_chunks_for_scope(scope, item_id=item_id, doc_range=doc_range)
        
        # Apply strategy
        strategy_kwargs = {k: v for k, v in kwargs.items() if k not in ["item_id", "scope", "doc_range"]}
        result = _apply_strategy(strategy, chunks, **strategy_kwargs)
        
        # Save result
        doc_range_str = str(doc_range).replace(":", "-").replace(",", "_")
        filename = f"{item_id}_docs_{doc_range_str}"
        _save_item_result(results_dir, filename, result["response"])
        
        # Update metadata
        summary_metadata[filename] = _extract_item_metadata_new(result, chunked_dataset, item_id, scope, doc_range)
        _update_total_usage(total_usage, result["usage"])
        config["total_items_processed"] = 1
    
    # Finalize config
    config["total_usage"] = total_usage
    
    # Save experiment files
    _save_experiment_config(experiment_dir, config)
    _save_summary_metadata(experiment_dir, summary_metadata)
    
    logger.info(f"Experiment completed: {experiment_id}")
    logger.info(f"Results saved to: {experiment_dir}")
    
    return {
        "experiment_id": experiment_id,
        "experiment_dir": experiment_dir,
        "config": config,
        "summary_metadata": summary_metadata,
        "total_usage": total_usage
    }


def _apply_strategy(strategy: str, chunks: List[str], **kwargs) -> Dict[str, Any]:
    """Apply the specified strategy to the given chunks."""
    if strategy == "no_op":
        return no_op(chunks, **kwargs)
    elif strategy == "concat_and_summarize":
        return concat_and_summarize(chunks, **kwargs)
    elif strategy == "iterative_summarize":
        return iterative_summarize(chunks, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def load_chunked_data(chunked_directory_path: str) -> ChunkedDataset:
    """
    Load pre-chunked data from outputs/chunks/ directory.
    
    Args:
        chunked_directory_path: Path to chunked directory
        
    Returns:
        ChunkedDataset instance
    """
    return ChunkedDataset(chunked_directory_path)





def _extract_item_metadata_new(result: Dict[str, Any], chunked_dataset: ChunkedDataset, 
                             item_id: str, scope: str, doc_range: Optional[str]) -> Dict[str, Any]:
    """
    Extract metadata for an item using the new ChunkedDataset.
    
    Args:
        result: Result from strategy application
        chunked_dataset: ChunkedDataset instance
        item_id: Item ID
        scope: Processing scope
        doc_range: Document range (if applicable)
        
    Returns:
        Metadata dictionary
    """
    # Get chunking stats from the dataset
    chunking_stats = chunked_dataset.get_item_stats(item_id)
    
    # Calculate chunk counts based on scope
    if scope == "item":
        chunks_processed = len(chunked_dataset.get_item_chunks(item_id))
    elif scope == "doc_range":
        chunks_processed = len(chunked_dataset.get_chunks_for_scope(scope, item_id=item_id, doc_range=doc_range))
    else:
        chunks_processed = len(chunked_dataset.get_item_chunks(item_id))
    
    return {
        "scope": scope,
        "doc_range": doc_range,
        "chunks_processed": chunks_processed,
        "chunking_stats": chunking_stats,
        "usage": result.get("usage", {}),
        "response_length": len(result.get("response", "")),
        "method": result.get("method", "unknown")
    }


def _find_default_chunked_directory(dataset: str) -> str:
    """Find the most recent chunked directory for a dataset."""
    chunks_dir = "outputs/chunks"
    if not os.path.exists(chunks_dir):
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    
    # Look for directories matching the pattern
    pattern = f"{dataset}_"
    matching_dirs = [d for d in os.listdir(chunks_dir) 
                    if d.startswith(pattern) and os.path.isdir(os.path.join(chunks_dir, d))]
    
    if not matching_dirs:
        raise FileNotFoundError(f"No chunked directories found for dataset {dataset} in {chunks_dir}")
    
    # Return the first match (could be enhanced to find most recent)
    default_dir = os.path.join(chunks_dir, matching_dirs[0])
    logger.info(f"Using default chunked directory: {default_dir}")
    return default_dir


def _update_total_usage(total_usage: Dict, item_usage: Dict) -> None:
    """Update total usage statistics with item usage."""
    if not item_usage:
        return
        
    total_usage["input_tokens"] += item_usage.get("input_tokens", 0)
    total_usage["output_tokens"] += item_usage.get("output_tokens", 0)
    total_usage["total_tokens"] += item_usage.get("total_tokens", 0)
    total_usage["input_cost"] += item_usage.get("input_cost", 0.0)
    total_usage["output_cost"] += item_usage.get("output_cost", 0.0)
    total_usage["total_cost"] += item_usage.get("total_cost", 0.0)
    total_usage["api_calls"] += 1


def _save_item_result(results_dir: str, item_id: str, response: str) -> None:
    """Save individual item result to file."""
    filename = f"{item_id}.txt"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(response)


def _save_experiment_config(experiment_dir: str, config: Dict) -> None:
    """Save experiment configuration."""
    config_path = os.path.join(experiment_dir, "config.json")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _save_summary_metadata(experiment_dir: str, metadata: Dict) -> None:
    """Save per-summary metadata."""
    metadata_path = os.path.join(experiment_dir, "summary_metadata.json")
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# Pipeline Integration Functions

def load_summary_experiment(experiment_id: str) -> SummaryDataset:
    """
    Load a completed summarization experiment as a SummaryDataset.
    
    Args:
        experiment_id: ID of the experiment (e.g., "20250801_210856_no_op_item")
        
    Returns:
        SummaryDataset instance for the experiment
        
    Raises:
        ValueError: If experiment directory doesn't exist
    """
    experiment_dir = f"outputs/summaries/{experiment_id}"
    
    if not os.path.exists(experiment_dir):
        raise ValueError(f"Experiment directory not found: {experiment_dir}")
    
    return SummaryDataset(experiment_dir)


def list_summary_experiments() -> List[str]:
    """
    List all available summarization experiments.
    
    Returns:
        List of experiment IDs
    """
    summaries_dir = "outputs/summaries"
    
    if not os.path.exists(summaries_dir):
        return []
    
    experiments = []
    for item in os.listdir(summaries_dir):
        experiment_path = os.path.join(summaries_dir, item)
        if os.path.isdir(experiment_path):
            # Check if it's a valid experiment (has config.json)
            config_file = os.path.join(experiment_path, "config.json")
            if os.path.exists(config_file):
                experiments.append(item)
    
    return sorted(experiments)


def get_pipeline_summary(dataset_name: str, chunk_experiment: str, 
                        summary_experiment: str) -> Dict[str, Any]:
    """
    Get a complete pipeline summary: Dataset -> ChunkedDataset -> SummaryDataset.
    
    Args:
        dataset_name: Name of the original dataset (e.g., "bmds")
        chunk_experiment: Name of the chunking experiment (e.g., "bmds_fixed_count_3")
        summary_experiment: ID of the summarization experiment
        
    Returns:
        Dictionary with pipeline statistics and metadata
    """
    try:
        # Load all three stages of the pipeline
        from ..data import Dataset
        
        original_dataset = Dataset(f"datasets/{dataset_name}")
        chunked_dataset = ChunkedDataset(f"outputs/chunks/{chunk_experiment}")
        summary_dataset = load_summary_experiment(summary_experiment)
        
        # Get collection-level stats
        pipeline_stats = {
            "dataset_name": dataset_name,
            "pipeline_stages": {
                "original": {
                    "name": original_dataset.name,
                    "items": len(original_dataset),
                    "metadata": original_dataset.metadata
                },
                "chunked": {
                    "name": chunked_dataset.name,
                    "items": len(chunked_dataset),
                    "metadata": chunked_dataset.metadata
                },
                "summarized": {
                    "name": summary_dataset.name,
                    "items": len(summary_dataset),
                    "metadata": summary_dataset.metadata
                }
            },
            "pipeline_consistency": {
                "all_stages_same_item_count": (
                    len(original_dataset) == len(chunked_dataset) == len(summary_dataset)
                ),
                "original_items": len(original_dataset),
                "chunked_items": len(chunked_dataset),
                "summarized_items": len(summary_dataset)
            }
        }
        
        return pipeline_stats
        
    except Exception as e:
        logger.error(f"Error creating pipeline summary: {e}")
        raise