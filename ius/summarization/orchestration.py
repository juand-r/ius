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
    
    # Load chunked data
    if chunked_file_path is None:
        # Use default chunked file - look for most recent in outputs/chunks/
        chunked_file_path = _find_default_chunked_file(dataset)
    
    chunked_data = load_chunked_data(chunked_file_path)
    
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
    
    # Process based on scope
    if scope == "item":
        # Single item processing
        item_id = kwargs["item_id"]
        # Remove item_id from kwargs since it's passed positionally
        strategy_kwargs = {k: v for k, v in kwargs.items() if k != "item_id"}
        result = _process_single_item(strategy, chunked_data, item_id, **strategy_kwargs)
        
        # Save result
        _save_item_result(results_dir, item_id, result["response"])
        
        # Update metadata
        summary_metadata[item_id] = _extract_item_metadata(result, chunked_data, item_id, scope, kwargs.get("doc_range"))
        _update_total_usage(total_usage, result["usage"])
        config["total_items_processed"] = 1
        
    elif scope == "dataset":
        # Process all items in dataset
        item_ids = list(chunked_data["items"].keys())
        
        for item_id in item_ids:
            logger.info(f"Processing item {item_id} ({len(summary_metadata)+1}/{len(item_ids)})")
            
            try:
                # Create kwargs for this item (excluding item_id since it's passed positionally)
                item_kwargs = kwargs.copy()
                
                result = _process_single_item(strategy, chunked_data, item_id, **item_kwargs)
                
                # Save result
                _save_item_result(results_dir, item_id, result["response"])
                
                # Update metadata
                summary_metadata[item_id] = _extract_item_metadata(result, chunked_data, item_id, scope, None)
                _update_total_usage(total_usage, result["usage"])
                
            except Exception as e:
                logger.error(f"Failed to process item {item_id}: {e}")
                summary_metadata[item_id] = {"error": str(e)}
        
        config["total_items_processed"] = len([m for m in summary_metadata.values() if "error" not in m])
        
    elif scope == "doc_range":
        # Document range within an item
        item_id = kwargs["item_id"]
        # Remove item_id from kwargs since it's passed positionally
        strategy_kwargs = {k: v for k, v in kwargs.items() if k != "item_id"}
        result = _process_single_item(strategy, chunked_data, item_id, **strategy_kwargs)
        
        # Save result
        doc_range_str = str(kwargs["doc_range"]).replace(":", "-")
        filename = f"{item_id}_docs_{doc_range_str}"
        _save_item_result(results_dir, filename, result["response"])
        
        # Update metadata
        summary_metadata[filename] = _extract_item_metadata(result, chunked_data, item_id, scope, kwargs["doc_range"])
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


def _process_single_item(strategy: str, chunked_data: Dict, item_id: str, **kwargs) -> Dict[str, Any]:
    """Process a single item with the specified strategy."""
    # Get chunks directly from chunked data
    if item_id not in chunked_data["items"]:
        raise ValueError(f"Item {item_id} not found in chunked data")
    
    # Extract actual text chunks from document objects
    document_objects = chunked_data["items"][item_id]["chunks"]
    item_chunks = []
    for doc_obj in document_objects:
        item_chunks.extend(doc_obj["chunks"])  # Get the actual text chunks
    
    # Extract chunks based on scope and doc_range
    scope = kwargs.get("scope", "item")  # Default to item scope for single processing
    doc_range = kwargs.get("doc_range", None)
    chunks = _extract_chunks_for_scope(item_chunks, scope, doc_range)
    
    # Apply strategy
    strategy_kwargs = {k: v for k, v in kwargs.items() if k not in ["item_id", "scope", "doc_range"]}
    
    if strategy == "no_op":
        return no_op(chunks, **strategy_kwargs)
    elif strategy == "concat_and_summarize":
        return concat_and_summarize(chunks, **strategy_kwargs)
    elif strategy == "iterative_summarize":
        return iterative_summarize(chunks, **strategy_kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def load_chunked_data(chunked_file_path: str) -> Dict[str, Any]:
    """
    Load pre-chunked data from outputs/chunks/ directory.
    
    Args:
        chunked_file_path: Path to chunked JSON file
        
    Returns:
        Dict with chunked data for all items
    """
    if not os.path.exists(chunked_file_path):
        raise FileNotFoundError(f"Chunked data file not found: {chunked_file_path}")
    
    with open(chunked_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded chunked data: {len(data.get('items', {}))} items from {chunked_file_path}")
    return data





def _extract_chunks_for_scope(chunks: List[str], scope: str, doc_range: Optional[str]) -> List[str]:
    """
    Extract chunks based on scope and document range.
    
    Args:
        chunks: List of text chunks for an item
        scope: "item", "dataset", or "doc_range"
        doc_range: Document range specification (e.g., "0:2", "1", None)
        
    Returns:
        List of text chunks for the specified scope
    """    
    if scope in ["item", "dataset"]:
        # All chunks in the item
        return chunks
        
    elif scope == "doc_range":
        if doc_range is None:
            raise ValueError("doc_range required for doc_range scope")
        
        # Parse document range
        if ":" in str(doc_range):
            # Range format: "0:2" means docs 0, 1, 2
            start, end = map(int, str(doc_range).split(":"))
            if start < 0 or end >= len(chunks):
                raise ValueError(f"Document range {doc_range} out of bounds (0 to {len(chunks)-1})")
            return chunks[start:end+1]
        else:
            # Single document: "1" means just doc 1
            doc_idx = int(doc_range)
            if doc_idx < 0 or doc_idx >= len(chunks):
                raise ValueError(f"Document index {doc_idx} out of bounds (0 to {len(chunks)-1})")
            return [chunks[doc_idx]]
    
    else:
        raise ValueError(f"Unknown scope: {scope}")


def _find_default_chunked_file(dataset: str) -> str:
    """Find the most recent chunked file for a dataset."""
    chunks_dir = "outputs/chunks"
    if not os.path.exists(chunks_dir):
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    
    # Look for files matching the pattern
    pattern = f"{dataset}_"
    matching_files = [f for f in os.listdir(chunks_dir) if f.startswith(pattern) and f.endswith('.json')]
    
    if not matching_files:
        raise FileNotFoundError(f"No chunked files found for dataset {dataset} in {chunks_dir}")
    
    # Return the first match (could be enhanced to find most recent)
    default_file = os.path.join(chunks_dir, matching_files[0])
    logger.info(f"Using default chunked file: {default_file}")
    return default_file


def _extract_item_metadata(result: Dict, chunked_data: Dict, item_id: str, scope: str, doc_range: Optional[str]) -> Dict[str, Any]:
    """Extract metadata for a single item result."""
    chunks_used = result.get("input_chunks", 1)
    
    # Determine input type
    if scope == "doc_range" and doc_range:
        if ":" in str(doc_range):
            start, end = map(int, str(doc_range).split(":"))
            input_docs = f"doc_range_{start}_{end}"
        else:
            input_docs = f"single_doc_{doc_range}"
    else:
        # Calculate total chunks from all documents in the item
        document_objects = chunked_data["items"][item_id]["chunks"]
        total_chunks = sum(len(doc_obj["chunks"]) for doc_obj in document_objects)
        input_docs = "all_docs" if total_chunks > 1 else "single_doc"
    
    # Calculate lengths
    response = result.get("response", "")
    
    return {
        "input_docs": input_docs,
        "total_chunks": chunks_used,
        "input_length_chars": len(result.get("input_text", "")),  # Would need to track this
        "input_length_words": result.get("input_word_count", 0),  # Would need to track this
        "summary_length_chars": len(response),
        "summary_length_words": len(response.split()),
        "processing_time": result.get("processing_time", 0.0),
        "model": result.get("model", "unknown"),
        "usage": result.get("usage", {})
    }


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