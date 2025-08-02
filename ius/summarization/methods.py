"""
High-level summarization strategies for the IUS system.

This module provides different approaches to combining and summarizing chunks:
- no_op: Do not summarize, just concatenate chunks
- concat_and_summarize: Concatenate chunks and send to LLM
- iterative_summarize: Placeholder for future iterative approaches
"""

import json
import time
from typing import Any, Dict, List
from pathlib import Path

from ..logging_config import get_logger
from ..utils import call_llm

logger = get_logger(__name__)


def no_op(chunks: list[str], **kwargs) -> dict[str, Any]:
    """
    No-operation strategy: simply concatenate chunks without LLM processing.

    Args:
        chunks: List of text chunks to concatenate
        **kwargs: Ignored for no-op

    Returns:
        Dict with concatenated text and metadata
    """
    # Join chunks with newlines to preserve boundaries
    response = "\n\n".join(chunks)

    return {
        "response": response,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0
        },
        "model": "no-op",
        "processing_time": 0.0,
        "method": "no_op"
    }


def concat_and_summarize(chunks: list[str],
                         final_only: bool = False,
                         prompt_name: str = "default-concat-prompt",
                         model: str = "gpt-4.1-mini",
                         ask_user_confirmation: bool = False,
                         **kwargs) -> dict[str, Any]:
    """
    Concatenate chunks and send to LLM for summarization.

    Args:
        chunks: List of text chunks to summarize
        model: LLM model to use
        system_and_user_prompt: Dict with "system" and "user" prompt content
        ask_user_confirmation: Whether to ask user confirmation before API call
        **kwargs: Additional parameters for LLM call

    Returns:
        Dict with summary and metadata
    """
    # load system and user prompt from prompts/
    system_prompt = Path(f"prompts/summarization/{prompt_name}/system.txt").read_text()
    user_prompt = Path(f"prompts/summarization/{prompt_name}/user.txt").read_text()

    system_and_user_prompt = {
        "system": system_prompt,
        "user": user_prompt
    }

    if final_only:
        # Concatenate chunks with newlines
        full_text = "\n\n".join(chunks)

        logger.info(f"Summarizing {len(chunks)} chunks ({len(full_text.split())} words) with {model}")

        result = call_llm(full_text, model, system_and_user_prompt, template_vars={"text": full_text}, ask_user_confirmation=ask_user_confirmation, **kwargs)
        result["method"] = "concat_and_summarize"
        result["input_chunks"] = len(chunks)
        result["final_only"] = True
        return result
    else:
        results = []
        for ii in range(len(chunks)):
            print(f"Summarizing chunks from 1 to {ii+1}")
            full_text = "\n\n".join(chunks[:ii+1])
            result = call_llm(full_text, model, system_and_user_prompt, template_vars={"text": full_text}, ask_user_confirmation=ask_user_confirmation, **kwargs)
            result["method"] = "concat_and_summarize"
            result["input_chunks"] = len(chunks)
            result["final_only"] = False
            result["chunk_index"] = ii
            results.append(result)
        return results


def iterative_summarize(chunks: list[str], **kwargs) -> dict[str, Any]:
    """
    Iterative summarization strategy (placeholder for future implementation).

    Args:
        chunks: List of text chunks to summarize iteratively
        **kwargs: Strategy-specific parameters

    Returns:
        Dict with summary and metadata

    Raises:
        NotImplementedError: This strategy is not yet implemented
    """
    raise NotImplementedError("Iterative summarization strategy not yet implemented")


def save_summaries(
    item_id: str,
    summaries: List[str], 
    original_item_data: Dict[str, Any],
    output_dir: str,
    experiment_metadata: Dict[str, Any] = None
) -> None:
    """
    Save summaries using the same structure as ChunkedDataset but with 'summaries' instead of 'chunks'.
    
    Args:
        item_id: ID of the item being summarized
        summaries: List of summary strings
        original_item_data: Original item data from ChunkedDataset (to preserve metadata)
        output_dir: Directory to save summaries (e.g., 'outputs/summaries/experiment_name')
        experiment_metadata: Additional metadata about the summarization experiment
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create items subdirectory
    items_dir = output_path / "items"
    items_dir.mkdir(exist_ok=True)
    
    # Prepare summary item data (same structure as chunked data, but with 'summaries')
    summary_item = {
        "item_metadata": {
            **original_item_data.get("item_metadata", {}),
            "summarization_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_summaries": len(summaries)
        },
        "documents": []
    }
    
    # Convert chunks structure to summaries structure
    original_docs = original_item_data.get("documents", [])
    
    if len(original_docs) == 1:
        # Single document case - replace 'chunks' with 'summaries'
        doc_data = original_docs[0].copy()
        doc_data["summaries"] = summaries  # Replace 'chunks' with 'summaries'
        doc_data.pop("chunks", None)  # Remove 'chunks' key if it exists
        
        # Update metadata
        if "metadata" not in doc_data:
            doc_data["metadata"] = {}
        doc_data["metadata"].update({
            "num_summaries": len(summaries),
            "original_num_chunks": len(original_docs[0].get("chunks", [])),
            "summarization_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        summary_item["documents"].append(doc_data)
    
    else:
        # Multi-document case - create single document with all summaries
        summary_item["documents"].append({
            "summaries": summaries,
            "metadata": {
                "num_summaries": len(summaries),
                "original_num_documents": len(original_docs),
                "original_total_chunks": sum(len(doc.get("chunks", [])) for doc in original_docs),
                "summarization_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        })
    
    # Save individual item
    item_file = items_dir / f"{item_id}.json"
    with open(item_file, "w", encoding="utf-8") as f:
        json.dump(summary_item, f, indent=2, ensure_ascii=False)
    
    # Create/update collection.json
    collection_file = output_path / "collection.json"
    
    if collection_file.exists():
        # Load existing collection metadata
        with open(collection_file, "r", encoding="utf-8") as f:
            collection_data = json.load(f)
    else:
        # Create new collection metadata
        collection_data = {
            "summarization_info": {
                "experiment_metadata": experiment_metadata or {},
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "items_processed": []
            }
        }
    
    # Add this item to the processed list
    if item_id not in collection_data["summarization_info"]["items_processed"]:
        collection_data["summarization_info"]["items_processed"].append(item_id)
    
    # Save collection.json
    with open(collection_file, "w", encoding="utf-8") as f:
        json.dump(collection_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved summary for '{item_id}' to: {output_path}")
    logger.info(f"Saved {len(summaries)} summaries for item {item_id} to {output_path}")
