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


def summarize_chunks_independently(chunks: list[str],
                         final_only: bool = False,
                         prompt_name: str = "default-independent-chunks",
                         model: str = "gpt-4.1-mini",
                         ask_user_confirmation: bool = False,
                         **kwargs) -> dict[str, Any]:
    """
    Summarize chunks independently.

    Args:
        chunks: List of text chunks to summarize
        final_only: Whether to summarize only the final chunk
        prompt_name: Name of the prompt to use
        model: LLM model to use
        ask_user_confirmation: Whether to ask user confirmation before API call
        **kwargs: Additional parameters for LLM call

    Returns:
        Dict with summary and metadata
    """
    # load all prompts from prompts/ directory
    prompt_dir = Path(f"prompts/summarization/{prompt_name}")
    prompts = {}
    for prompt_file in prompt_dir.glob("*.txt"):
        key = prompt_file.stem  # filename without .txt extension
        prompts[key] = prompt_file.read_text()
    
    # For backward compatibility, create system_and_user_prompt dict
    system_and_user_prompt = {
        "system": prompts.get("system", ""),
        "user": prompts.get("user", "")
    }

    results = []
    for ii in range(len(chunks)):
        print(f"Summarizing chunk {ii+1}")
        full_text = chunks[ii]

        template_vars = {"text": full_text,
                        "domain": "detective story",
                        "optional_summary_length": ""}

        result = call_llm(full_text, model, system_and_user_prompt, template_vars=template_vars, ask_user_confirmation=ask_user_confirmation, **kwargs)
        result["method"] = "summarize_chunks_independently"
        result["input_chunks"] = len(chunks)
        result["final_only"] = False
        result["chunk_index"] = ii
        result["prompt_name"] = prompt_name
        result["prompts_used"] = prompts  # Save all prompt templates
        result["template_vars"] = template_vars  # Save template variables used
        result["summary_type"] = "chunk summary"
        # final_prompts_used (with variables replaced) comes from call_llm result
        results.append(result)
    return results

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
    # load all prompts from prompts/ directory
    prompt_dir = Path(f"prompts/summarization/{prompt_name}")
    prompts = {}
    for prompt_file in prompt_dir.glob("*.txt"):
        key = prompt_file.stem  # filename without .txt extension
        prompts[key] = prompt_file.read_text()
    
    # For backward compatibility, create system_and_user_prompt dict
    system_and_user_prompt = {
        "system": prompts.get("system", ""),
        "user": prompts.get("user", "")
    }

    if final_only:
        # Concatenate chunks with newlines
        full_text = "\n\n".join(chunks)

        logger.info(f"Summarizing {len(chunks)} chunks ({len(full_text.split())} words) with {model}")


        template_vars = {"text": full_text,
                        "domain": "detective story",
                        "optional_summary_length": ""}

        result = call_llm(full_text, model, system_and_user_prompt, template_vars=template_vars, ask_user_confirmation=ask_user_confirmation, **kwargs)
        result["method"] = "concat_and_summarize"
        result["input_chunks"] = len(chunks)
        result["final_only"] = True
        result["prompt_name"] = prompt_name
        result["prompts_used"] = prompts  # Save all prompt templates
        result["template_vars"] = template_vars  # Save template variables used
        result["summary_type"] = "cumulative summary"
        # final_prompts_used (with variables replaced) comes from call_llm result
        return result
    else:
        results = []
        for ii in range(len(chunks)):
            print(f"Summarizing chunks from 1 to {ii+1}")
            full_text = "\n\n".join(chunks[:ii+1])

            template_vars = {"text": full_text,
                            "domain": "detective story",
                            "optional_summary_length": ""}

            result = call_llm(full_text, model, system_and_user_prompt, template_vars=template_vars, ask_user_confirmation=ask_user_confirmation, **kwargs)
            result["method"] = "concat_and_summarize"
            result["input_chunks"] = len(chunks)
            result["final_only"] = False
            result["chunk_index"] = ii
            result["prompt_name"] = prompt_name
            result["summary_content"] = "cumulative summary"
            result["prompts_used"] = prompts  # Save all prompt templates
            result["template_vars"] = template_vars  # Save template variables used
            result["summary_type"] = "cumulative summary"
            # final_prompts_used (with variables replaced) comes from call_llm result
            results.append(result)
        return results


def iterative_summarize(chunks: list[str], 
                        final_only: bool = False,
                        prompt_name: str = "incremental",
                        model: str = "gpt-4.1-mini",
                        ask_user_confirmation: bool = False,
                        **kwargs) -> list[dict[str, Any]]:
    """
    Iterative summarization strategy - builds summaries incrementally using previous context.
    
    First chunk gets summarized using "first-chunk-summary.txt" prompt.
    Each subsequent chunk uses the previous summary as context with 
    "summarize-chunk-with-previous-summary-context.txt" prompt.
    
    Args:
        chunks: List of text chunks to summarize iteratively
        final_only: If True, return only the final summary. If False, return all n summaries
        prompt_name: Name of prompt directory to use (default: "incremental")
        model: LLM model name
        ask_user_confirmation: Whether to ask for confirmation before API calls
        **kwargs: Additional parameters passed to call_llm
        
    Returns:
        List of result dictionaries, one for each incremental step
    """
    if not chunks:
        return []
    
    # Load prompt templates
    prompts_dir = Path(f"prompts/summarization/{prompt_name}")
    if not prompts_dir.exists():
        raise ValueError(f"Prompt directory not found: {prompts_dir}")
    
    prompts = {}
    for prompt_file in prompts_dir.glob("*.txt"):
        prompt_key = prompt_file.stem
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompts[prompt_key] = f.read().strip()
    
    if "system" not in prompts:
        raise ValueError(f"Missing 'system.txt' in {prompts_dir}")
    if "first-chunk-summary" not in prompts:
        raise ValueError(f"Missing 'first-chunk-summary.txt' in {prompts_dir}")
    if "summarize-chunk-with-previous-summary-context" not in prompts:
        raise ValueError(f"Missing 'summarize-chunk-with-previous-summary-context.txt' in {prompts_dir}")
    
    results = []
    previous_summary = None
    
    for i, chunk in enumerate(chunks):
        chunk_num = i + 1  # 1-indexed for human readability
        
        if i == 0:
            # First chunk - use first-chunk-summary prompt
            user_prompt = prompts["first-chunk-summary"]
            template_vars = {
                "domain": "detective story",  # TODO: Could be parameterized
                "chunk_num": str(chunk_num),
                "total_chunks": str(len(chunks)),
                "text": chunk,
                "optional_summary_length": ""
            }
        else:
            # Subsequent chunks - use incremental prompt with previous context
            user_prompt = prompts["summarize-chunk-with-previous-summary-context"]
            template_vars = {
                "domain": "detective story",  # TODO: Could be parameterized
                "chunk_num": str(chunk_num),
                "total_chunks": str(len(chunks)),
                "previous_summary": previous_summary,
                "text": chunk,
                "optional_summary_length": ""
            }
        
        system_and_user_prompt = {
            "system": prompts["system"],
            "user": user_prompt
        }
        
        print(f"Incremental summarization step {chunk_num}/{len(chunks)}")
        
        result = call_llm(chunk, model, system_and_user_prompt, template_vars=template_vars, 
                         ask_user_confirmation=ask_user_confirmation, **kwargs)
        
        # Add iterative-specific metadata
        result["method"] = "iterative_summarize"
        result["step"] = chunk_num
        result["total_steps"] = len(chunks)
        result["is_first_chunk"] = (i == 0)
        result["prompt_name"] = prompt_name
        result["prompts_used"] = prompts  # Save all prompt templates
        result["template_vars"] = template_vars  # Save template variables used
        result["summary_type"] = "incremental summary"
        
        # Update previous_summary for next iteration
        previous_summary = result["response"]
        
        results.append(result)
    
    # Return based on final_only flag
    if final_only:
        return [results[-1]]  # Return only the final summary as a list for consistency
    else:
        return results  # Return all incremental summaries





def save_summaries(
    item_id: str,
    summaries: List[str], 
    original_item_data: Dict[str, Any],
    output_dir: str,
    collection_metadata: Dict[str, Any] = None,
    item_metadata: Dict[str, Any] = None
) -> None:
    """
    Save summaries using the same structure as ChunkedDataset but with 'summaries' instead of 'chunks'.
    
    Args:
        item_id: ID of the item being summarized
        summaries: List of summary strings
        original_item_data: Original item data from ChunkedDataset (to preserve metadata)
        output_dir: Directory to save summaries (e.g., 'outputs/summaries/experiment_name')
        collection_metadata: Collection-level metadata (strategy, model, prompts templates, etc.)
        item_metadata: Item-specific metadata (final prompts with actual text, template vars, etc.)
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
        
        # Add item-specific metadata (final prompts with actual text, etc.)
        if item_metadata:
            doc_data["metadata"]["item_experiment_metadata"] = item_metadata
        
        summary_item["documents"].append(doc_data)
    
    else:
        # Multi-document case - create single document with all summaries
        multi_doc_metadata = {
            "num_summaries": len(summaries),
            "original_num_documents": len(original_docs),
            "original_total_chunks": sum(len(doc.get("chunks", [])) for doc in original_docs),
            "summarization_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add item-specific metadata (final prompts with actual text, etc.)
        if item_metadata:
            multi_doc_metadata["item_experiment_metadata"] = item_metadata
            
        summary_item["documents"].append({
            "summaries": summaries,
            "metadata": multi_doc_metadata
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
        # Update collection_metadata if provided (e.g., for command_run updates)
        if collection_metadata:
            collection_data["summarization_info"]["collection_metadata"].update(collection_metadata)
    else:
        # Create new collection metadata
        collection_data = {
            "summarization_info": {
                "collection_metadata": collection_metadata or {},
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
