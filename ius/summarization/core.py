"""
Core summarization functionality for the IUS system.

Simplified, clean interface for summarizing chunks from ChunkedDataset objects.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..data import ChunkedDataset
from ..logging_config import get_logger
from ..utils import call_llm

logger = get_logger(__name__)


def summarize_chunks(
    chunked_dataset: ChunkedDataset,
    item_id: str,
    output_summary_type: str = "final-only",  # "final-only" or "intermediate"
    prompt_dir: Union[str, Path] = "prompts",
    document_spec: Union[str, int, None] = None,  # None=all docs, int=single doc, "0:2"=range
    model: str = "gpt-4.1-mini",
    **kwargs
) -> Dict[str, Any]:
    """
    Summarize chunks from a ChunkedDataset item.
    
    Args:
        chunked_dataset: ChunkedDataset object containing chunked data
        item_id: ID of the item to summarize
        output_summary_type: "final-only" or "intermediate"
        prompt_dir: Directory containing prompt files
        document_spec: Document specification:
            - None: all documents in item
            - int: single document by index
            - "start:end": range of documents (e.g., "0:2")
            - "N": single document N as string
        model: LLM model to use
        **kwargs: Additional parameters passed to LLM
        
    Returns:
        Dict with summary results:
        {
            "intermediate_summaries": None | List[str],
            "final_summary": str,
            "metadata": {
                "document_span": str,
                "num_chunks": int,
                "num_documents": int,
                "model": str,
                "processing_time": float,
                "token_usage": {...}
            }
        }
    """
    start_time = time.time()
    
    # Load prompts
    prompts = _load_prompts(prompt_dir)
    
    # Get chunks based on document specification
    chunks, doc_span = _get_chunks_for_documents(chunked_dataset, item_id, document_spec)
    
    if not chunks:
        raise ValueError(f"No chunks found for item '{item_id}' with document_spec '{document_spec}'")
    
    # Perform summarization based on output type
    if output_summary_type == "final-only":
        result = _summarize_final_only(chunks, prompts, model, **kwargs)
    elif output_summary_type == "intermediate":
        result = _summarize_with_intermediates(chunks, prompts, model, **kwargs)
    else:
        raise ValueError(f"Invalid output_summary_type: {output_summary_type}. Must be 'final-only' or 'intermediate'")
    
    # Add metadata
    processing_time = time.time() - start_time
    result["metadata"] = {
        "document_span": doc_span,
        "num_chunks": len(chunks),
        "num_documents": len(_parse_doc_span(doc_span)),
        "model": model,
        "processing_time": processing_time,
        "token_usage": result.get("token_usage", {}),
        "output_summary_type": output_summary_type
    }
    
    return result


def _load_prompts(prompt_dir: Union[str, Path]) -> Dict[str, str]:
    """Load prompts from directory."""
    prompt_path = Path(prompt_dir)
    prompts = {}
    
    # Look for common prompt files
    prompt_files = {
        "system": "system.txt",
        "user": "user.txt", 
        "intermediate": "intermediate.txt",
        "final": "final.txt"
    }
    
    for key, filename in prompt_files.items():
        file_path = prompt_path / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts[key] = f.read().strip()
        else:
            logger.debug(f"Prompt file not found: {file_path}")
    
    # Default prompts if none found
    if not prompts:
        prompts = {
            "system": "You are a helpful assistant that summarizes text.",
            "user": "Please summarize the following text:\n\n{text}"
        }
    
    return prompts


def _get_chunks_for_documents(
    chunked_dataset: ChunkedDataset, 
    item_id: str, 
    document_spec: Union[str, int, None]
) -> tuple[List[str], str]:
    """
    Get chunks for specified documents and return document span description.
    
    Returns:
        (chunks, document_span_description)
    """
    if document_spec is None:
        # All documents
        chunks = chunked_dataset.get_item_chunks(item_id)
        item_data = chunked_dataset.load_item(item_id)
        num_docs = len(item_data["documents"])
        doc_span = f"all_{num_docs}_documents"
        
    elif isinstance(document_spec, int):
        # Single document by index
        chunks = chunked_dataset.get_document_chunks(item_id, document_spec)
        doc_span = f"document_{document_spec}"
        
    elif isinstance(document_spec, str):
        if ":" in document_spec:
            # Range of documents
            chunks = chunked_dataset.get_chunks_for_scope(f"doc_range:{document_spec}", item_id=item_id)
            doc_span = f"documents_{document_spec}"
        else:
            # Single document as string
            doc_idx = int(document_spec)
            chunks = chunked_dataset.get_document_chunks(item_id, doc_idx)
            doc_span = f"document_{doc_idx}"
    else:
        raise ValueError(f"Invalid document_spec type: {type(document_spec)}")
    
    return chunks, doc_span


def _parse_doc_span(doc_span: str) -> List[int]:
    """Parse document span to get list of document indices."""
    if doc_span.startswith("all_"):
        num_docs = int(doc_span.split("_")[1])
        return list(range(num_docs))
    elif doc_span.startswith("document_"):
        doc_idx = int(doc_span.split("_")[1])
        return [doc_idx]
    elif doc_span.startswith("documents_"):
        range_part = doc_span.split("_")[1]
        if ":" in range_part:
            start, end = map(int, range_part.split(":"))
            return list(range(start, end))
        else:
            return [int(range_part)]
    else:
        logger.warning(f"Could not parse doc_span: {doc_span}")
        return []


def _summarize_final_only(
    chunks: List[str], 
    prompts: Dict[str, str], 
    model: str, 
    **kwargs
) -> Dict[str, Any]:
    """Summarize chunks into final summary only."""
    
    # Concatenate all chunks
    combined_text = "\n\n".join(chunks)
    
    # Prepare prompt
    system_prompt = prompts.get("system", "You are a helpful assistant that summarizes text.")
    user_prompt = prompts.get("user", "Please summarize the following text:\n\n{text}")
    user_prompt = user_prompt.format(text=combined_text)
    
    # Call LLM
    llm_result = call_llm(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        **kwargs
    )
    
    return {
        "intermediate_summaries": None,
        "final_summary": llm_result["response"],
        "token_usage": llm_result.get("usage", {})
    }


def _summarize_with_intermediates(
    chunks: List[str], 
    prompts: Dict[str, str], 
    model: str, 
    **kwargs
) -> Dict[str, Any]:
    """Summarize chunks with intermediate summaries for each chunk."""
    
    # Get prompts
    system_prompt = prompts.get("system", "You are a helpful assistant that summarizes text.")
    intermediate_prompt = prompts.get("intermediate", prompts.get("user", "Please summarize the following text:\n\n{text}"))
    final_prompt = prompts.get("final", "Please create a final summary from these intermediate summaries:\n\n{summaries}")
    
    # Create intermediate summaries
    intermediate_summaries = []
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost": 0.0}
    
    for i, chunk in enumerate(chunks):
        logger.debug(f"Creating intermediate summary {i+1}/{len(chunks)}")
        
        user_prompt = intermediate_prompt.format(text=chunk)
        llm_result = call_llm(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs
        )
        
        intermediate_summaries.append(llm_result["response"])
        
        # Accumulate usage stats
        usage = llm_result.get("usage", {})
        for key in total_usage:
            total_usage[key] += usage.get(key, 0)
    
    # Create final summary from intermediates
    logger.debug("Creating final summary from intermediate summaries")
    combined_summaries = "\n\n".join(f"Summary {i+1}: {summary}" for i, summary in enumerate(intermediate_summaries))
    final_user_prompt = final_prompt.format(summaries=combined_summaries)
    
    final_result = call_llm(
        model=model,
        system_prompt=system_prompt,
        user_prompt=final_user_prompt,
        **kwargs
    )
    
    # Accumulate final usage
    final_usage = final_result.get("usage", {})
    for key in total_usage:
        total_usage[key] += final_usage.get(key, 0)
    
    return {
        "intermediate_summaries": intermediate_summaries,
        "final_summary": final_result["response"], 
        "token_usage": total_usage
    }