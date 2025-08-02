"""
High-level summarization strategies for the IUS system.

This module provides different approaches to combining and summarizing chunks:
- no_op: Do not summarize, just concatenate chunks
- concat_and_summarize: Concatenate chunks and send to LLM
- iterative_summarize: Placeholder for future iterative approaches
"""

from typing import Any

from ..logging_config import get_logger
from ..utils import call_llm
from pathlib import Path

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
