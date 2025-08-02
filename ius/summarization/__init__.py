"""
IUS Summarization Module

This module provides LLM-based summarization capabilities with comprehensive
cost tracking and multiple summarization strategies.

Main Functions:
- summarize: High-level orchestration function with experimental tracking
- call_llm: Direct LLM calls with cost tracking
- concat_and_summarize: Concatenate chunks and summarize
- no_op: Simple concatenation without LLM processing
- load_chunked_data: Load pre-chunked data files

Usage Example:
    from ius.summarization import summarize

    # High-level orchestration with experimental tracking
    result = summarize(
        strategy="concat_and_summarize",
        dataset="bmds", 
        scope="item",
        item_id="ADP02",
        model="gpt-4.1-mini"
    )
    
    # Direct function usage
    from ius.summarization import concat_and_summarize
    
    result = concat_and_summarize(
        chunks=["text1", "text2"],
        model="gpt-4.1-mini",
        system_and_user_prompt={
            "system": "You are a helpful assistant.",
            "user": "Please summarize: {text}"
        }
    )
"""

from .methods import concat_and_summarize, iterative_summarize, no_op
from .orchestration import load_chunked_data, summarize
from ..utils import call_llm


__all__ = [
    "summarize",
    "call_llm",
    "no_op",
    "concat_and_summarize",
    "iterative_summarize",
    "load_chunked_data"
]
