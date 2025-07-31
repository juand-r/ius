"""
Chunking module for splitting documents into smaller pieces.

This module provides simple, delimiter-aware chunking strategies that preserve
text boundaries and ensure no content is lost.
"""

from .chunkers import chunk_custom, chunk_fixed_count, chunk_fixed_size
from .utils import analyze_chunks, preview_chunks, validate_chunks


__all__ = [
    "chunk_fixed_size",
    "chunk_fixed_count",
    "chunk_custom",
    "validate_chunks",
    "analyze_chunks",
    "preview_chunks",
]
