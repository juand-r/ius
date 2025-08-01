#!/usr/bin/env python3
"""
Tests for the chunking module using BMDS data.
"""

import unittest

from ius.chunk import (
    analyze_chunks,
    chunk_fixed_count,
    chunk_fixed_size,
    preview_chunks,
    validate_chunks,
)
from ius.data import load_data
from ius.exceptions import ChunkingError


class TestChunking(unittest.TestCase):
    """Test cases for chunking functions."""

    @classmethod
    def setUpClass(cls):
        """Load BMDS data for testing."""
        data = load_data("bmds", item_id="ADP02")
        cls.item = data["items"]["ADP02"]
        cls.document = cls.item["documents"][0]
        cls.text = cls.document["content"]

    def test_fixed_size_chunking(self):
        """Test fixed_size chunking with default newline delimiter."""
        chunks = chunk_fixed_size(self.text, chunk_size=1000, delimiter="\n")

        # Basic checks
        assert len(chunks) > 0, "Should create at least one chunk"
        assert all(isinstance(chunk, str) for chunk in chunks), (
            "All chunks should be strings"
        )

        # Content preservation
        assert validate_chunks(self.text, chunks, delimiter="\n"), (
            "Content should be preserved"
        )

        # Size constraints (allowing for delimiter boundary flexibility)
        for chunk in chunks[:-1]:  # All but last chunk
            assert len(chunk) <= 1200, (
                f"Chunk too large: {len(chunk)} chars"
            )  # Allow some flexibility

        print(f"✅ Fixed size chunking: {len(chunks)} chunks created")

    def test_fixed_count_chunking(self):
        """Test fixed_count chunking with specified number of chunks."""
        target_chunks = 5
        chunks = chunk_fixed_count(self.text, num_chunks=target_chunks, delimiter="\n")

        # Basic checks
        assert len(chunks) > 0, "Should create at least one chunk"
        assert len(chunks) <= target_chunks, f"Should not exceed {target_chunks} chunks"

        # Content preservation
        assert validate_chunks(self.text, chunks, delimiter="\n"), (
            "Content should be preserved"
        )

        print(
            f"✅ Fixed count chunking: {len(chunks)} chunks created (target: {target_chunks})"
        )

    def test_content_preservation(self):
        """Test that chunking preserves all content."""
        test_cases = [
            (1000, "\n"),
            (500, "\n"),
            # Only test delimiters that actually exist in the text
            (2000, "\n"),  # Use newline instead of paragraph breaks
        ]

        for chunk_size, delimiter in test_cases:
            chunks = chunk_fixed_size(
                self.text, chunk_size=chunk_size, delimiter=delimiter
            )
            assert validate_chunks(self.text, chunks, delimiter=delimiter), (
                f"Content preservation failed for chunk_size={chunk_size}, delimiter={repr(delimiter)}"
            )

    def test_analyze_chunks(self):
        """Test chunk analysis functionality."""
        chunks = chunk_fixed_size(self.text, chunk_size=1000, delimiter="\n")
        analysis = analyze_chunks(chunks, delimiter="\n")

        # Check required fields
        required_fields = [
            "num_chunks",
            "total_chars",
            "avg_chunk_size",
            "min_chunk_size",
            "max_chunk_size",
            "size_std",
            "delimiter",
        ]
        for field in required_fields:
            assert field in analysis, f"Analysis missing field: {field}"

        # Sanity checks
        assert analysis["num_chunks"] == len(chunks)
        assert analysis["total_chars"] > 0
        assert analysis["avg_chunk_size"] > 0

        print(f"✅ Analysis: {analysis}")

    def test_preview_chunks(self):
        """Test chunk preview functionality."""
        chunks = chunk_fixed_size(self.text, chunk_size=1000, delimiter="\n")
        previews = preview_chunks(chunks[:3], max_preview=100)

        assert len(previews) == 3, "Should create previews for first 3 chunks"
        for i, preview in enumerate(previews):
            assert preview.startswith(f"Chunk {i + 1}:"), (
                "Preview should start with chunk number"
            )
            assert len(preview) <= 150, (
                "Preview should be reasonably short"
            )  # Some overhead for formatting

    def test_empty_text(self):
        """Test chunking behavior with empty text - should raise error."""
        with self.assertRaises(ChunkingError):
            chunk_fixed_size("", chunk_size=1000, delimiter="\n")

        with self.assertRaises(ChunkingError):
            chunk_fixed_count("", num_chunks=5, delimiter="\n")

    def test_no_delimiters(self):
        """Test chunking behavior when delimiter is not found in text - should raise error."""
        text_no_newlines = (
            "This is a text without any newlines or specified delimiters."
        )

        with self.assertRaises(ChunkingError):
            chunk_fixed_size(text_no_newlines, chunk_size=10, delimiter="\n")

        with self.assertRaises(ChunkingError):
            chunk_fixed_count(text_no_newlines, num_chunks=3, delimiter="\n")


class TestChunkingValidation(unittest.TestCase):
    """Test cases for input validation and error handling."""

    def test_invalid_input_types(self):
        """Test that invalid input types raise ChunkingError."""
        # Test non-string text
        with self.assertRaises(ChunkingError):
            chunk_fixed_size(123, chunk_size=100)

        with self.assertRaises(ChunkingError):
            chunk_fixed_count(None, num_chunks=5)

        # Test non-integer parameters
        with self.assertRaises(ChunkingError):
            chunk_fixed_size("test text", chunk_size="100")

        with self.assertRaises(ChunkingError):
            chunk_fixed_count("test text", num_chunks=5.5)

        # Test non-string delimiter
        with self.assertRaises(ChunkingError):
            chunk_fixed_size("test text", chunk_size=100, delimiter=123)

    def test_invalid_parameter_values(self):
        """Test that invalid parameter values raise ChunkingError."""
        # Test negative/zero chunk sizes
        with self.assertRaises(ChunkingError):
            chunk_fixed_size("test text", chunk_size=0)

        with self.assertRaises(ChunkingError):
            chunk_fixed_size("test text", chunk_size=-100)

        # Test negative/zero chunk counts
        with self.assertRaises(ChunkingError):
            chunk_fixed_count("test text", num_chunks=0)

        with self.assertRaises(ChunkingError):
            chunk_fixed_count("test text", num_chunks=-5)

        # Test empty delimiter
        with self.assertRaises(ChunkingError):
            chunk_fixed_size("test text", chunk_size=100, delimiter="")

    def test_missing_delimiter_error(self):
        """Test that missing delimiters raise ChunkingError with helpful message."""
        text = "This has no newlines just spaces"

        with self.assertRaises(ChunkingError) as cm:
            chunk_fixed_size(text, chunk_size=10, delimiter="\n")

        self.assertIn("Delimiter", str(cm.exception))
        self.assertIn("not found", str(cm.exception))

    def test_empty_text_error(self):
        """Test that empty text raises ChunkingError with helpful message."""
        with self.assertRaises(ChunkingError) as cm:
            chunk_fixed_size("", chunk_size=100)

        self.assertIn("Cannot chunk empty text", str(cm.exception))


def test_chunking_integration():
    """Integration test that prints detailed results like the original test."""
    print("\n🧪 Testing Chunking Module with BMDS Data")

    # Load data
    print("📚 Loading BMDS dataset...")
    data = load_data("bmds", item_id="ADP02")
    item = data["items"]["ADP02"]
    document = item["documents"][0]
    text = document["content"]

    print(f"✅ Loaded item: {item['item_metadata']['item_id']}")
    print(f"📄 Document length: {len(text):,} characters")
    print(f"📝 Lines in document: {text.count(chr(10)) + 1}")

    # Test fixed size chunking
    print("\n🔪 Testing fixed_size chunking (1000 chars, newline delimiter)...")
    chunks_fixed_size = chunk_fixed_size(text, chunk_size=1000, delimiter="\n")

    print(f"📊 Created {len(chunks_fixed_size)} chunks")
    analysis = analyze_chunks(chunks_fixed_size, delimiter="\n")
    print(f"📈 Analysis: {analysis}")

    # Validate content preservation
    is_valid = validate_chunks(text, chunks_fixed_size, delimiter="\n")
    print(f"✅ Content preservation: {'PASS' if is_valid else 'FAIL'}")

    # Test fixed count chunking
    print("\n🔪 Testing fixed_count chunking (5 chunks, newline delimiter)...")
    chunks_fixed_count = chunk_fixed_count(text, num_chunks=5, delimiter="\n")

    print(f"📊 Created {len(chunks_fixed_count)} chunks")
    analysis = analyze_chunks(chunks_fixed_count, delimiter="\n")
    print(f"📈 Analysis: {analysis}")

    # Validate content preservation
    is_valid = validate_chunks(text, chunks_fixed_count, delimiter="\n")
    print(f"✅ Content preservation: {'PASS' if is_valid else 'FAIL'}")

    # Show chunk previews
    print("\n👀 First 3 chunk previews (fixed_size):")
    previews = preview_chunks(chunks_fixed_size[:3], max_preview=150)
    for preview in previews:
        print(f"  {preview}")

    print("\n🎉 Chunking tests completed!")


if __name__ == "__main__":
    # Run the integration test when called directly
    test_chunking_integration()

    # Also run unittest cases
    print("\n" + "=" * 50)
    print("Running Unit Tests")
    print("=" * 50)
    unittest.main(verbosity=2, exit=False)
