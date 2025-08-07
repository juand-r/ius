#!/usr/bin/env python3
"""
Quick chunking examples for IPython/Jupyter.

Copy and paste these code blocks into IPython for quick testing.
"""

# =============================================================================
# Method 1: Quick CLI chunking (fastest for testing)
# =============================================================================

"""
# In IPython:
!python -m ius chunk --dataset bmds --strategy fixed_count --count 3 --output outputs/chunks/quick_test

# Then load and inspect:
from ius.data import ChunkedDataset
chunked = ChunkedDataset("outputs/chunks/quick_test")
print(f"Loaded: {chunked}")
print(f"Items: {chunked.item_ids[:3]}")

# Get chunks for an item:
chunks = chunked.get_item_chunks("ADP02")
print(f"Item has {len(chunks)} chunks")
print(f"First chunk: {chunks[0][:200]}...")
"""

# =============================================================================
# Method 2: Programmatic chunking (more control)
# =============================================================================

"""
# In IPython:
from ius.cli.chunk import chunk_dataset

# Create chunked data programmatically
result = chunk_dataset(
    dataset_name="bmds",
    strategy="fixed_count", 
    num_chunks=2,
    output_path="outputs/chunks/ipython_test"
)

print(f"Chunked {result['overall_stats']['total_items']} items")
print(f"Total chunks: {result['overall_stats']['total_chunks']}")

# Load the result
from ius.data import ChunkedDataset
chunked = ChunkedDataset("outputs/chunks/ipython_test")
print(f"Loaded: {chunked}")

# Quick inspection
item_id = chunked.item_ids[0]  # First item
chunks = chunked.get_item_chunks(item_id)
print(f"Item '{item_id}' has {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1}: {len(chunk)} chars - {chunk[:100]}...")
"""

# =============================================================================
# Method 3: One-liner for quick testing
# =============================================================================

"""
# In IPython - complete workflow in a few lines:
from ius.cli.chunk import chunk_dataset
from ius.data import ChunkedDataset

# Chunk and load in one go
chunk_dataset("bmds", "fixed_count", num_chunks=2, output_path="outputs/chunks/test")
chunked = ChunkedDataset("outputs/chunks/test")
chunks = chunked.get_item_chunks("ADP02")
print(f"Created {len(chunks)} chunks from ADP02")
"""

# =============================================================================
# Method 4: Inspect existing chunked data
# =============================================================================

"""
# If you already have chunked data:
from ius.data import ChunkedDataset
import os

# List available chunked datasets
if os.path.exists("outputs/chunks"):
    print("Available chunked datasets:")
    for name in os.listdir("outputs/chunks"):
        if os.path.isdir(f"outputs/chunks/{name}"):
            try:
                ds = ChunkedDataset(f"outputs/chunks/{name}")
                print(f"  - {name}: {len(ds)} items")
            except:
                print(f"  - {name}: (invalid)")

# Load and inspect
chunked = ChunkedDataset("outputs/chunks/YOUR_DATASET_NAME")
print(f"Dataset: {chunked}")
print(f"First few items: {chunked.item_ids[:5]}")

# Get chunks for specific item
item = "ADP02"  # or chunked.item_ids[0] for first available
chunks = chunked.get_item_chunks(item)
print(f"\\nItem '{item}' chunks:")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1}: {len(chunk):,} chars")
    print(f"    Preview: {chunk[:150]}...")
    print()
"""

if __name__ == "__main__":
    print("Copy the code blocks above into IPython for quick chunking examples!")