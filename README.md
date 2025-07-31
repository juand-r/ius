# Incremental Update Summarization (IUS)

A research framework for studying strategies to incrementally summarize long documents or sequences of documents.

## Overview

Documents often arrive in chunks over time. This project explores how to effectively maintain and update summaries as new content becomes available, rather than re-summarizing from scratch each time.

### Key Concepts

- **Item**: A collection unit containing one or more related documents (e.g., a detective story, a news topic over time, a book series)
- **Document**: An individual text unit within an item (e.g., a single detective story, one news article, one book). Sometimes there is only one document in an Item. NOTE: may need to expand "Document" to include "multi-document" (e.g., various news stories on single day)
- **Chunk**: Smaller consecutive pieces of a document, created through various chunking strategies
- **Incremental Summarization**: Updating summaries efficiently as new chunks or documents arrive

### Document vs. Chunk Relationship

**Current datasets** (BMDS, True Detective): Each item contains one document. We chunk these documents and study incremental summarization within single long texts.

**Future datasets**: Items will contain multiple sequential documents (e.g., news articles over time, TV episodes, book series chapters) enabling study of incremental summarization across document boundaries.

### Chunking Strategies

The framework supports multiple chunking approaches:

1. **Fixed-size chunking**: Split documents into chunks of roughly equal length
2. **Fixed-count chunking**: Split documents into a fixed number of chunks  
3. **Natural chunking**: Use document structure (paragraphs, sections) when available

When chunks ≠ documents, the system tracks information to enable:
- **Document-level summarization**: Summarize individual documents independently  
- **Sequence-level summarization**: Summarize across multiple documents using all contained chunks

## Repository Structure

```
ius/                    # Main project code
├── data/              # Dataset loading and manipulation
│   ├── loader.py      # Standard dataset loader 
│   └── __init__.py    
├── chunk/             # Chunking strategies (TODO)
├── eval/              # Evaluation and experiment tracking (TODO)  
├── summarization/     # Core summarization strategies (TODO)
└── ...

datasets/              # Standardized datasets
├── bmds/             # Birth of Modern Detection Stories (34 items)
├── true-detective/   # True Detective puzzles (191 items) 
├── fables/           # Fables collection
└── booookscore/      # Book collection (TODO: populate)

data-source/          # Raw data for ingestion
└── ...

trash/                # Deprecated/experimental code
```

## Dataset Format

Each dataset follows a standardized JSON structure:

### Collection Metadata (`collection.json`)
```json
{
  "domain": "detective_stories",
  "source": "Source description", 
  "num_items": 34,
  "total_documents": 34,
  "description": "Dataset description",
  "items": ["item1", "item2", ...]
}
```

### Item Structure (`items/{item_id}.json`)
```json
{
  "item_metadata": {
    "item_id": "unique_id",
    "num_documents": 1
  },
  "documents": [
    {
      "content": "Full document text...",
      "metadata": {
        "title": "Document title",
        "author": "Author name",
        ...
      }
    }
  ]
}
```

## Current Datasets

### BMDS (Birth of Modern Detection Stories)
- **Items**: 34 classic detective stories
- **Documents per item**: 1 (each story is one document)
- **Use case**: Study incremental summarization for detective stories, evaluate on the downstream task of guessing the culprit.

### True Detective  
- **Items**: 191 short mystery puzzles
- **Documents per item**: 1 (each puzzle is one document)
- **Use case**: Study incremental summarization on detective stories, evaluate on the downstream task of guessing the culprit.

## Development Principles. Cursor, you need to follow this!

- **Lean and modular**: Keep code clean, simple, readable, well-documented, modular, and easily extensible. The goals are ease of use and reproducibility.
- **Start small**: Get everything working with BMDS before expanding to other datasets
- **Comprehensive evaluation**: Track experiments systematically with detailed metrics
- **LLM flexibility**: Abstract LLM calls to easily switch between APIs and local models

## Getting Started

1. **Load a dataset**:
```python
from ius.data import load_data

# Load full dataset
data = load_data("bmds")
print(f"Loaded {data['num_items_loaded']} items")

# Load single item  
item_data = load_data("bmds", item_id="ADP02")
```

2. **Explore dataset structure**:
```python
from ius.data import list_datasets, get_dataset_info

datasets = list_datasets()
info = get_dataset_info("bmds")
print(info)
```

## Future Directions

- **Multi-document datasets**: News sequences, TV show episodes, book series
- **Advanced chunking**: Semantic and structure-aware chunking strategies  
- **Incremental strategies**: Various approaches to update summaries efficiently
- **Comprehensive evaluation**: Content preservation, summary quality, computational efficiency
- **Interactive tools**: Visualization and analysis of incremental summarization results

## Research Questions

- How do different chunking strategies affect incremental summarization quality?
- What are the trade-offs between summary freshness and computational cost?
- How does incremental performance vary across document types and domains?
- Can we predict when full re-summarization is needed vs. incremental updates?

---

*This is a research framework. Code should be clean, well-documented, and easy to extend for various experimental approaches.*