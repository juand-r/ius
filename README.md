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
├── __main__.py        # Main entry point for CLI commands
├── config.py          # Configuration management with environment variables
├── exceptions.py      # Custom exception hierarchy
├── logging_config.py  # Structured logging setup
├── cli/              # Command-line interfaces
│   ├── chunk.py      # Chunking CLI with --verbose, --dry-run flags
│   └── common.py     # Shared CLI utilities
├── chunk/            # Text chunking strategies ✅ IMPLEMENTED
│   ├── chunkers.py   # Fixed-size, fixed-count, custom chunking
│   └── utils.py      # Chunk analysis and validation
├── data/             # Dataset loading and manipulation ✅ IMPLEMENTED
│   ├── loader.py     # Standard dataset loader with error handling
│   └── __init__.py   # Data loading convenience functions
├── eval/             # Evaluation and experiment tracking (TODO)  
└── summarization/    # Core summarization strategies (TODO)

datasets/             # Standardized datasets
├── bmds/             # Birth of Modern Detection Stories (34 items)
├── true-detective/   # True Detective puzzles (191 items) 
├── fables/           # Fables collection
└── booookscore/      # Book collection (TODO: populate)

tests/                # Comprehensive test suite (102 tests)
├── test_chunking.py         # Chunking function tests
├── test_cli_chunk.py        # CLI functionality tests  
├── test_data_loader.py      # Data loading tests
├── test_cli_common.py       # CLI utilities tests
├── test_main.py            # Main entry point tests
└── test_logging_config.py  # Logging configuration tests

outputs/              # Generated output files
└── chunks/          # Chunking results in JSON format

data-source/          # Raw data for ingestion
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
      "doc_id": "unique_document_id",
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

## CLI Usage

The IUS framework provides a comprehensive command-line interface for text chunking with built-in progress tracking, logging, and validation.

### Basic Commands

```bash
# List available datasets
python -m ius chunk --list-datasets

# Basic chunking with fixed size
python -m ius chunk --dataset bmds --strategy fixed_size --size 2048

# Fixed count chunking  
python -m ius chunk --dataset true-detective --strategy fixed_count --count 8

# Custom delimiter chunking
python -m ius chunk --dataset bmds --strategy fixed_size --size 1000 --delimiter "\n\n"
```

### Advanced Features

```bash
# Verbose logging with timestamps
python -m ius chunk --dataset bmds --strategy fixed_count --count 4 --verbose

# Dry run to preview without processing  
python -m ius chunk --dataset bmds --strategy fixed_size --size 1500 --dry-run

# Combine flags for detailed preview
python -m ius chunk --dataset true-detective --strategy fixed_count --count 6 --dry-run --verbose

# Save output to specific location
python -m ius chunk --dataset bmds --strategy fixed_size --size 2048 \
  --output outputs/bmds_large_chunks.json --preview
```

### CLI Features

- **📊 Smart Progress Bars**: Automatic progress tracking with `tqdm` (only shows when helpful)
- **🔍 Dry Run Mode**: Preview what will be processed with `--dry-run`  
- **📝 Verbose Logging**: Detailed timestamps and module info with `--verbose`
- **✅ Input Validation**: Comprehensive error checking with helpful messages
- **💾 Flexible Output**: Custom output paths or automatic naming

### Example Output

```bash
$ python -m ius chunk --dataset bmds --strategy fixed_count --count 4 --verbose

2024-01-15 23:09:29 - ius.cli.chunk - INFO - Loading dataset: bmds
2024-01-15 23:09:29 - ius.data.loader - INFO - Loaded 34 items from bmds
Processing items: 100%|████████████| 34/34 [00:02<00:00, 12.5 item/s]
2024-01-15 23:09:31 - ius.cli.chunk - INFO - Chunking completed successfully!
2024-01-15 23:09:31 - ius.cli.chunk - INFO - Results saved to: outputs/chunks/bmds_fixed_count_4.json
```

## Configuration

The framework supports flexible configuration through environment variables and code.

### Environment Variables

```bash
# Dataset and output directories  
export IUS_DATASETS_DIR="/path/to/datasets"
export IUS_OUTPUTS_DIR="/path/to/outputs"

# Default chunking parameters
export IUS_DEFAULT_CHUNK_SIZE="1000"    # ~1000 delimiter-separated units
export IUS_DEFAULT_NUM_CHUNKS="4"       # Create 4 chunks (fixed_count)

# System settings
export IUS_MAX_MEMORY="524288000"       # 500MB memory limit (future use)
export IUS_LOG_LEVEL="INFO"             # Logging verbosity
```

### Configuration in Code

```python
from pathlib import Path
from ius.config import get_config, set_config, Config

# Get current configuration (loads from environment)
config = get_config() 
print(f"Datasets directory: {config.datasets_dir}")
print(f"Default chunk size: {config.default_chunk_size}")

# Override configuration programmatically
custom_config = Config(
    datasets_dir=Path("/custom/datasets"),
    default_chunk_size=2000,
    default_num_chunks=6,
    log_level="DEBUG"
)
set_config(custom_config)
```

### Configuration Validation

The configuration system provides automatic validation:

- ✅ **Datasets directory must exist** (required for operation)
- ✅ **Output directories created automatically** as needed
- ✅ **Positive numeric values** enforced for chunk sizes and counts
- ✅ **Valid log levels** (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- ✅ **Environment variable parsing** with sensible fallbacks

## Current Datasets

### BMDS (Birth of Modern Detection Stories)
- **Items**: 34 classic detective stories
- **Documents per item**: 1 (each story is one document)
- **Use case**: Study incremental summarization for detective stories, evaluate on the downstream task of guessing the culprit.

### True Detective  
- **Items**: 191 short mystery puzzles
- **Documents per item**: 1 (each puzzle is one document)
- **Use case**: Study incremental summarization on detective stories, evaluate on the downstream task of guessing the culprit.

## Development Principles ✅ **ACHIEVED**

- **✅ Lean and modular**: Clean, simple, readable, well-documented, modular code that's easily extensible. Optimized for ease of use and reproducibility.
- **✅ Start small**: Everything works reliably with BMDS dataset, ready for expansion
- **⏳ Comprehensive evaluation**: Track experiments systematically with detailed metrics (TODO: Priority 3)
- **⏳ LLM flexibility**: Abstract LLM calls to easily switch between APIs and local models (TODO: Future work)

**Code Quality Standards Met**:
- Zero linting errors, consistent formatting
- 102 comprehensive tests with 90%+ coverage  
- Professional error handling and logging
- Production-ready CLI with user-friendly features

## Getting Started

### 1. Quick Start with CLI

```bash
# Install dependencies
pip install -r requirements.txt

# List available datasets  
python -m ius chunk --list-datasets

# Try basic chunking
python -m ius chunk --dataset bmds --strategy fixed_size --size 1500 --dry-run

# Run actual chunking with progress tracking
python -m ius chunk --dataset bmds --strategy fixed_count --count 4 --verbose
```

### 2. Python API Usage

```python
from ius.data import load_data, list_datasets, get_dataset_info
from ius.chunk.chunkers import chunk_fixed_size, process_dataset_items

# Load and explore datasets
datasets = list_datasets()
info = get_dataset_info("bmds") 
data = load_data("bmds")
print(f"Loaded {data['num_items_loaded']} items")

# Load single item
item_data = load_data("bmds", item_id="ADP02")

# Chunk individual text
document_text = item_data["documents"][0]["content"]
chunks = chunk_fixed_size(document_text, chunk_size=1000, delimiter="\n")

# Process full dataset
results = process_dataset_items(
    items=data["items"],
    strategy="fixed_count", 
    num_chunks=4,
    delimiter="\n"
)
```

### 3. Configuration and Logging

```python
from ius.config import get_config
from ius.logging_config import setup_logging

# Set up structured logging
setup_logging(log_level="INFO", verbose=True)

# Get configuration (loads from environment)
config = get_config()
print(f"Using datasets from: {config.datasets_dir}")
print(f"Default chunk settings: {config.default_chunk_size} size, {config.default_num_chunks} count")
```

### 4. Running Tests

```bash
# Run all tests (102 tests)
python -m pytest

# Run specific test modules
python -m pytest tests/test_chunking.py -v
python -m pytest tests/test_cli_chunk.py -v

# Run with coverage
python -m pytest --cov=ius tests/
```

## New Features & Improvements

The framework has been significantly enhanced with production-ready features:

### ✅ **Robust Error Handling**
- Custom exception hierarchy (`IUSError`, `ChunkingError`, `ValidationError`, `DatasetError`)
- Comprehensive input validation across all functions
- Informative error messages with user guidance
- Graceful handling of edge cases (empty text, missing delimiters, invalid data)

### ✅ **Professional CLI Experience**
- Smart progress bars with `tqdm` (auto-disabled for small operations)
- `--verbose` flag for detailed logging with timestamps
- `--dry-run` mode for safe preview without processing
- Comprehensive help text with examples
- Input validation with clear error messages

### ✅ **Structured Logging System**
- Module-specific loggers with configurable verbosity
- Console and optional file output
- Third-party logger suppression
- Integration with CLI flags and configuration

### ✅ **Configuration Management**
- Environment variable support (`IUS_*` prefix)
- Automatic directory creation and validation
- Programmatic configuration override
- Sensible defaults for research workflows

### ✅ **Comprehensive Test Suite**
- **102 tests** covering all modules and edge cases
- Unit, integration, and CLI tests
- Error handling validation
- Progress bar and logging tests
- 90%+ code coverage

### ✅ **Code Quality**
- Zero linting errors with `ruff` formatting
- Consistent code style and documentation
- Type hints throughout
- Modular, well-organized architecture

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