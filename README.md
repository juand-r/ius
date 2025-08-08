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

1. **Fixed-size chunking**: Split documents into chunks of roughly equal length (measured in characters)
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
│   ├── datasets.py   # Dataset and ChunkedDataset classes for object-oriented data access
│   └── __init__.py   # Data loading convenience functions
├── eval/             # Evaluation and experiment tracking (TODO)  
└── summarization/    # LLM-based summarization with experimental tracking ✅ IMPLEMENTED

datasets/             # Standardized datasets
├── bmds/             # Birth of Modern Detection Stories (34 items)
├── true-detective/   # True Detective puzzles (191 items) 
├── fables/           # Fables collection
└── booookscore/      # Book collection (TODO: populate)

tests/                # Comprehensive test suite (128 tests)
├── test_chunking.py         # Chunking function tests
├── test_cli_chunk.py        # CLI functionality tests  
├── test_data_loader.py      # Data loading tests
├── test_cli_common.py       # CLI utilities tests
├── test_config.py          # Configuration management tests
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

# Basic chunking with fixed size (characters)
python -m ius chunk --dataset bmds --strategy fixed_size --size 10000

# Fixed count chunking  
python -m ius chunk --dataset true-detective --strategy fixed_count --count 8

# Custom delimiter chunking
python -m ius chunk --dataset bmds --strategy fixed_size --size 8000 --delimiter "\n\n"

# List available summarization strategies
python -m ius summarize --list-strategies

# Summarize chunked data (basic)
python -m ius summarize --input outputs/chunks/bmds_fixed_count_3

# Summarize with specific strategy and item
python -m ius summarize --input outputs/chunks/bmds_fixed_count_3 --item ADP02 --strategy summarize_chunks_independently
```

### Advanced Features

```bash
# Verbose logging with timestamps
python -m ius chunk --dataset bmds --strategy fixed_count --count 4 --verbose

# Dry run to preview without processing  
python -m ius chunk --dataset bmds --strategy fixed_size --size 1500 --dry-run

# Combine flags for detailed preview
python -m ius chunk --dataset true-detective --strategy fixed_count --count 6 --dry-run --verbose

# Save output to specific location (specify directory, not filename)
python -m ius chunk --dataset bmds --strategy fixed_size --size 10000 \
  --output outputs/chunks/bmds_large_chunks --preview
```

### Important: Output Path Format

⚠️ **The `--output` parameter expects a DIRECTORY path, not a file path:**

```bash
# ✅ CORRECT - specify directory only
python -m ius chunk --dataset bmds --strategy fixed_size --size 8000 \
  --output outputs/chunks/bmds_custom_chunks

# ❌ WRONG - don't include collection.json in the path  
python -m ius chunk --dataset bmds --strategy fixed_size --size 8000 \
  --output outputs/chunks/bmds_custom_chunks/collection.json
```

The CLI will automatically create:
- `collection.json` (dataset metadata)
- `items/` directory with individual chunk files
- Proper directory structure within the specified output path

### Summarization Commands

The CLI provides comprehensive summarization capabilities with multiple strategies and automatic output naming.

```bash
# Basic summarization (auto-generated output name)
python -m ius summarize --input outputs/chunks/bmds_fixed_count_3

# Summarize specific item with preview
python -m ius summarize --input outputs/chunks/bmds_fixed_count_3 --item ADP02 --preview

# Use independent chunk strategy (each chunk summarized separately)
python -m ius summarize --input outputs/chunks/bmds_fixed_count_3 \
  --strategy summarize_chunks_independently

# Use cumulative strategy with intermediate summaries
python -m ius summarize --input outputs/chunks/bmds_fixed_count_3 \
  --strategy concat_and_summarize --intermediate

# Custom model and prompt with manual output naming
python -m ius summarize --input outputs/chunks/bmds_fixed_count_3 \
  --model gpt-4 --prompt custom-detective-prompt --output detective_analysis

# Summarize single item file directly
python -m ius summarize --input outputs/chunks/bmds_fixed_count_3/items/ADP02.json

# List available summarization strategies
python -m ius summarize --list-strategies

# Skip existing results (default behavior)
python -m ius summarize --input outputs/chunks/bmds_fixed_count_3 --item ADP02

# Overwrite existing results
python -m ius summarize --input outputs/chunks/bmds_fixed_count_3 --item ADP02 --overwrite
```

### Discovering Available Strategies

Before choosing a summarization approach, you can list all available strategies:

```bash
$ python -m ius summarize --list-strategies

Available summarization strategies:

📋 concat_and_summarize
   • Concatenates all chunks into a single text
   • Produces cumulative summaries (final summary of all content)
   • Use --intermediate flag to get progressive summaries
   • Best for: Getting overall summary of entire document

📋 summarize_chunks_independently
   • Summarizes each chunk separately
   • Produces chunk summaries (one summary per chunk)
   • Maintains chunk-level granularity
   • Best for: Analyzing content at chunk level

Note: Use --strategy <name> to specify which strategy to use
Default strategy: concat_and_summarize
```

### Summarization Strategies

**Cumulative Strategy (`concat_and_summarize`)** - Default
- `--intermediate` flag: Creates progressive summaries (chunk1, chunk1+2, chunk1+2+3...)
- `final_only` (default): Creates single summary of all chunks combined

**Independent Strategy (`summarize_chunks_independently`)**
- Always creates separate summary for each chunk individually
- Useful for analyzing chunk-level content and patterns

### CLI Features

- **📊 Smart Progress Bars**: Automatic progress tracking with `tqdm` (only shows when helpful)
- **🔍 Dry Run Mode**: Preview what will be processed with `--dry-run`  
- **📝 Verbose Logging**: Detailed timestamps and module info with `--verbose`
- **✅ Input Validation**: Comprehensive error checking with helpful messages
- **💾 Flexible Output**: Custom output paths or automatic naming
- **🎯 Auto-generated Names**: Intelligent naming based on input, strategy, model, and options
- **⚡ Multiple Strategies**: Support for cumulative and independent summarization approaches
- **💰 Cost Tracking**: Real-time cost estimation and usage reporting
- **🔄 Skip/Overwrite Control**: Automatically skip existing results (default) or force overwrite with `--overwrite`
- **📋 Strategy Discovery**: List and compare available strategies with `--list-strategies`
- **🔗 Command Reproducibility**: Full command history stored in metadata for perfect experiment reproduction

### Example Output

**Chunking:**
```bash
$ python -m ius chunk --dataset bmds --strategy fixed_count --count 4 --verbose

2024-01-15 23:09:29 - ius.cli.chunk - INFO - Loading dataset: bmds
2024-01-15 23:09:29 - ius.data.loader - INFO - Loaded 34 items from bmds
2024-01-15 23:09:29 - ius.cli.chunk - INFO - Processing 34 items...
[1/34] Processing: ADP02
2024-01-15 23:09:29 - ius.cli.chunk - INFO - Created 4 chunks, avg size: 8703.0
[2/34] Processing: ADP06
2024-01-15 23:09:29 - ius.cli.chunk - INFO - Created 4 chunks, avg size: 10969.0
...
📊 Summary Statistics:
  total_items: 34
  total_chunks: 136
  total_characters: 1027755
  avg_chunks_per_item: 4.0
  processing_time_seconds: 0.1
  error_count: 0
✅ Collection and 34 chunked items saved to: outputs/chunks/bmds_fixed_count_4
2024-01-15 23:09:31 - ius.cli.chunk - INFO - Chunking completed successfully!
```

**Summarization:**
```bash
$ python -m ius summarize --input outputs/chunks/bmds_fixed_count_4 --item ADP02 --preview

🎯 Auto-generated output name: bmds_fixed_count_4_ADP02_concat_default-concat-prompt_final
🤖 Starting summarization...
📥 Input: outputs/chunks/bmds_fixed_count_4
📤 Output: outputs/summaries/bmds_fixed_count_4_ADP02_concat_default-concat-prompt_final
⚡ Strategy: concat_and_summarize
🧠 Model: gpt-4.1-mini
📝 Prompt: default-concat-prompt
📋 Processing specified item: ADP02

🔄 Processing ADP02...
📦 Loaded 4 chunks (34,810 chars)
👀 First chunk preview: WAS it a specter?
For days I could not answer this question...

💰 Estimated Cost: $0.005046
💰 Actual API Cost: $0.004849
🎉 Summarization completed!
⏱️  Total time: 8.2s
💰 Total cost: $0.004849
🔢 Total tokens: 9,491
📁 Results saved to: outputs/summaries/bmds_fixed_count_4_ADP02_concat_default-concat-prompt_final
```

## Summarization Usage

The IUS framework provides LLM-based summarization with comprehensive experimental tracking and cost monitoring. Available through both CLI commands (see above) and Python API.

### Basic Summarization

```python
from ius.summarization import summarize

# Summarize a single item (all chunks within one item)
result = summarize(
    strategy="concat_and_summarize",
    dataset="bmds", 
    scope="item",
    item_id="ADP02"
)

print(f"Experiment ID: {result['experiment_id']}")
print(f"Results saved to: {result['experiment_dir']}")
print(f"Total cost: ${result['total_usage']['total_cost']:.6f}")
```

### Scope Options

The summarization system supports three different scopes:

#### 1. Single Item Summarization
```python  
# Summarize all chunks in one specific item
result = summarize(
    strategy="concat_and_summarize",
    dataset="bmds",
    scope="item", 
    item_id="ADP02",
    model="gpt-4o-mini"
)
```

#### 2. Full Dataset Summarization  
```python
# Summarize every item in the dataset (creates one summary per item)
result = summarize(
    strategy="concat_and_summarize", 
    dataset="bmds",
    scope="dataset"  # Processes all 34 items in BMDS
)

# Each item gets its own summary file:
# outputs/summaries/{timestamp}_concat_and_summarize_dataset/results/
# ├── ADP02.txt
# ├── ADP06.txt  
# ├── ASH03.txt
# └── ... (one per item)
```

#### 3. Document Range Summarization
```python
# Summarize specific chunk ranges within an item
result = summarize(
    strategy="concat_and_summarize",
    dataset="bmds", 
    scope="doc_range",
    item_id="ADP02",
    doc_range="0:2"  # Chunks 0, 1, and 2
)

# Or summarize just one chunk
result = summarize(
    strategy="concat_and_summarize",
    dataset="bmds",
    scope="doc_range", 
    item_id="ADP02",
    doc_range="1"  # Just chunk 1
)
```

### Summarization Strategies

#### No-Op Strategy (Baseline)
```python
# Simple concatenation without LLM processing
result = summarize(
    strategy="no_op",
    dataset="bmds",
    scope="item",
    item_id="ADP02"
)
# Returns concatenated chunks as baseline for comparison
```

#### Concat and Summarize
```python
# Concatenate all chunks and send to LLM for summarization
result = summarize(
    strategy="concat_and_summarize", 
    dataset="bmds",
    scope="item",
    item_id="ADP02",
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=500
)
```

### Custom Prompts

```python
# Use custom system and user prompts with template variables
custom_prompts = {
    "system": "You are an expert at analyzing detective stories and identifying key plot elements.",
    "user": """Please provide a concise summary of this detective story focusing on:
1. The main characters
2. The mystery or crime
3. Key clues discovered
4. The resolution

Text to summarize ({word_count} words, {char_count} characters):
{text}"""
}

result = summarize(
    strategy="concat_and_summarize",
    dataset="bmds", 
    scope="item",
    item_id="ADP02",
    system_and_user_prompt=custom_prompts,
    model="gpt-4o-mini"
)
```

### Cost Management and Confirmation

```python
# Get cost estimate and require confirmation before API calls
result = summarize(
    strategy="concat_and_summarize",
    dataset="bmds",
    scope="dataset",  # This could be expensive for full dataset!
    ask_user_confirmation=True,  # Will show estimate and ask for confirmation
    model="gpt-4o"
)

# The system will print:
# 💰 Estimated Cost: $0.045000
# Do you want to proceed? (y/N): 
```

### Working with Pre-chunked Data

```python
from ius.data import ChunkedDataset

# Load and inspect chunked data using the new ChunkedDataset class
chunked_dataset = ChunkedDataset("outputs/chunks/bmds_fixed_count_3")
print(f"Available items: {chunked_dataset.item_ids}")
print(f"Dataset name: {chunked_dataset.name}")

# Get chunks for a specific item
chunks = chunked_dataset.get_item_chunks("ADP02")
print(f"ADP02 has {len(chunks)} chunks")

# Get statistics for an item
stats = chunked_dataset.get_item_stats("ADP02")
print(f"Item stats: {stats}")

# Load collection metadata
collection_info = chunked_dataset.load_collection_metadata()
print(f"Chunking strategy used: {collection_info['chunking_info']['strategy']}")

# Use specific chunked directory with summarization
result = summarize(
    strategy="concat_and_summarize",
    dataset="bmds",
    scope="item", 
    item_id="ADP02",
    chunked_file_path="outputs/chunks/bmds_fixed_size_1000"  # Now points to directory
)
```

### Chunked Data Directory Structure

**Important**: Understanding this structure is critical for working with chunked data files.

The chunked data generated by `python -m ius chunk` creates a **directory structure** that mirrors the original dataset format:

```
outputs/chunks/bmds_fixed_count_3/
├── collection.json          # Collection-level metadata and statistics
└── items/                   # Individual chunked items
    ├── ADP02.json          # Chunked data for item ADP02
    ├── ADP06.json          # Chunked data for item ADP06
    ├── ASH03.json          # Chunked data for item ASH03
    └── ...                 # One JSON file per item
```

#### Collection Metadata (`collection.json`)
```json
{
  "domain": "detective_stories",
  "source": "BMDS (https://github.com/ahmmnd/BMDS)",
  "created": "2024-12-20T00:00:00",
  "num_items": 34,
  "total_documents": 34,
  "description": "Dataset description",
  "items": ["ADP02", "ADP06", "ASH03", ...],
  "chunking_info": {
    "strategy": "fixed_count",
    "overall_stats": {
      "total_items": 34,
      "total_chunks": 102,
      "total_characters": 1027755,
      "avg_chunks_per_item": 3.0,
      "processing_time_seconds": 0.1,
      "error_count": 0
    },
    "timestamp": "2025-08-01 22:32:56"
  }
}
```

#### Individual Item Structure (`items/{item_id}.json`)
```json
{
  "item_metadata": {
    "item_id": "ADP02",
    "num_documents": 1,
    "chunking_method": "fixed_count",
    "chunking_params": {
      "delimiter": "\n",
      "num_chunks": 3
    },
    "chunking_timestamp": "2025-08-01 22:32:56"
  },
  "documents": [
    {
      "chunks": [
        "First chunk text...",
        "Second chunk text...",
        "Third chunk text..."
      ],
      "metadata": {
        "original_metadata": {
          "story_code": "ADP02",
          "story_title": "The Gray Madam",
          "author_name": "Unknown"
        },
        "chunking_stats": {
          "num_chunks": 3,
          "total_chars": 34811,
          "avg_chunk_size": 11603.7,
          "original_length": 34811
        }
      }
    }
  ]
}
```

**Key Structure Benefits:**
- **Clean separation**: Collection-wide stats stay in `collection.json`, item-specific data in individual files
- **Scalable**: Each item is a separate file, enabling efficient processing of large datasets
- **Metadata preservation**: Original item metadata is preserved alongside chunking information
- **Well-organized**: Mirrors the original dataset structure for intuitive navigation

### Output and Experimental Tracking

Each summarization run creates a comprehensive experimental record:

```
outputs/summaries/20240115_143022_concat_and_summarize_item/
├── config.json              # Complete experiment configuration
├── summary_metadata.json    # Per-item processing metadata  
└── results/
    └── ADP02.txt            # Generated summary text
```

**config.json** contains:
- Experiment parameters (strategy, scope, model, prompts)
- Processing timestamp and unique experiment ID
- Total usage statistics (tokens, costs, API calls)
- Chunked data file path used

**summary_metadata.json** contains per-item details:
- Input/output text lengths (characters and words)
- Processing time and token usage
- Model parameters and costs
- Chunk counts and document ranges processed

### Advanced Usage

```python  
# Direct function usage for integration with other code
from ius.summarization import concat_and_summarize

chunks = ["First chunk text...", "Second chunk text...", "Third chunk..."]

result = concat_and_summarize(
    chunks=chunks,
    model="gpt-4o-mini", 
    system_and_user_prompt={
        "system": "You are a helpful summarization assistant.",
        "user": "Summarize the following text concisely:\n\n{text}"
    },
    temperature=0.0,
    max_tokens=300
)

print(f"Summary: {result['response']}")
print(f"Cost: ${result['usage']['total_cost']:.6f}")
print(f"Tokens used: {result['usage']['total_tokens']}")
```

## Dependency Management

This project uses **pip-tools** to keep `requirements.txt` synchronized with `pyproject.toml`:

```bash
# Sync requirements.txt with pyproject.toml changes
make sync-requirements

# After adding/changing dependencies in pyproject.toml:
# 1. Run sync command above
# 2. Commit both pyproject.toml and requirements*.txt files
```

**File structure:**
- `pyproject.toml` - Source of truth for dependencies
- `requirements.txt` - Auto-generated, pinned production dependencies  
- `requirements-dev.txt` - Auto-generated, includes dev tools (pytest, ruff)
- `requirements*.in` - Input files for pip-tools (don't edit directly)

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
- 128 comprehensive tests with 90%+ coverage  
- Professional error handling and logging
- Production-ready CLI with user-friendly features

## Getting Started

### 1. Quick Start with CLI

```bash
# Install dependencies (choose one)
make install          # Recommended: production dependencies
make install-dev      # Recommended: includes testing/linting tools

# Or install manually:
pip install -r requirements.txt        # Production only  
pip install -r requirements-dev.txt    # Development

# For summarization features, set up OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# List available datasets  
python -m ius chunk --list-datasets

# Try basic chunking (size = characters, not words)
python -m ius chunk --dataset bmds --strategy fixed_size --size 8000 --dry-run

# Run actual chunking with progress tracking
python -m ius chunk --dataset bmds --strategy fixed_count --count 4 --verbose

# Try summarization (requires OPENAI_API_KEY)
python -m ius summarize --input outputs/chunks/bmds_fixed_count_4 --item ADP02 --preview
```

### 2. Python API Usage

#### Data Loading and Chunking
```python
from ius.data import load_data, list_datasets, get_dataset_info, Dataset, ChunkedDataset
from ius.chunk.chunkers import chunk_fixed_size, process_dataset_items

# Load and explore datasets
datasets = list_datasets()
info = get_dataset_info("bmds") 
data = load_data("bmds")
print(f"Loaded {data['num_items_loaded']} items")

# Use the new Dataset class for object-oriented access
dataset = Dataset("datasets/bmds")
print(f"Dataset has {len(dataset.item_ids)} items")
item_data = dataset.load_item("ADP02")
documents = dataset.get_item_documents("ADP02")

# Load chunked data using ChunkedDataset class
chunked_dataset = ChunkedDataset("outputs/chunks/bmds_fixed_count_3")
chunks = chunked_dataset.get_item_chunks("ADP02")
stats = chunked_dataset.get_item_stats("ADP02")

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

#### Summarization
```python
from ius.summarization import summarize

# Quick single-item summarization
result = summarize(
    strategy="concat_and_summarize",
    dataset="bmds",
    scope="item", 
    item_id="ADP02"
)

# Full dataset processing with custom prompts
detective_prompts = {
    "system": "You are an expert at analyzing detective stories.", 
    "user": "Summarize this detective story, focusing on the mystery and resolution:\n\n{text}"
}

result = summarize(
    strategy="concat_and_summarize",
    dataset="bmds", 
    scope="dataset",
    system_and_user_prompt=detective_prompts,
    model="gpt-4o-mini"
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

### 4. Development Commands (Makefile)

For convenience, common development tasks are available via `make`:

```bash
# See all available commands
make help

# Install dependencies
make install        # Production only
make install-dev    # Includes testing/linting tools

# Development workflow
make test          # Run all tests (128 tests)
make lint          # Check code style
make fix-lint      # Fix code formatting issues

# Dependency management
make sync-requirements  # Update requirements.txt from pyproject.toml
```

### 5. Running Tests Manually

```bash
# Run all tests (128 tests)
python -m pytest

# Run specific test modules
python -m pytest tests/test_chunking.py -v
python -m pytest tests/test_cli_chunk.py -v

# Run with coverage
python -m pytest --cov=ius tests/
```

## CLI Testing Results ✅ **COMPREHENSIVE VALIDATION**

The chunking CLI has been thoroughly tested across multiple scenarios to ensure production reliability:

### Testing Summary

| Test Scenario | Strategy | Dataset | Items | Chunks | Avg/Item | Success Rate | Notes |
|---------------|----------|---------|-------|--------|----------|--------------|-------|
| **BMDS Standard** | fixed_count=2 | bmds | 34/34 | 68 | 2.0 | 100% | ✅ Perfect fixed count |
| **BMDS Adaptive** | fixed_size=15k | bmds | 34/34 | 88 | 2.6 | 100% | ✅ Intelligent size adaptation |
| **BMDS High Count** | fixed_count=5 | bmds | 34/34 | 170 | 5.0 | 100% | ✅ Large chunk counts |
| **True-Detective** | fixed_count=3 | true-detective | 191/191 | 573 | 3.0 | 100% | ✅ Large dataset + alt delimiter |

### Key Validations Achieved

**✅ Strategy Flexibility**: Both `fixed_count` and `fixed_size` strategies work correctly
- Fixed-count produces exact chunk counts as expected
- Fixed-size adapts intelligently to content length while respecting delimiters

**✅ Dataset Diversity**: Works across different text formats and sizes
- BMDS: 34 classic detective stories with newline delimiters
- True-Detective: 191 mystery puzzles with period delimiters (no newlines)

**✅ Error Resilience**: Graceful handling of edge cases
- Automatic delimiter detection and fallback options
- Clear error messages when delimiters are incompatible
- Processes what it can, reports what it can't

**✅ Data Organization**: New directory structure works flawlessly
- Clean separation of collection-level vs. item-level metadata
- Scalable individual JSON files per item
- Intuitive organization mirroring original dataset structure

**✅ Mathematical Accuracy**: All statistics are precise and consistent
- Chunk counts, averages, and character totals verified across all tests
- Success rates and error handling accurately tracked

**✅ Large Scale Processing**: Successfully handles datasets with 191+ items
- Efficient processing of hundreds of items
- Consistent performance across different dataset sizes

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
- **128 tests** covering all modules and edge cases
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

### ✅ Recently Completed
- **LLM-based Summarization**: OpenAI integration with cost tracking and experimental management
- **Summarization CLI**: Command-line interface with multiple strategies, auto-naming, and comprehensive options
- **Multiple Strategies**: Cumulative (concat_and_summarize) and independent (summarize_chunks_independently) approaches
- **Flexible Scope Handling**: Single item, full dataset, and document range summarization
- **Improved Data Structure**: New directory-based chunked output format with collection.json + items/
- **Object-Oriented Data Access**: Dataset and ChunkedDataset classes for clean data handling
- **Comprehensive CLI Testing**: Validated across multiple strategies, datasets, and delimiters (191+ items tested)
- **Enhanced Error Handling**: Graceful delimiter mismatch handling and detailed error reporting
- **Skip/Overwrite Control**: Intelligent result caching with `--overwrite` flag for cost-effective incremental processing
- **Strategy Discovery**: Built-in `--list-strategies` command for easy exploration of available approaches
- **Command Reproducibility**: Complete command tracking in both chunking and summarization metadata for scientific transparency

### 🚧 In Progress  
- **Evaluation Framework**: Metrics for content preservation, summary quality, and computational efficiency

### 📋 Planned Features
- **Multi-document datasets**: News sequences, TV show episodes, book series
- **Advanced chunking**: Semantic and structure-aware chunking strategies  
- **Incremental strategies**: Various approaches to update summaries efficiently (iterative_summarize)
- **Local LLM support**: Integration with local models (Ollama, Hugging Face)
- **Interactive tools**: Visualization and analysis of incremental summarization results

## Research Questions

- How do different chunking strategies affect incremental summarization quality?
- What are the trade-offs between summary freshness and computational cost?
- How does incremental performance vary across document types and domains?
- Can we predict when full re-summarization is needed vs. incremental updates?

---

*This is a research framework. Code should be clean, well-documented, and easy to extend for various experimental approaches.*