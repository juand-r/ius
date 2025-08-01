# Implementation Plan

## Phase 1: Core Infrastructure

### 1.1 Chunking Module (`ius/chunk/`)

**Purpose**: Implement various strategies to split documents into chunks for incremental processing.

**Structure**:
```
ius/chunk/
├── __init__.py          # Exports main classes/functions
├── chunkers.py          # All chunking strategies in one file
└── utils.py             # Content validation utilities
```

**Key Classes** (`chunkers.py`):

```python
# Simple chunking functions - no complex class hierarchy
def chunk_fixed_size(text: str, chunk_size: int, delimiter: str = "\n") -> List[str]:
    """Split text into fixed-size chunks, respecting delimiter boundaries"""
    pass

def chunk_fixed_count(text: str, num_chunks: int, delimiter: str = "\n") -> List[str]:
    """Split text into N chunks, respecting delimiter boundaries"""
    pass

def chunk_custom(text: str, strategy: str, delimiter: str = "\n", **kwargs) -> List[str]:
    """Split text using custom strategy (dataset-specific, configured later)"""
    pass

# Simple validation
def validate_chunks(original_text: str, chunks: List[str], delimiter: str = "\n") -> bool:
    """Verify chunks preserve all content when joined with delimiter"""
    return delimiter.join(chunks) == original_text
```

**Chunking Requirements**:
- **No overlap**: Chunks are consecutive with no gaps or overlaps
- **Delimiter-aware**: Respect text boundaries (don't split mid-word/sentence)
- **Content preservation**: Joining chunks with delimiter must recover original document exactly
- **Validation**: Built-in verifier to ensure no content is lost during chunking
- **Configurable boundaries**: Default `"\n"` delimiter, can use `"\n\n"` (paragraphs) or others

**Implementation Priority**:
1. `chunk_fixed_size()` function with delimiter parameter (default `"\n"`)
2. `validate_chunks()` utility for content preservation with delimiter support
3. `chunk_fixed_count()` with delimiter awareness
4. `chunk_custom()` placeholder for dataset-specific strategies (to be configured later)

### 1.2 Summarization Module (`ius/summarization/`)

**Purpose**: Core summarization strategies including incremental approaches.

**Structure**:
```
ius/summarization/
├── __init__.py          # Main exports
├── base.py             # Abstract summarizer classes
├── utils.py            # LLM utilities, prompt management
├── baselines.py        # Simple baseline approaches  
├── incremental.py      # Incremental update strategies
└── models.py           # Multi-API LLM client abstractions

ius/cli/                 # Command-line interface modules
├── __init__.py          # CLI package initialization
├── chunk.py            # python -m ius.chunk commands
├── summarize.py        # python -m ius.summarize commands
├── evaluate.py         # python -m ius.evaluate commands
├── experiment.py       # python -m ius.experiment commands
├── analyze.py          # python -m ius.analyze commands
└── common.py           # Shared CLI utilities and parsers

prompts/                 # Prompt templates directory
├── summarization/       # Summarization prompts
├── evaluation/          # Evaluation prompts  
└── templates/           # Reusable prompt templates
```

**Key Classes**:

```python
# Abstract interfaces
class BaseSummarizer:
    def summarize_chunks(self, chunks: List[Dict]) -> str:
        """Summarize a sequence of chunks"""
        pass

class IncrementalSummarizer(BaseSummarizer):
    def initialize_summary(self, initial_chunks: List[Dict]) -> str:
        """Create initial summary"""
        pass
    
    def update_summary(self, current_summary: str, new_chunks: List[Dict]) -> str:
        """Update summary with new chunks"""
        pass

# Multi-API LLM abstraction for easy model switching
class LLMClient:
    """Abstract base for different LLM APIs"""
    
    @classmethod
    def create(cls, api_type: str, api_key: str, model: str):
        """Factory method for creating API-specific clients"""
        if api_type == "openai":
            return OpenAIClient(api_key, model)
        elif api_type == "anthropic":
            return AnthropicClient(api_key, model)
        elif api_type == "together":
            return TogetherClient(api_key, model)
        else:
            raise ValueError(f"Unknown API type: {api_type}")
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generic completion interface"""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Token counting for cost/length management"""
        pass
```

**Multi-API Support (`models.py`)**:
- OpenAI API (gpt-4.1-mini, gpt-4, etc.)
- Anthropic API (Claude models)
- Together AI API (open source models)
- Unified interface with automatic retry logic and rate limiting
- Cost tracking across different APIs
- Graceful error handling and fallback strategies

**LLM Utilities (`utils.py`)**:
- Prompt template loading and variable substitution  
- Token counting and cost tracking across APIs
- Retry logic with exponential backoff
- Response parsing and validation utilities
- Checkpointing for long-running operations

**Prompt Management**:
- Template system with variable substitution (e.g., {document} → "detective story")
- Organized prompt library in `prompts/` directory
- Reusable components and consistent formatting

**Baseline Strategies (`baselines.py`)**:
1. **No-op**: Summary = original text (for downstream task evaluation)
2. **Concatenate + Summarize**: Concatenate chunks 1...k and summarize

**Incremental Strategies (`incremental.py`)**:
1. **Summary Merging**: Separately summarize new content, then merge summaries
2. **Contextualized Update**: Use previous summary as context for new content
3. **Hierarchical**: Multi-level summaries with different update frequencies
4. **Selective Update**: Decide when full re-summarization is needed

### 1.3 Evaluation Module (`ius/eval/`)

**Purpose**: Systematic experiment tracking, metrics, and result analysis.

**Structure**:
```
ius/eval/
├── __init__.py          # Main exports
├── experiments.py       # Experiment orchestration  
├── metrics.py          # Evaluation metrics
├── tracking.py         # Result storage and tracking
├── analysis.py         # Result analysis and visualization
└── outputs.py          # Output format management
```

**Experiment Management**:
```python
class ExperimentRunner:
    def run_experiment(self, 
                      dataset_name: str,
                      chunker_config: Dict,
                      summarizer_config: Dict,
                      eval_config: Dict) -> ExperimentResult:
        """Run a complete experiment"""
        pass
    
    def batch_experiments(self, experiments: List[Dict]) -> List[ExperimentResult]:
        """Run multiple experiments"""
        pass
```

**Metrics Categories**:
1. **Content Quality**: ROUGE, BLEU, semantic similarity
2. **Downstream Task Performance**: Accuracy on detective story culprit identification
3. **Efficiency**: Processing time, token usage, API costs  
4. **Incremental-Specific**: Update consistency, drift detection
5. **Content Preservation**: Validation that chunking preserves all content

**Output Organization**:
```
outputs/
├── experiments/
│   ├── {experiment_name}_{timestamp}/
│   │   ├── config.yaml              # Experiment configuration
│   │   ├── results.json             # Metrics and summary results
│   │   ├── summaries/               # Generated summaries by item
│   │   │   ├── {item_id}_chunks.json    # Chunking results
│   │   │   ├── {item_id}_summary.txt    # Final summary text
│   │   │   └── {item_id}_logs.json      # Processing logs/intermediate
│   │   └── analysis/                # Analysis outputs
│   │       ├── metrics_summary.json     # Aggregated metrics
│   │       └── visualizations/          # Plots, charts, etc.
└── cache/                           # Cached LLM responses (optional)
```

**Tracking**:
- Timestamped experiment directories for reproducibility
- JSON storage for structured results and metrics
- Configuration preservation for exact reproduction
- Detailed per-item outputs for debugging and analysis
- Parameter grid search support
- Result comparison and statistical testing

## Phase 2: Integration & Testing

### 2.1 End-to-End Pipeline

**Goal**: Complete pipeline from raw data → chunks → summaries → evaluation

**Test with BMDS**:
1. Load BMDS dataset
2. Apply different chunking strategies with content validation
3. Run baseline summarization (no-op, concatenate+summarize)
4. Evaluate on downstream task (culprit identification)
5. Track efficiency metrics and generate analysis reports

### 2.2 Configuration System

**YAML-based experiment configs**:
```yaml
experiment:
  name: "bmds_fixed_size_baseline"
  dataset: "bmds"
  
chunking:
  strategy: "fixed_size"
  chunk_size: 1000
  delimiter: "\n"     # Respect newline boundaries
  overlap: 0          # No overlap - consecutive chunks only
  
summarization:
  strategy: "concatenate_summarize"  
  model: "gpt-4.1-mini"
  max_tokens: 150
  prompt_template: "detective_summary"
  
evaluation:
  metrics: ["rouge", "culprit_accuracy", "cost", "content_preservation"]
  downstream_task: "detective_culprit"
  human_eval: false
```

### 2.3 CLI Interface

**Core Command-line Tools**:
```bash
# Chunking operations
python -m ius.chunk --dataset bmds --strategy fixed_size --size 2048
python -m ius.chunk --dataset true-detective --strategy fixed_count --count 10
python -m ius.chunk --dataset bmds --strategy custom --delimiter "\n\n"

# Summarization operations  
python -m ius.summarize --dataset bmds --method incremental --model gpt-4.1-mini --api openai
python -m ius.summarize --dataset bmds --method concatenate --model claude-3-haiku --api anthropic
python -m ius.summarize --config experiments/bmds_incremental.yaml

# Individual evaluation
python -m ius.evaluate --summaries outputs/summaries.json --metrics rouge,bleu,culprit_accuracy
python -m ius.evaluate --baseline no-op --dataset bmds --downstream-task culprit

# Full experiment pipeline
python -m ius.experiment run --config experiments/bmds_baseline.yaml
python -m ius.experiment batch --config-dir experiments/parameter_sweep/
python -m ius.experiment resume --checkpoint outputs/experiments/bmds_baseline_20241220_143022/

# Analysis and comparison
python -m ius.analyze --experiment-dir outputs/experiments/bmds_baseline_20241220_143022/
python -m ius.compare --experiments exp1/ exp2/ --metrics rouge,culprit_accuracy
```

**CLI Features**:
- Argparse-based interface (simple, no extra dependencies)
- Comprehensive help system with examples
- Configuration file support (YAML)
- Progress tracking and checkpointing
- Resume capability for interrupted experiments
- Flexible output formatting (JSON, CSV, pretty-print)

## Phase 3: Advanced Features

### 3.1 Advanced Chunking
- Semantic similarity-based chunking
- Content-aware boundary detection
- Cross-document chunking for multi-document items

### 3.2 Sophisticated Incremental Strategies
- Learning-based update decisions
- Summary compression and hierarchical structures
- Temporal decay and relevance weighting

### 3.3 Comprehensive Evaluation
- Automated quality assessment
- Interactive evaluation interfaces
- Longitudinal analysis tools

## Implementation Order

**Week 1-2**: Core infrastructure
1. `ius/chunk/chunkers.py` with simple chunking functions and validation
2. `ius/summarization/utils.py`, `models.py` (start with gpt-4.1-mini)  
3. `prompts/` directory structure and template system
4. `ius/summarization/baselines.py` (no-op, concatenate+summarize)

**Week 3**: Integration and testing
5. `ius/eval/experiments.py` (basic experiment runner)
6. End-to-end pipeline with BMDS dataset
7. JSON result storage and basic metrics (including culprit accuracy)

**Week 4+**: Enhancement and expansion  
7. More chunking strategies
8. Incremental summarization approaches
9. Advanced evaluation and analysis

## Success Criteria

- [ ] Can chunk BMDS documents with content preservation validation
- [ ] Can run baseline summarization (no-op, concatenate+summarize)
- [ ] Can evaluate downstream task performance (detective culprit identification)
- [ ] Can track experiments with organized output structure and reproduce from config
- [ ] Experiment outputs are clearly organized and easy to analyze
- [ ] Code is clean, modular, well-documented, readable, and easily extensible (following development principles)

## Resolved Decisions

1. **Chunking overlap**: ❌ No overlap - consecutive chunks only with content preservation validation
2. **Summary formats**: Will depend on prompts (to be provided later)
3. **Evaluation baselines**: No-op (summary=original) and concatenate+summarize baselines
4. **Model selection**: Start with gpt-4o-mini  
5. **Result storage**: JSON files
6. **Prompts**: Organized in `prompts/` directory with template system (prompts to be provided later)

---

*This plan focuses on getting a working system with BMDS before expanding scope. Each module is designed to be independently testable and easily extensible.*