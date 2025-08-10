"""
Command-line interface for summarization operations.

Usage:
    python -m ius summarize --input outputs/chunks/ipython_test --output my_experiment --model gpt-4.1-mini
    python -m ius summarize --input outputs/chunks/ipython_test/items/ADP02.json --output my_experiment
    python -m ius summarize --input outputs/chunks/ipython_test --item ADP02 --output test_run
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from ius.data import ChunkedDataset
from ius.exceptions import DatasetError, ValidationError
from ius.logging_config import get_logger, setup_logging
from ius.summarization import concat_and_summarize, summarize_chunks_independently, iterative_summarize, save_summaries


# Set up logger for this module
logger = get_logger(__name__)


def _truncate_template_vars(template_vars: Dict[str, Any], max_length: int = 89) -> Dict[str, Any]:
    """Truncate string values in template_vars to max_length characters."""
    truncated = {}
    for key, value in template_vars.items():
        if isinstance(value, str) and len(value) > max_length:
            truncated[key] = value[:max_length] + "..."
        else:
            truncated[key] = value
    return truncated


def _generate_output_name(
    input_path: str,
    item_id: str | None,
    strategy: str,
    prompt_name: str | None,  # Can be None
    final_only: bool
) -> str:
    """Generate automatic output directory name based on input parameters."""
    input_path_obj = Path(input_path)
    
    # Get base name from input
    if input_path_obj.is_file():
        # Single file: use item name from filename
        base_name = input_path_obj.stem
        scope = "single"
    else:
        # Directory: use directory name
        base_name = input_path_obj.name
        scope = "all" if item_id is None else item_id
    
    # Clean strategy name (shorten to key parts)
    if strategy == "concat_and_summarize":
        strategy_short = "concat"
    elif strategy == "summarize_chunks_independently":
        strategy_short = "independent" 
    elif strategy == "iterative_summarize":
        strategy_short = "iterative"
    else:
        strategy_short = strategy
    
    # Handle None prompt_name by using strategy defaults
    if prompt_name is None:
        if strategy == "concat_and_summarize":
            prompt_name = "default-concat-prompt"
        elif strategy == "summarize_chunks_independently":
            prompt_name = "default-independent-chunks"
        elif strategy == "iterative_summarize":
            prompt_name = "incremental"
        else:
            prompt_name = "default"
    
    # Clean prompt name (remove path separators)
    clean_prompt = prompt_name.replace("/", "-").replace("\\", "-")
    
    # Build name components
    components = [base_name]
    if scope != "single":
        components.append(scope)
    components.append(strategy_short)
    components.append(clean_prompt)
    components.append("final" if final_only else "intermediate")
    
    return "_".join(components)


def summarize_chunks(
    input_path: str,
    output_name: str | None = None,
    item_id: str | None = None,
    strategy: str = "concat_and_summarize",
    model: str = "gpt-4.1-mini",
    prompt_name: str | None = None,  # No default - let strategy functions handle it
    final_only: bool = True,
    preview: bool = False,
    overwrite: bool = False,
    optional_summary_length: str = "summary",
) -> Dict[str, Any]:
    """
    CLI wrapper for summarizing chunks with progress printing and file I/O.

    Args:
        input_path: Path to chunked dataset directory or specific item JSON
        output_name: Name for the experiment output directory (auto-generated if None)
        item_id: Specific item ID to process (optional, processes all if None)
        strategy: Summarization strategy ("concat_and_summarize" or "summarize_chunks_independently")
        model: LLM model to use for summarization
        prompt_name: Name of prompt template to use
        final_only: Whether to return only final summary or intermediate ones too
        preview: Whether to show summary previews
        overwrite: Whether to overwrite existing item results (default: skip existing items)

    Returns:
        Dictionary with summarization results and metadata
    """
    # Capture the command that generated this experiment for reproducibility
    # Replace the full path to __main__.py with user-friendly "python -m ius" format
    argv_copy = sys.argv.copy()
    if argv_copy[0].endswith("__main__.py"):
        argv_copy[0] = "python -m ius"
    command_run = " ".join(argv_copy)
    
    start_time = time.time()
    
    # Generate output name if not provided
    if output_name is None:
        output_name = _generate_output_name(input_path, item_id, strategy, prompt_name, final_only)
        print(f"🎯 Auto-generated output name: {output_name}")
    
    print(f"🤖 Starting summarization...")
    print(f"📥 Input: {input_path}")
    print(f"📤 Output: outputs/summaries/{output_name}")
    print(f"⚡ Strategy: {strategy}")
    print(f"🧠 Model: {model}")
    # Display prompt name (show function default if None)
    display_prompt = prompt_name or "(function default)"
    print(f"📝 Prompt: {display_prompt}")
    
    # Determine if input is a directory or specific file
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file() and input_path_obj.suffix == '.json':
        # Single item file
        parent_dir = input_path_obj.parent.parent  # Go up from items/ to collection dir
        chunked_dataset = ChunkedDataset(str(parent_dir))
        item_id = input_path_obj.stem  # Get filename without .json
        items_to_process = [item_id]
        print(f"📋 Processing single item: {item_id}")
    else:
        # Directory with chunked dataset
        chunked_dataset = ChunkedDataset(input_path)
        if item_id:
            items_to_process = [item_id]
            print(f"📋 Processing specified item: {item_id}")
        else:
            items_to_process = chunked_dataset.item_ids
            print(f"📋 Processing {len(items_to_process)} items: {', '.join(items_to_process)}")
    
    # Extract domain from collection metadata (fallback to "story" if not found)
    collection_metadata = chunked_dataset.metadata
    domain = collection_metadata.get("domain", "story")
    print(f"📚 Dataset domain: {domain}")
    
    results = {}
    total_cost = 0.0
    total_tokens = 0
    
    for current_item_id in items_to_process:
        print(f"\n🔄 Processing {current_item_id}...")
        
        # Check if item already exists and should be skipped
        output_dir = f"outputs/summaries/{output_name}"
        item_output_file = Path(output_dir) / "items" / f"{current_item_id}.json"
        
        if item_output_file.exists() and not overwrite:
            print(f"⏭️  Skipping {current_item_id} (already exists, use --overwrite to replace)")
            # Still need to track this item for collection.json
            results[current_item_id] = {
                "skipped": True,
                "item_metadata": {"skipped": True}
            }
            continue
        
        # Load chunks for this item
        item_data = chunked_dataset.load_item(current_item_id)
        
        # Add chunk_file information to chunking_stats for traceability
        chunk_file_path = f"{chunked_dataset.collection_path}/items/{current_item_id}.json"
        for doc in item_data["documents"]:
            if "metadata" in doc and "chunking_stats" in doc["metadata"]:
                doc["metadata"]["chunking_stats"]["chunk_file"] = chunk_file_path
        
        chunks = []
        for doc in item_data["documents"]:
            chunks.extend(doc["chunks"])  # Use "chunks" (plural) not "chunk"
        
        print(f"📦 Loaded {len(chunks)} chunks ({sum(len(c) for c in chunks):,} chars)")
        
        if preview and chunks:
            preview_text = chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0]
            print(f"👀 First chunk preview: {preview_text}")
        
        # Validate strategy and final_only combination
        if final_only and strategy == "summarize_chunks_independently":
            raise ValueError(
                "Strategy 'summarize_chunks_independently' requires --intermediate flag. "
                "This strategy produces one summary per chunk, so there's no single 'final' summary. "
                "Use --intermediate to get all chunk summaries."
            )
        
        if final_only and strategy == "iterative_summarize":
            raise ValueError(
                "Strategy 'iterative_summarize' requires --intermediate flag. "
                "This strategy builds incremental summaries step-by-step. "
                "Use --intermediate to see the progression of summaries."
            )
        
        # Run summarization based on strategy
        if strategy == "concat_and_summarize":
            # Only pass prompt_name if user specified one
            kwargs = {
                "chunks": chunks,
                "final_only": final_only,
                "model": model,
                "domain": domain,
                "optional_summary_length": optional_summary_length
            }
            if prompt_name is not None:
                kwargs["prompt_name"] = prompt_name
                
            concat_result = concat_and_summarize(**kwargs)
            
            if final_only:
                # Single result dict
                summaries = [concat_result["response"]]
                result = concat_result  # Use the single result for metadata
                summary_content_type = concat_result.get("summary_content_type", "--")
                step_k_inputs = concat_result.get("step_k_inputs", "--")
                prompts_used = concat_result.get("prompts_used", {})
                final_prompts_used = [concat_result.get("final_prompts_used", {})]  # Wrap in list for consistency
                template_vars = concat_result.get("template_vars", {})
                actual_prompt_name = concat_result.get("prompt_name", prompt_name)
            else:
                # List of result dicts (like independent strategy)
                summaries = [r["response"] for r in concat_result]
                # Aggregate metadata from list
                result = {
                    "processing_time": sum(r.get("processing_time", 0) for r in concat_result),
                    "usage": {
                        "total_cost": sum(r.get("usage", {}).get("total_cost", 0) for r in concat_result),
                        "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in concat_result)
                    }
                }
                
                # Extract summary_content_type, step_k_inputs, and prompts from first result (all should be the same)
                summary_content_type = concat_result[0].get("summary_content_type", "--") if concat_result else "--"
                step_k_inputs = concat_result[0].get("step_k_inputs", "--") if concat_result else "--"
                prompts_used = concat_result[0].get("prompts_used", {}) if concat_result else {}
                final_prompts_used = [r.get("final_prompts_used", {}) for r in concat_result] if concat_result else []
                template_vars = concat_result[0].get("template_vars", {}) if concat_result else {}
                actual_prompt_name = concat_result[0].get("prompt_name", prompt_name) if concat_result else prompt_name
            
        elif strategy == "summarize_chunks_independently":  
            # Only pass prompt_name if user specified one
            kwargs = {
                "chunks": chunks,
                "final_only": final_only,
                "model": model,
                "domain": domain,
                "optional_summary_length": optional_summary_length
            }
            if prompt_name is not None:
                kwargs["prompt_name"] = prompt_name
                
            chunk_results = summarize_chunks_independently(**kwargs)
            # Extract summaries from list of results (one per chunk)
            summaries = [r["response"] for r in chunk_results]
            # Combine metadata from all chunk results
            result = {
                "processing_time": sum(r.get("processing_time", 0) for r in chunk_results),
                "usage": {
                    "total_cost": sum(r.get("usage", {}).get("total_cost", 0) for r in chunk_results),
                    "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in chunk_results)
                }
            }
            
            # Extract summary_content_type, step_k_inputs, and prompts from first result (all should be the same)
            summary_content_type = chunk_results[0].get("summary_content_type", "--") if chunk_results else "--"
            step_k_inputs = chunk_results[0].get("step_k_inputs", "--") if chunk_results else "--"
            prompts_used = chunk_results[0].get("prompts_used", {}) if chunk_results else {}
            final_prompts_used = [r.get("final_prompts_used", {}) for r in chunk_results] if chunk_results else []
            template_vars = chunk_results[0].get("template_vars", {}) if chunk_results else {}
            actual_prompt_name = chunk_results[0].get("prompt_name", prompt_name) if chunk_results else prompt_name
            
        elif strategy == "iterative_summarize":
            # Only pass prompt_name if user specified one
            kwargs = {
                "chunks": chunks,
                "final_only": final_only,
                "model": model,
                "domain": domain,
                "optional_summary_length": optional_summary_length
            }
            if prompt_name is not None:
                kwargs["prompt_name"] = prompt_name
                
            iterative_results = iterative_summarize(**kwargs)
            # Extract summaries from list of results (one per step)
            summaries = [r["response"] for r in iterative_results]
            # Combine metadata from all iterative results
            result = {
                "processing_time": sum(r.get("processing_time", 0) for r in iterative_results),
                "usage": {
                    "total_cost": sum(r.get("usage", {}).get("total_cost", 0) for r in iterative_results),
                    "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in iterative_results)
                }
            }
            
            # Extract summary_content_type, step_k_inputs, and prompts from first result (all should be the same)
            summary_content_type = iterative_results[0].get("summary_content_type", "--") if iterative_results else "--"
            step_k_inputs = iterative_results[0].get("step_k_inputs", "--") if iterative_results else "--"
            prompts_used = iterative_results[0].get("prompts_used", {}) if iterative_results else {}
            final_prompts_used = [r.get("final_prompts_used", {}) for r in iterative_results] if iterative_results else []
            template_vars = iterative_results[0].get("template_vars", {}) if iterative_results else {}
            actual_prompt_name = iterative_results[0].get("prompt_name", prompt_name) if iterative_results else prompt_name
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # actual_prompt_name is now extracted within each strategy branch above
        
        # Separate collection-level vs item-level metadata
        collection_metadata = {
            "strategy_function": strategy,
            "summary_content_type": summary_content_type,
            "step_k_inputs": step_k_inputs,
            "model": model,
            "prompt_name": actual_prompt_name,  # Use actual prompt name from function
            "prompts_used": prompts_used,  # Template prompts (collection-level)
            "final_only": final_only,
            "command_run": command_run  # Full command for reproducibility
        }
        
        item_metadata = {
            "strategy_function": strategy,  # For item-level completeness
            "summary_content_type": summary_content_type,   # For item-level completeness
            "step_k_inputs": step_k_inputs,  # For item-level completeness
            "final_prompts_used": final_prompts_used,  # Item-specific (contains actual text)
            "template_vars": _truncate_template_vars(template_vars),  # Item-specific (contains actual text)
            "processing_time": result.get("processing_time", 0),
            "cost": result.get("usage", {}).get("total_cost", 0),
            "total_tokens": result.get("usage", {}).get("total_tokens", 0),
            "command_run": command_run  # Full command for reproducibility
        }
        
        # Save using the existing save_summaries function
        output_dir = f"outputs/summaries/{output_name}"
        save_summaries(
            item_id=current_item_id,
            summaries=summaries,
            original_item_data=item_data,
            output_dir=output_dir,
            collection_metadata=collection_metadata,
            item_metadata=item_metadata
        )
        
        # Track results
        results[current_item_id] = {
            "summaries": summaries,
            "collection_metadata": collection_metadata,
            "item_metadata": item_metadata
        }
        
        # Accumulate costs
        usage = result.get("usage", {})
        total_cost += usage.get("total_cost", 0)
        total_tokens += usage.get("total_tokens", 0)
        
        print(f"✅ Completed {current_item_id}")
        if preview and summaries:
            preview_summary = summaries[0][:300] + "..." if len(summaries[0]) > 300 else summaries[0]
            print(f"📄 Summary preview: {preview_summary}")
    
    elapsed_time = time.time() - start_time
    
    # Count processed vs skipped items
    processed_items = [item_id for item_id, result in results.items() if not result.get("skipped", False)]
    skipped_items = [item_id for item_id, result in results.items() if result.get("skipped", False)]
    
    print(f"\n🎉 Summarization completed!")
    print(f"✅ Processed: {len(processed_items)} items")
    if skipped_items:
        print(f"⏭️  Skipped: {len(skipped_items)} items (already existed)")
    print(f"⏱️  Total time: {elapsed_time:.2f}s")  
    print(f"💰 Total cost: ${total_cost:.6f}")
    print(f"🔢 Total tokens: {total_tokens:,}")
    print(f"📁 Results saved to: outputs/summaries/{output_name}")
    
    return {
        "items_processed": list(results.keys()),
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "processing_time": elapsed_time,
        "output_dir": f"outputs/summaries/{output_name}",
        "results": results
    }


def _list_strategies() -> None:
    """List available summarization strategies and exit."""
    print("Available summarization strategies:")
    print()
    print("📋 concat_and_summarize")
    print("   • Concatenates all chunks into a single text")
    print("   • Produces cumulative summaries (final summary of all content)")
    print("   • Use --intermediate flag to get progressive summaries")
    print("   • Best for: Getting overall summary of entire document")
    print()
    print("📋 summarize_chunks_independently") 
    print("   • Summarizes each chunk separately")
    print("   • Produces chunk summaries (one summary per chunk)")
    print("   • Maintains chunk-level granularity")
    print("   • Best for: Analyzing content at chunk level")
    print()
    print("📋 iterative_summarize") 
    print("   • Builds summaries incrementally, using previous summary as context")
    print("   • First chunk gets initial summary, then each subsequent summary")
    print("     incorporates previous summary + new chunk")
    print("   • Produces incremental summaries (n summaries building on each other)")
    print("   • Best for: True incremental summarization with context continuity")
    print()
    print("Note: Use --strategy <name> to specify which strategy to use")
    print("Default strategy: concat_and_summarize")


def _validate_input_path(path: str) -> None:
    """Validate that the input path exists and is accessible."""
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise ValidationError(f"Input path does not exist: {path}")
    
    if path_obj.is_file():
        if path_obj.suffix != '.json':
            raise ValidationError(f"Input file must be a JSON file: {path}")
        
        # Check if it's in the expected items/ structure
        if path_obj.parent.name != 'items':
            raise ValidationError(f"Item JSON files must be in an 'items/' directory: {path}")
    
    elif path_obj.is_dir():
        # Check for collection.json
        collection_file = path_obj / "collection.json"
        if not collection_file.exists():
            raise ValidationError(f"Directory must contain collection.json: {path}")


def main():
    """Main entry point for summarization CLI."""
    parser = argparse.ArgumentParser(
        description="Summarize chunked text using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize all items with cumulative strategy (default)
  python -m ius summarize --input outputs/chunks/ipython_test
  
  # Summarize each chunk independently  
  python -m ius summarize --input outputs/chunks/ipython_test --strategy summarize_chunks_independently
  
  # Summarize a specific item with auto-generated output name
  python -m ius summarize --input outputs/chunks/ipython_test --item ADP02
  
  # Summarize with custom output name and strategy
  python -m ius summarize --input outputs/chunks/ipython_test --strategy summarize_chunks_independently --output my_experiment
  
  # Summarize a single item file directly
  python -m ius summarize --input outputs/chunks/ipython_test/items/ADP02.json
  
  # Use different model, strategy, and prompt with intermediate summaries
  python -m ius summarize --input outputs/chunks/ipython_test --strategy concat_and_summarize --model gpt-4 --prompt custom-prompt --intermediate
  
  # Overwrite existing results instead of skipping them  
  python -m ius summarize --input outputs/chunks/ipython_test --item ADP02 --overwrite
  
  # List available summarization strategies
  python -m ius summarize --list-strategies
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        help="Path to chunked dataset directory or specific item JSON file"
    )
    
    parser.add_argument(
        "--output", "-o", 
        help="Name for the experiment output directory (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--item",
        help="Specific item ID to process (optional, processes all items if not specified)"
    )
    
    parser.add_argument(
        "--strategy", "-s",
        default="concat_and_summarize",
        choices=["concat_and_summarize", "summarize_chunks_independently", "iterative_summarize"],
        help="Summarization strategy (default: concat_and_summarize)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gpt-4.1-mini",
        help="LLM model to use (default: gpt-4.1-mini)"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        default=None,  # Let CLI choose based on strategy
        help="Prompt template name to use (default: strategy-specific)"
    )
    
    parser.add_argument(
        "--summary-length",
        default="summary",
        help="Optional summary length specification (e.g., 'brief summary', 'detailed summary', 'one-paragraph summary', 'summary in less than 100 words') (default: summary)"
    )
    
    parser.add_argument(
        "--intermediate",
        action="store_true",
        help="Generate intermediate summaries for each chunk (default: final summary only)"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show previews of chunks and summaries"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing item results (default: skip existing items)"
    )
    
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List available summarization strategies and exit"
    )
    
    args = parser.parse_args()
    
    # Handle --list-strategies flag early
    if args.list_strategies:
        _list_strategies()
        return
    
    # Validate required arguments when not listing strategies
    if not args.input:
        parser.error("--input/-i is required (unless using --list-strategies)")
        
    # Set up logging with specified level
    setup_logging(log_level=args.log_level)
    
    try:
        # Validate input path
        _validate_input_path(args.input)
        
        # Run summarization
        kwargs = {
            "input_path": args.input,
            "output_name": args.output,
            "item_id": args.item,
            "strategy": args.strategy,
            "model": args.model,
            "final_only": not args.intermediate,
            "preview": args.preview,
            "overwrite": args.overwrite,
            "optional_summary_length": args.summary_length
        }
        
        # Only pass prompt_name if user specified it, let function defaults handle None
        if args.prompt is not None:
            kwargs["prompt_name"] = args.prompt
            
        results = summarize_chunks(**kwargs)
        
        logger.info(f"Summarization completed successfully: {len(results['items_processed'])} items processed")
        
    except (ValidationError, DatasetError) as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during summarization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()