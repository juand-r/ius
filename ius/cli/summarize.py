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
from ius.summarization import concat_and_summarize, save_summaries


# Set up logger for this module
logger = get_logger(__name__)


def summarize_chunks(
    input_path: str,
    output_name: str,
    item_id: str | None = None,
    model: str = "gpt-4.1-mini",
    prompt_name: str = "default-concat-prompt",
    final_only: bool = True,
    preview: bool = False,
) -> Dict[str, Any]:
    """
    CLI wrapper for summarizing chunks with progress printing and file I/O.

    Args:
        input_path: Path to chunked dataset directory or specific item JSON
        output_name: Name for the experiment output directory  
        item_id: Specific item ID to process (optional, processes all if None)
        model: LLM model to use for summarization
        prompt_name: Name of prompt template to use
        final_only: Whether to return only final summary or intermediate ones too
        preview: Whether to show summary previews

    Returns:
        Dictionary with summarization results and metadata
    """
    start_time = time.time()
    
    print(f"🤖 Starting summarization...")
    print(f"📥 Input: {input_path}")
    print(f"📤 Output: outputs/summaries/{output_name}")
    print(f"🧠 Model: {model}")
    print(f"📝 Prompt: {prompt_name}")
    
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
    
    results = {}
    total_cost = 0.0
    total_tokens = 0
    
    for current_item_id in items_to_process:
        print(f"\n🔄 Processing {current_item_id}...")
        
        # Load chunks for this item
        item_data = chunked_dataset.load_item(current_item_id)
        chunks = []
        for doc in item_data["documents"]:
            chunks.extend(doc["chunks"])  # Use "chunks" (plural) not "chunk"
        
        print(f"📦 Loaded {len(chunks)} chunks ({sum(len(c) for c in chunks):,} chars)")
        
        if preview and chunks:
            preview_text = chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0]
            print(f"👀 First chunk preview: {preview_text}")
        
        # Run summarization
        result = concat_and_summarize(
            chunks=chunks,
            final_only=final_only,
            prompt_name=prompt_name,
            model=model
        )
        
        # Extract summaries
        if final_only:
            summaries = [result["response"]]
        else:
            summaries = result.get("intermediate_summaries", [result["response"]])
        
        # Save results
        experiment_metadata = {
            "model": model,
            "prompt_name": prompt_name,
            "final_only": final_only,
            "processing_time": result.get("processing_time", 0),
            "cost": result.get("usage", {}).get("total_cost", 0),
            "total_tokens": result.get("usage", {}).get("total_tokens", 0)
        }
        
        # Save using the existing save_summaries function
        output_dir = f"outputs/summaries/{output_name}"
        save_summaries(
            item_id=current_item_id,
            summaries=summaries,
            original_item_data=item_data,
            output_dir=output_dir,
            experiment_metadata=experiment_metadata
        )
        
        # Track results
        results[current_item_id] = {
            "summaries": summaries,
            "metadata": experiment_metadata
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
    
    print(f"\n🎉 Summarization completed!")
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
  # Summarize all items in a chunked dataset
  python -m ius summarize --input outputs/chunks/ipython_test --output my_experiment
  
  # Summarize a specific item
  python -m ius summarize --input outputs/chunks/ipython_test --item ADP02 --output test_run
  
  # Summarize a single item file directly
  python -m ius summarize --input outputs/chunks/ipython_test/items/ADP02.json --output single_test
  
  # Use different model and prompt
  python -m ius summarize --input outputs/chunks/ipython_test --output gpt4_test --model gpt-4 --prompt custom-prompt
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to chunked dataset directory or specific item JSON file"
    )
    
    parser.add_argument(
        "--output", "-o", 
        required=True,
        help="Name for the experiment output directory (will be saved to outputs/summaries/<name>)"
    )
    
    parser.add_argument(
        "--item",
        help="Specific item ID to process (optional, processes all items if not specified)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gpt-4.1-mini",
        help="LLM model to use (default: gpt-4.1-mini)"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        default="default-concat-prompt", 
        help="Prompt template name to use (default: default-concat-prompt)"
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
    
    args = parser.parse_args()
    
    # Set up logging with specified level
    setup_logging(log_level=args.log_level)
    
    try:
        # Validate input path
        _validate_input_path(args.input)
        
        # Run summarization
        results = summarize_chunks(
            input_path=args.input,
            output_name=args.output,
            item_id=args.item,
            model=args.model,
            prompt_name=args.prompt,
            final_only=not args.intermediate,
            preview=args.preview
        )
        
        logger.info(f"Summarization completed successfully: {len(results['items_processed'])} items processed")
        
    except (ValidationError, DatasetError) as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during summarization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()