#!/usr/bin/env python3
"""
CLI for evaluating faithfulness of extracted claims using the faithfulness evaluator.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from faithfulness_evaluator import FaithfulnessEvaluator
from faithfulness_evaluator.data.loaders import GenericDataLoader

from ius.exceptions import ValidationError
from ius.logging_config import get_logger, setup_logging

# Set up logger for this module
logger = get_logger(__name__)


def extract_dataset_name(input_path: str) -> str:
    """Extract dataset name from input path."""
    input_path = input_path.rstrip('/')
    last_dir = os.path.basename(input_path)
    dataset_name = last_dir.split("_")[0]
    return dataset_name


def get_dataset_source_path(dataset_name: str) -> Path:
    """Get source dataset path for a dataset."""
    dataset_mappings = {
        "bmds": "datasets/bmds",
        "true-detective": "datasets/true-detective", 
        "squality": "datasets/squality",
        "detectiveqa": "datasets/detectiveqa"
    }
    
    if dataset_name not in dataset_mappings:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(dataset_mappings.keys())}")
    
    return Path(dataset_mappings[dataset_name])


def generate_output_hash(hash_parameters: Dict[str, Any]) -> str:
    """Generate hash for output directory name."""
    param_str = json.dumps(hash_parameters, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:6]


def load_claim_extraction_results(claims_path: str, item_id: Optional[str] = None) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Load claim extraction results from the claims output directory.
    
    Args:
        claims_path: Path to the claim extraction output directory
        item_id: If provided, only process this specific item
    
    Returns:
        Tuple of (item_data, collection_metadata) where:
        - item_data: Dict mapping item_id to list of summary claim results
        - collection_metadata: Collection metadata including summarization info
    """
    claims_path_obj = Path(claims_path)
    items_dir = claims_path_obj / "items"
    collection_file = claims_path_obj / "collection.json"
    
    if not items_dir.exists():
        raise ValueError(f"Claims items directory not found: {items_dir}")
    
    if not collection_file.exists():
        raise ValueError(f"Claims collection metadata not found: {collection_file}")
    
    logger.info(f"Loading claim extraction results from: {claims_path}")
    
    # Load collection metadata
    with open(collection_file) as f:
        collection_metadata = json.load(f)
    
    data = {}
    
    # Iterate through item directories
    for item_dir in items_dir.iterdir():
        if not item_dir.is_dir():
            continue
            
        current_item_id = item_dir.name
        
        # Filter to specific item if requested
        if item_id and item_id != current_item_id:
            continue
        
        # Load all summary claim files for this item
        summary_claims = []
        claim_files = sorted(item_dir.glob("*.json"), key=lambda x: int(x.stem))
        
        for claim_file in claim_files:
            with open(claim_file) as f:
                claim_data = json.load(f)
            summary_claims.append(claim_data)
        
        if summary_claims:
            data[current_item_id] = summary_claims
            logger.debug(f"Loaded {len(summary_claims)} summary claim sets for {current_item_id}")
    
    logger.info(f"Loaded claim extraction results for {len(data)} items")
    return data, collection_metadata


def load_source_documents(dataset_name: str, item_ids: List[str]) -> Dict[str, str]:
    """
    Load source documents for the given items.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'bmds', 'true-detective')
        item_ids: List of item IDs to load
    
    Returns:
        Dict mapping item_id to source document content
    """
    source_path = get_dataset_source_path(dataset_name)
    
    logger.info(f"Loading source documents from: {source_path}")
    
    source_docs = {}
    
    for item_id in item_ids:
        source_file = source_path / "items" / f"{item_id}.json"
        
        if not source_file.exists():
            logger.warning(f"Source file not found for {item_id}: {source_file}")
            continue
            
        with open(source_file) as f:
            source_data = json.load(f)
            
        source_docs_list = source_data.get("documents", [])
        if not source_docs_list:
            logger.warning(f"No documents found in source file: {source_file}")
            continue
            
        source_content = source_docs_list[0].get("content", "")
        if not source_content:
            logger.warning(f"No content found in source file: {source_file}")
            continue
        
        source_docs[item_id] = source_content
        logger.debug(f"Loaded source document for {item_id} (length: {len(source_content)} chars)")
    
    logger.info(f"Loaded source documents for {len(source_docs)} items")
    return source_docs


async def evaluate_claims_faithfulness(source_text: str, claims: List[str], method: str = "bm25", model: str = "gpt-5-mini") -> List[Dict[str, Any]]:
    """
    Evaluate faithfulness of claims against source text.
    
    Args:
        source_text: The original document text
        claims: List of claims to evaluate
        method: Evaluation method ("bm25" or "full_text")
        model: Model to use for faithfulness evaluation
    
    Returns:
        List of evaluation results for each claim
    """
    if not claims:
        return []
    
    # Create evaluator
    evaluator = FaithfulnessEvaluator(llm_model=model, method=method)
    
    # Convert claims to proper format
    loader = GenericDataLoader()
    claims_data = loader.create_claims_from_list(claims, "document")
    
    # Load source text and claims
    evaluator.load_document_and_claims(source_text, claims_data)
    
    # Evaluate all claims
    results = await evaluator.evaluate_all_claims()
    
    # Convert results to serializable format
    formatted_results = []
    for i, result in enumerate(results):
        formatted_results.append({
            "claim_index": i,
            "claim_text": claims[i],
            "predicted_label": result.predicted_label,
            "reasoning": result.reasoning,
            "faithful": result.predicted_label.lower() in ["yes", "true", "faithful"]
        })
    
    return formatted_results


def calculate_summary_faithfulness_score(claim_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate overall faithfulness score for a summary.
    
    Args:
        claim_results: List of claim evaluation results
    
    Returns:
        Dict with faithfulness statistics
    """
    if not claim_results:
        return {
            "total_claims": 0,
            "faithful_claims": 0,
            "faithfulness_score": 0.0
        }
    
    total_claims = len(claim_results)
    faithful_claims = sum(1 for result in claim_results if result["faithful"])
    faithfulness_score = faithful_claims / total_claims if total_claims > 0 else 0.0
    
    return {
        "total_claims": total_claims,
        "faithful_claims": faithful_claims,
        "faithfulness_score": faithfulness_score
    }


def save_individual_summary_result(
    item_id: str,
    summary_result: Dict[str, Any],
    original_claim_data: Dict[str, Any],
    method: str,
    output_path: Path,
    summary_index: int
) -> None:
    """Save individual summary result immediately after processing."""
    items_dir = output_path / "items"
    items_dir.mkdir(exist_ok=True)
    
    # Create item subdirectory
    item_dir = items_dir / item_id
    item_dir.mkdir(exist_ok=True)
    
    summary_file_result = {
        "item_id": item_id,
        "faithfulness_results": [summary_result],  # Keep as list for consistency
        "parameters": {
            "method": method,
            "implementation": "faithfulness-evaluator"
        },
        "original_data": {
            "summary_text": original_claim_data.get("summary_text", ""),
            "domain": original_claim_data.get("domain", ""),
            "summary_index": summary_result.get("summary_index", 0)
        }
    }
    
    with open(item_dir / f"{summary_index}.json", 'w') as f:
        json.dump(summary_file_result, f, indent=2)
    
    logger.debug(f"Saved summary result to {item_id}/{summary_index}.json")


def save_faithfulness_results(
    results: Dict[str, List[Dict[str, Any]]],
    claim_data: Dict[str, List[Dict[str, Any]]],
    summary_collection_metadata: Dict[str, Any],
    output_path: Path,
    input_path: str,
    method: str,
    command_run: str
) -> None:
    """Save faithfulness evaluation results to files."""
    
    # Calculate statistics
    total_items = len(results)
    successful_items = sum(1 for item_results in results.values() if item_results)
    failed_items = total_items - successful_items
    
    total_summaries = sum(len(item_results) for item_results in results.values())
    total_claims = sum(
        sum(len(summary_result["claim_evaluations"]) for summary_result in item_results)
        for item_results in results.values()
    )
    
    all_scores = []
    for item_results in results.values():
        for summary_result in item_results:
            all_scores.append(summary_result["faithfulness_statistics"]["faithfulness_score"])
    
    mean_faithfulness = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    # Extract dataset name
    dataset_name = extract_dataset_name(input_path)
    
    # Extract summarization info from summary collection metadata
    summarization_info = summary_collection_metadata.get("summarization_info", {}).get("collection_metadata", {})
    
    # Save collection-level results
    collection_data = {
        "experiment_id": output_path.name,
        "created_at": datetime.now().isoformat(),
        "input_path": input_path,
        "dataset_name": dataset_name,
        "faithfulness_evaluation_info": {
            "collection_metadata": {
                "evaluation_function": "evaluate_faithfulness",
                "content_type": "faithfulness_scores",
                "input_type": "claims",
                "method": method,
                "source_collection": input_path,
                "implementation": "faithfulness-evaluator"
            },
            "processing_stats": {
                "total_items": total_items,
                "successful_items": successful_items,
                "failed_items": failed_items,
                "total_summaries": total_summaries,
                "total_claims": total_claims
            },
            "parameters": {
                "method": method,
                "implementation": "faithfulness-evaluator"
            },
            "command_run": command_run,
            "faithfulness_statistics": {
                "mean_faithfulness_score": mean_faithfulness,
                "total_claims_evaluated": total_claims
            },
            "summarization_info": summarization_info
        },
        "statistics": {
            "total_items": total_items,
            "successful_items": successful_items,
            "failed_items": failed_items,
            "mean_score": mean_faithfulness
        }
    }
    
    with open(output_path / "collection.json", 'w') as f:
        json.dump(collection_data, f, indent=2)
    
    # Individual results are now saved incrementally during processing
    logger.info(f"Collection metadata saved to {output_path / 'collection.json'}")


async def evaluate_dataset(
    input_path: str,
    method: str = "full_text",
    model: str = "gpt-5-mini",
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = False,
    stop: Optional[int] = None,
    item_id: Optional[str] = None,
    range_spec: str = "last",
) -> Dict[str, Any]:
    """
    Evaluate faithfulness of claims in a claim extraction dataset.
    
    Args:
        input_path: Path to the claim extraction output directory
        method: Faithfulness evaluation method ("bm25" or "full_text")
        model: Model to use for faithfulness evaluation (default: "gpt-5-mini")
        output_dir: Output directory name (auto-generated if not provided)
        overwrite: Whether to overwrite existing results
        verbose: Enable verbose logging
        stop: Stop after processing this many items
        item_id: Process only this specific item
        range_spec: Which summary chunks to process ("all" or "last")
    
    Returns:
        Dictionary containing processing results and statistics
    """
    if verbose:
        setup_logging(log_level="DEBUG")
    
    logger.info(f"Starting faithfulness evaluation for: {input_path}")
    logger.info(f"Method: {method}")
    logger.info(f"Model: {model}")
    
    start_time = time.time()
    
    try:
        # Generate output directory name if not provided
        if output_dir is None:
            hash_parameters = {
                "method": method,
                "model": model,
                "implementation": "faithfulness-evaluator"
            }
            
            hash_value = generate_output_hash(hash_parameters)
            input_basename = Path(input_path).name
            output_dir = f"{input_basename}_faithfulness_{hash_value}"
        
        # Create output directory
        output_path = Path("outputs") / "eval" / "intrinsic" / "faithfulness" / output_dir
        
        if output_path.exists() and not overwrite:
            logger.info(f"Output directory already exists: {output_path}")
            logger.info("Use --overwrite to regenerate results")
            
            # Load existing results
            collection_file = output_path / "collection.json"
            if collection_file.exists():
                with open(collection_file, 'r') as f:
                    existing_results = json.load(f)
                return {
                    "output_path": str(output_path), 
                    "duration": 0,
                    "statistics": existing_results.get("statistics", {})
                }
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load claim extraction results and metadata
        claim_data, claims_collection_metadata = load_claim_extraction_results(input_path, item_id)
        
        # Extract the source summary collection path from claims metadata
        source_collection_path = claims_collection_metadata.get("claim_extraction_info", {}).get("collection_metadata", {}).get("source_collection")
        if not source_collection_path:
            raise ValueError("Cannot find source summary collection path in claims metadata")
        
        # Load summarization metadata from the original summary collection
        summary_collection_file = Path(source_collection_path) / "collection.json"
        if not summary_collection_file.exists():
            raise ValueError(f"Summary collection metadata not found: {summary_collection_file}")
        
        with open(summary_collection_file) as f:
            summary_collection_metadata = json.load(f)
        
        if not claim_data:
            raise ValueError("No claim extraction results found")
        
        # Apply stop limit if specified
        if stop and stop < len(claim_data):
            logger.info(f"Limiting evaluation to first {stop} items")
            claim_data = dict(list(claim_data.items())[:stop])
        
        # Extract dataset name from the claims path
        # Claims path format: .../{summary_collection_name}_claims_{hash}
        # Summary collection name format: {dataset_name}_{other_params}
        claims_dir_name = Path(input_path).name
        if "_claims_" in claims_dir_name:
            summary_collection_name = claims_dir_name.split("_claims_")[0]
            # Extract just the dataset name (first part before underscore)
            dataset_name = summary_collection_name.split("_")[0]
        else:
            raise ValueError(f"Cannot extract dataset name from claims path: {input_path}")
        
        # Load source documents
        item_ids = list(claim_data.keys())
        source_docs = load_source_documents(dataset_name, item_ids)
        
        # Evaluate faithfulness for each item
        results = {}
        
        for item_id, summary_claims_list in claim_data.items():
            if item_id not in source_docs:
                logger.warning(f"No source document found for {item_id}, skipping")
                results[item_id] = []
                continue
            
            source_text = source_docs[item_id]
            item_results = []
            
            logger.info(f"Evaluating faithfulness for {item_id} ({len(summary_claims_list)} summaries)")

            # Filter summaries based on range_spec
            if range_spec == "last":
                summary_claims_list = [summary_claims_list[-1]] if summary_claims_list else []
            # For "all", use the full list (no filtering needed)
            
            for summary_claims in summary_claims_list:
                claims = summary_claims.get("claims", [])
                summary_index = summary_claims.get("summary_index", 0)
                
                if not claims:
                    logger.warning(f"No claims found for {item_id} summary {summary_index}")
                    continue
                
                # Evaluate claims faithfulness
                claim_evaluations = await evaluate_claims_faithfulness(source_text, claims, method, model)
                
                # Calculate summary-level statistics
                faithfulness_stats = calculate_summary_faithfulness_score(claim_evaluations)
                
                summary_result = {
                    "summary_index": summary_index,
                    "claim_evaluations": claim_evaluations,
                    "faithfulness_statistics": faithfulness_stats,
                    "original_data": {
                        "summary_text": summary_claims.get("summary_text", ""),
                        "domain": summary_claims.get("domain", ""),
                        "total_claims": len(claims)
                    }
                }
                
                # Save individual result immediately
                save_individual_summary_result(
                    item_id=item_id,
                    summary_result=summary_result,
                    original_claim_data=summary_claims,
                    method=method,
                    output_path=output_path,
                    summary_index=summary_index
                )
                
                item_results.append(summary_result)
                
                logger.info(f"  Summary {summary_index}: {faithfulness_stats['faithful_claims']}/{faithfulness_stats['total_claims']} faithful claims ({faithfulness_stats['faithfulness_score']:.3f}) -> Saved as {item_id}/{summary_index}.json")
            
            results[item_id] = item_results
        
        # Capture the command that was run
        argv_copy = sys.argv.copy()
        if argv_copy[0].endswith("__main__.py"):
            argv_copy[0] = "python -m ius"
        command_run = " ".join(argv_copy)
        
        # Save results
        save_faithfulness_results(results, claim_data, summary_collection_metadata, output_path, input_path, method, command_run)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Faithfulness evaluation completed in {duration:.2f} seconds")
        logger.info(f"Results saved to: {output_path}")
        
        # Load and return results summary
        collection_file = output_path / "collection.json"
        with open(collection_file, 'r') as f:
            results_summary = json.load(f)
            
        # Extract key statistics
        stats = results_summary.get("statistics", {})
        
        summary = {
            "output_path": str(output_path),
            "duration": duration,
            "total_items": stats.get("total_items", 0),
            "successful_items": stats.get("successful_items", 0),
            "failed_items": stats.get("failed_items", 0),
            "mean_faithfulness": stats.get("mean_score", 0.0),
            "statistics": stats
        }
        
        logger.info(f"Summary - Processed: {summary['successful_items']}/{summary['total_items']} items")
        logger.info(f"Failed: {summary['failed_items']}")
        logger.info(f"Mean faithfulness score: {summary['mean_faithfulness']:.4f}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Faithfulness evaluation failed: {e}")
        raise


def main():
    """Main entry point for faithfulness evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate faithfulness of extracted claims using faithfulness evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate faithfulness of claims in a claim extraction output
  python -m ius.cli.faithfulness --input outputs/summaries-claims/bmds_claims_default-claim-extraction
  
  # Use bm25 method with specific model
  python -m ius.cli.faithfulness --input outputs/summaries-claims/bmds_claims_default-claim-extraction --method bm25 --model gpt-4o-mini
  
  # Process only specific item
  python -m ius.cli.faithfulness --input outputs/summaries-claims/bmds_claims_default-claim-extraction --item-id ADP02

  # Limit to first 10 items for testing
  python -m ius.cli.faithfulness --input outputs/summaries-claims/bmds_claims_default-claim-extraction --stop 10
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the claim extraction output directory"
    )
    
    parser.add_argument(
        "--method", "-m",
        choices=["bm25", "full_text"],
        default="full_text",
        help="Faithfulness evaluation method (default: full_text)"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Model to use for faithfulness evaluation (default: gpt-5-mini)"
    )
    
    parser.add_argument(
        "--range",
        default="last",
        help="Which summary chunks to process: 'all' for all chunks, 'last' for only the last chunk (default: all)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output directory name (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    
    parser.add_argument(
        "--stop",
        type=int,
        help="Stop after processing this many items"
    )
    
    parser.add_argument(
        "--item-id",
        help="Process only this specific item"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.input).exists():
        parser.error(f"Input path does not exist: {args.input}")
    
    # Execute faithfulness evaluation
    try:
        asyncio.run(evaluate_dataset(
            input_path=args.input,
            method=args.method,
            model=args.model,
            output_dir=args.output,
            overwrite=args.overwrite,
            verbose=args.verbose,
            stop=args.stop,
            item_id=args.item_id,
            range_spec=args.range,
        ))
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()