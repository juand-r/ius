"""
Claim extraction module for the IUS system.

This module provides LLM-based claim extraction capabilities with comprehensive
cost tracking and multiple extraction strategies.

Main Functions:
- extract_claims_from_summaries: Extract claims from summary data
- process_dataset_summaries: Process entire datasets of summaries for claim extraction
- save_claims: Save extracted claims to output files

Usage Example:
    from ius.claim_extract import extract_claims_from_summaries
    
    claims_result = extract_claims_from_summaries(
        summaries=["summary1", "summary2"],
        model="gpt-4o-mini",
        prompt_name="default-claim-extraction"
    )
"""

import json
import re
import time
from typing import Any, Dict, List
from pathlib import Path
import hashlib

from ius.logging_config import get_logger
from ius.utils import call_llm
from ius.exceptions import ClaimExtractionError, ValidationError

logger = get_logger(__name__)


def extract_claims_from_summaries(
    summaries: list[str],
    model: str = "gpt-4o-mini",
    prompt_name: str = "default-claim-extraction",
    ask_user_confirmation: bool = False,
    domain: str = "text",
    **kwargs
) -> list[dict[str, Any]]:
    """
    Extract claims from a list of summaries.
    
    Args:
        summaries: List of summary texts to extract claims from
        model: LLM model to use for claim extraction
        prompt_name: Name of the prompt to use
        ask_user_confirmation: Whether to ask user confirmation before API call
        domain: Domain context for the summaries
        **kwargs: Additional parameters for LLM call
    
    Returns:
        List of dictionaries containing extracted claims and metadata for each summary
        
    Raises:
        ClaimExtractionError: If extraction parameters are invalid
    """
    if not summaries:
        return []
    
    # Load prompt templates
    prompt_dir = Path(f"prompts/claim_extraction/{prompt_name}")
    if not prompt_dir.exists():
        raise ClaimExtractionError(f"Prompt directory not found: {prompt_dir}")
    
    prompts = {}
    for prompt_file in prompt_dir.glob("*.txt"):
        key = prompt_file.stem  # filename without .txt extension
        prompts[key] = prompt_file.read_text()
    
    # For backward compatibility, create system_and_user_prompt dict
    system_and_user_prompt = {
        "system": prompts.get("system", ""),
        "user": prompts.get("user", "")
    }
    
    results = []
    for i, summary in enumerate(summaries):
        logger.info(f"Extracting claims from summary {i+1}/{len(summaries)}")
        
        template_vars = {
            "text": summary,
            "domain": domain
        }
        
        result = call_llm(
            summary, 
            model, 
            system_and_user_prompt, 
            template_vars=template_vars,
            ask_user_confirmation=ask_user_confirmation,
            **kwargs
        )
        
        # Add claim extraction specific metadata
        result["method"] = "extract_claims_from_summaries"
        result["summary_index"] = i
        result["prompt_name"] = prompt_name
        result["prompts_used"] = prompts
        result["template_vars"] = template_vars
        result["content_type"] = "claims"
        result["input_type"] = "summary"
        
        results.append(result)
    
    return results


def process_dataset_summaries(
    summary_collection_path: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    prompt_name: str = "default-claim-extraction",
    ask_user_confirmation: bool = False,
    scope: str = "all",
    item_ids: list[str] | None = None,
    overwrite: bool = False,
    stop: int | None = None,
    **kwargs
) -> dict[str, Any]:
    """
    Process a dataset of summaries to extract claims from each item.
    
    Args:
        summary_collection_path: Path to the summary collection directory
        output_path: Path to save claim extraction results
        model: LLM model to use
        prompt_name: Name of the prompt directory to use
        ask_user_confirmation: Whether to ask for confirmation before API calls
        scope: Processing scope ("all" or "item")
        item_ids: List of specific item IDs to process (if scope is "item")
        overwrite: Whether to overwrite existing output files
        stop: Stop after processing this many items
        **kwargs: Additional parameters for LLM calls
    
    Returns:
        Dictionary containing processing results and metadata
        
    Raises:
        ClaimExtractionError: If input parameters are invalid
        ValidationError: If summary data is malformed
    """
    summary_path = Path(summary_collection_path)
    if not summary_path.exists():
        raise ClaimExtractionError(f"Summary collection path not found: {summary_path}")
    
    # Load collection metadata
    collection_file = summary_path / "collection.json"
    if not collection_file.exists():
        raise ClaimExtractionError(f"Collection file not found: {collection_file}")
    
    with open(collection_file, 'r') as f:
        collection_data = json.load(f)
    
    # Determine which items to process
    if scope == "item" and item_ids:
        items_to_process = item_ids
    else:
        items_to_process = collection_data.get("summarization_info", {}).get("items_processed", [])
    
    if not items_to_process:
        raise ClaimExtractionError("No items found to process")
    
    logger.info(f"Processing {len(items_to_process)} items for claim extraction")
    
    # Process each item
    results = {}
    errors = {}
    total_cost = 0.0
    total_tokens = 0
    processed_count = 0
    
    for item_id in items_to_process:
        try:
            logger.info(f"Processing item: {item_id}")
            
            # Load summary data for this item
            item_file = summary_path / "items" / f"{item_id}.json"
            if not item_file.exists():
                logger.warning(f"Summary file not found for item {item_id}: {item_file}")
                continue
            
            with open(item_file, 'r') as f:
                item_data = json.load(f)
            
            # Extract summaries from the item data
            summaries = []
            for document in item_data.get("documents", []):
                document_summaries = document.get("summaries", [])
                summaries.extend(document_summaries)
            
            if not summaries:
                logger.warning(f"No summaries found for item {item_id}")
                continue
            
            # Extract domain and summarization metadata if not provided
            try:
                # Get metadata from documents[0]["metadata"]["item_experiment_metadata"]
                first_doc = item_data.get("documents", [{}])[0]
                metadata = first_doc.get("metadata", {})
                item_exp_metadata = metadata.get("item_experiment_metadata", {})
                template_vars = item_exp_metadata.get("template_vars", {})
                
                # Extract domain
                extracted_domain = template_vars.get("domain", "story")
                logger.info(f"Extracted domain '{extracted_domain}' from item {item_id} metadata")
                
                # Extract summarization info for collection metadata (first item only)
                if item_id == items_to_process[0]:  # Only extract from first item
                    summarization_info = {
                        "optional_summary_length": template_vars.get("optional_summary_length"),
                        "strategy_function": item_exp_metadata.get("strategy_function"),
                        "summary_content_type": item_exp_metadata.get("summary_content_type"),
                        "step_k_inputs": item_exp_metadata.get("step_k_inputs")
                    }
                    # Store for later use in collection metadata
                    globals()['_summarization_info'] = summarization_info
                    logger.info(f"Extracted summarization info: {summarization_info}")
                    
            except (IndexError, KeyError, TypeError) as e:
                raise ValueError("No domain found.")
            
            # Create item directory for saving individual summary results
            item_output_dir = Path(output_path) / "items" / item_id
            item_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each summary individually and save immediately
            claim_results = []
            for summary_idx, summary in enumerate(summaries, 1):
                # Check if this summary has already been processed
                summary_output_file = item_output_dir / f"{summary_idx}.json"
                if summary_output_file.exists() and not overwrite:
                    logger.debug(f"Skipping {item_id} summary {summary_idx} (already exists)")
                    continue
                
                # Extract claims from this single summary
                single_summary_results = extract_claims_from_summaries(
                    summaries=[summary],  # Process one summary at a time
                    model=model,
                    prompt_name=prompt_name,
                    ask_user_confirmation=ask_user_confirmation,
                    domain=extracted_domain,
                    **kwargs
                )
                
                # Parse the claims from the LLM response
                if single_summary_results:
                    raw_response = single_summary_results[0].get("response", "")
                    parsed_claims = _parse_claims_from_response(raw_response)
                    llm_result = single_summary_results[0]  # Keep full LLM result for debugging
                else:
                    parsed_claims = []
                    llm_result = {}
                
                # Save this summary's claims immediately with relevant metadata
                summary_output_file = item_output_dir / f"{summary_idx}.json"
                with open(summary_output_file, 'w') as f:
                    json.dump({
                        "item_id": item_id,
                        "summary_index": summary_idx,
                        "claims": parsed_claims,  # Clean list of individual claims
                        "summary_text": summary,
                        "domain": extracted_domain,
                        "evaluation_metadata": {
                            "model": model,
                            "prompt_name": prompt_name,
                            "processing_time": llm_result.get("processing_time", 0.0),
                            "usage": llm_result.get("usage", {}),
                            "total_claims_extracted": len(parsed_claims)
                        },
                        "llm_result": llm_result  # Keep full LLM response for debugging
                    }, f, indent=2)
                
                logger.info(f"Saved claims for {item_id} summary {summary_idx}")
                claim_results.extend(single_summary_results)
            
            # Transform results to match expected output format
            # Convert summaries to claims with chunks structure
            documents_with_claims = []
            for doc_idx, document in enumerate(item_data.get("documents", [])):
                doc_summaries = document.get("summaries", [])
                
                # Create claims structure for this document
                claims_for_summaries = []
                for summary_idx, summary in enumerate(doc_summaries):
                    if summary_idx < len(claim_results):
                        # Extract the actual claims from the LLM response
                        claim_response = claim_results[summary_idx].get("response", "")
                        
                        # Parse individual claims from the response
                        individual_claims = _parse_claims_from_response(claim_response)
                        
                        # Structure as requested: list of dicts with "claims" key
                        claims_for_summaries.append({
                            "claims": individual_claims
                        })
                
                # Create document structure with claims
                doc_with_claims = {
                    "metadata": document.get("metadata", {}),
                    "summaries": claims_for_summaries
                }
                documents_with_claims.append(doc_with_claims)
            
            # Calculate totals for this item
            item_cost = sum(result.get("usage", {}).get("total_cost", 0.0) for result in claim_results)
            item_tokens = sum(result.get("usage", {}).get("total_tokens", 0) for result in claim_results)
            
            total_cost += item_cost
            total_tokens += item_tokens
            
            # Store results
            results[item_id] = {
                "item_id": item_id,
                "documents": documents_with_claims,
                "claim_extraction_metadata": {
                    "num_summaries_processed": len(summaries),
                    "num_claims_extracted": len(claim_results),
                    "total_cost": item_cost,
                    "total_tokens": item_tokens,
                    "model": model,
                    "prompt_name": prompt_name,
                    "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            # Increment processed count and check stop condition
            processed_count += 1
            if stop is not None and processed_count >= stop:
                logger.info(f"Stopping after processing {processed_count} items (--stop {stop})")
                break
            
        except Exception as e:
            error_msg = f"Error processing item {item_id}: {str(e)}"
            logger.error(error_msg)
            errors[item_id] = error_msg
            
            # Also count failed items towards the stop limit
            processed_count += 1
            if stop is not None and processed_count >= stop:
                logger.info(f"Stopping after processing {processed_count} items (--stop {stop})")
                break
    
    # Get summarization info if available
    summarization_info = globals().get('_summarization_info', {})
    
    # Create output metadata
    output_metadata = {
        "claim_extraction_info": {
            "collection_metadata": {
                "strategy_function": "extract_claims_from_summaries",
                "content_type": "claims",
                "input_type": "summaries",
                "model": model,
                "prompt_name": prompt_name,
                "scope": scope,
                "source_collection": str(summary_collection_path),
                "hash_parameters": _generate_hash_parameters(model, prompt_name, scope),
            },
            "summarization_info": summarization_info,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items_processed": list(results.keys()),
            "processing_stats": {
                "total_items": len(items_to_process),
                "successful_items": len(results),
                "failed_items": len(errors),
                "total_cost": total_cost,
                "total_tokens": total_tokens,
            }
        }
    }
    # Save results
    save_claims(results, output_metadata, output_path, errors)
    
    return {
        "results": results,
        "metadata": output_metadata,
        "errors": errors
    }


def save_claims(
    results: dict[str, Any],
    metadata: dict[str, Any],
    output_path: str,
    errors: dict[str, str] | None = None
) -> None:
    """
    Save claim extraction results to files.
    
    Args:
        results: Dictionary of item results
        metadata: Collection metadata
        output_path: Path to save results
        errors: Optional dictionary of errors encountered
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save collection metadata
    collection_file = output_dir / "collection.json"
    with open(collection_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save individual item results
    items_dir = output_dir / "items"
    items_dir.mkdir(exist_ok=True)
    
    for item_id, item_data in results.items():
        item_file = items_dir / f"{item_id}.json"
        
        # Create the full item structure with metadata
        full_item_data = {
            "item_metadata": {
                "item_id": item_id,
                **item_data.get("claim_extraction_metadata", {})
            },
            "documents": item_data.get("documents", [])
        }
        
        with open(item_file, 'w') as f:
            json.dump(full_item_data, f, indent=2)
    
    # Save errors if any
    if errors:
        errors_file = output_dir / "errors.json"
        with open(errors_file, 'w') as f:
            json.dump(errors, f, indent=2)
    
    logger.info(f"Claim extraction results saved to: {output_path}")


def _parse_claims_from_response(response: str) -> list[str]:
    """
    Parse individual claims from the LLM response.
    
    Expected format is bullet points, each claim on a separate line.
    
    Args:
        response: Raw LLM response containing claims
        
    Returns:
        List of individual claim strings
    """
    if not response or not response.strip():
        return []
    
    claims = []
    lines = response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove bullet point markers (-, *, •, numbers like "1.", etc.)
        line = line.lstrip('- *•')
        line = line.strip()
        
        # Remove numbered list markers (1., 2., etc.)
        line = re.sub(r'^\d+\.\s*', '', line)
        line = line.strip()
        
        # Skip empty lines or common headers
        if line and line.lower() not in ['claims:', 'claims', 'extracted claims:']:
            claims.append(line)
    
    return claims


def _generate_hash_parameters(model: str, prompt_name: str, scope: str) -> dict[str, Any]:
    """Generate hash parameters for directory naming consistency."""
    hash_input = f"{model}_{prompt_name}_{scope}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]
    
    return {
        "model": model,
        "prompt_name": prompt_name,
        "scope": scope,
        "hash_note": "Directory name contains 6-char MD5 hash of these parameters to keep names short",
        "hash_value": hash_value
    }


__all__ = [
    "extract_claims_from_summaries",
    "process_dataset_summaries", 
    "save_claims"
]