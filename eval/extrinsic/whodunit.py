#!/usr/bin/env python3
"""
Extrinsic evaluation for detective stories using whodunit prompts.
Evaluates summaries or chunks by asking an LLM to identify the culprit.
"""

import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from ius.utils import call_llm
from ius.logging_config import setup_logging
from ius.exceptions import IUSError

import logging
logger = logging.getLogger(__name__)


class WhodunitError(IUSError):
    """Whodunit evaluation-related errors."""
    pass


def detect_input_type(input_dir: Path) -> str:
    """
    Detect whether input contains chunks or summaries.
    
    Args:
        input_dir: Path to input directory
        
    Returns:
        'chunks' or 'summaries'
        
    Raises:
        WhodunitError: If input type cannot be determined
    """
    items_dir = input_dir / "items"
    if not items_dir.exists():
        raise WhodunitError(f"No items directory found in {input_dir}")
    
    # Check first item file to determine structure
    item_files = list(items_dir.glob("*.json"))
    if not item_files:
        raise WhodunitError(f"No item files found in {items_dir}")
    
    with open(item_files[0]) as f:
        item_data = json.load(f)
    
    if "documents" not in item_data:
        raise WhodunitError(f"Invalid item structure in {item_files[0]}")
    
    doc = item_data["documents"][0]
    
    if "chunks" in doc:
        return "chunks"
    elif "summaries" in doc:
        return "summaries"
    else:
        raise WhodunitError(f"Cannot determine input type from {item_files[0]} - no 'chunks' or 'summaries' found")


def load_item_segments(item_path: Path, input_type: str) -> Tuple[List[str], str]:
    """
    Load segments (chunks or summaries) and reveal segment from an item file.
    
    Args:
        item_path: Path to item JSON file
        input_type: 'chunks' or 'summaries'
        
    Returns:
        Tuple of (segments_list, reveal_segment)
    """
    with open(item_path) as f:
        item_data = json.load(f)
    
    doc = item_data["documents"][0]
    
    if input_type == "chunks":
        segments = doc["chunks"]
    elif input_type == "summaries":
        segments = doc["summaries"]
    else:
        raise WhodunitError(f"Unknown input type: {input_type}")
    
    if not segments:
        raise WhodunitError(f"No segments found in {item_path}")
    
    # Reveal segment is always the last one
    reveal_segment = segments[-1]
    
    return segments, reveal_segment


def parse_range_spec(range_spec: str, total_segments: int) -> List[int]:
    """
    Parse range specification into list of 1-indexed segment numbers.
    
    Args:
        range_spec: Range specification ('all', 'last', 'penultimate', '1', '1-3', etc.)
        total_segments: Total number of available segments
        
    Returns:
        List of 1-indexed segment numbers
        
    Raises:
        WhodunitError: If range specification is invalid
    """
    if total_segments == 0:
        raise WhodunitError("No segments available")
    
    if range_spec == "all":
        return list(range(1, total_segments + 1))
    elif range_spec == "last":
        return [total_segments]
    elif range_spec == "penultimate":
        if total_segments < 2:
            raise WhodunitError("Cannot use 'penultimate' - need at least 2 segments")
        return [total_segments - 1]
    elif "-" in range_spec:
        # Range like "1-3"
        try:
            start_str, end_str = range_spec.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            
            if start < 1 or end < 1:
                raise WhodunitError("Range indices must be >= 1")
            if start > total_segments or end > total_segments:
                raise WhodunitError(f"Range {range_spec} exceeds available segments (1-{total_segments})")
            if start > end:
                raise WhodunitError(f"Invalid range {range_spec} - start must be <= end")
                
            return list(range(start, end + 1))
        except ValueError:
            raise WhodunitError(f"Invalid range format: {range_spec}")
    else:
        # Single number
        try:
            index = int(range_spec)
            if index < 1:
                raise WhodunitError("Segment index must be >= 1")
            if index > total_segments:
                raise WhodunitError(f"Segment index {index} exceeds available segments (1-{total_segments})")
            return [index]
        except ValueError:
            raise WhodunitError(f"Invalid range specification: {range_spec}")


def select_text_segments(segments: List[str], range_spec: str) -> Tuple[str, List[int]]:
    """
    Select and concatenate text segments based on range specification.
    
    Args:
        segments: List of text segments
        range_spec: Range specification
        
    Returns:
        Tuple of (concatenated_text, selected_indices)
    """
    selected_indices = parse_range_spec(range_spec, len(segments))
    
    # Convert to 0-indexed for actual selection
    selected_texts = [segments[i - 1] for i in selected_indices]
    
    # Concatenate with double newlines
    concatenated_text = "\n".join(selected_texts)
    
    return concatenated_text, selected_indices


def load_prompts(prompt_dir: Path) -> Tuple[str, str]:
    """
    Load system and user prompts from directory.
    
    Args:
        prompt_dir: Path to prompt directory
        
    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    system_file = prompt_dir / "system.txt"
    user_file = prompt_dir / "user.txt"
    
    if not system_file.exists():
        raise WhodunitError(f"System prompt not found: {system_file}")
    if not user_file.exists():
        raise WhodunitError(f"User prompt not found: {user_file}")
    
    with open(system_file) as f:
        system_prompt = f.read().strip()
    
    with open(user_file) as f:
        user_prompt_template = f.read().strip()
    
    return system_prompt, user_prompt_template



def parse_llm_response(response: str) -> Dict[str, str]:
    """
    Parse LLM response into structured sections.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Dictionary with parsed sections
    """
    sections = {
        "thought_process": "",
        "suspects": "",
        "main_culprits": "",
        "accomplices": "",
        "event_reconstruction": "",
        "why_others_innocent": "",
        "raw_response": response
    }
    
    # Define section patterns (case-insensitive)
    patterns = {
        "thought_process": r"<THOUGHT PROCESS>(.*?)</THOUGHT PROCESS>",
        "suspects": r"<SUSPECTS>(.*?)</SUSPECTS>",
        "main_culprits": r"<MAIN CULPRIT\(S\)>(.*?)</MAIN CULPRIT\(S\)>",
        "accomplices": r"<ACCOMPLICE\(S\)>(.*?)</ACCOMPLICE\(S\)>",
        "event_reconstruction": r"<EVENT RECONSTRUCTION>(.*?)</EVENT RECONSTRUCTION>",
        "why_others_innocent": r"<WHY THE OTHER SUSPECTS ARE INNOCENT>(.*?)</WHY THE OTHER SUSPECTS ARE INNOCENT>"
    }
    
    for section_name, pattern in patterns.items():
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            sections[section_name] = match.group(1).strip()
        else:
            logger.warning(f"Could not find section '{section_name}' in LLM response")
    
    return sections


def generate_output_hash(params: Dict[str, Any]) -> str:
    """
    Generate a short hash for output directory naming.
    
    Args:
        params: Parameters to include in hash
        
    Returns:
        6-character hash string
    """
    # Create a string representation of key parameters
    hash_input = json.dumps(params, sort_keys=True)
    
    # Generate MD5 hash and take first 6 characters
    return hashlib.md5(hash_input.encode()).hexdigest()[:6]


def run_whodunit_evaluation(
    input_dir: str,
    range_spec: str = "all",
    prompt_name: str = "default-whodunit-culprits-and-accomplices",
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    item_ids: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    command_run: Optional[str] = None,
    ask_user_confirmation: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run whodunit evaluation on detective stories.
    
    Args:
        input_dir: Path to input directory (chunks or summaries)
        range_spec: Range specification for text selection
        prompt_name: Name of prompt directory
        model: LLM model to use
        temperature: LLM temperature
        max_tokens: Maximum tokens for LLM response
        item_ids: Specific item IDs to process (None for all)
        output_dir: Custom output directory (None for auto-generated)
        command_run: Command that was run (for reproducibility)
        ask_user_confirmation: Whether to ask for user confirmation
        verbose: Enable verbose logging
        
    Returns:
        Dictionary with evaluation results and metadata
    """
    if verbose:
        setup_logging(log_level="DEBUG")
    else:
        setup_logging(log_level="INFO")
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise WhodunitError(f"Input directory does not exist: {input_dir}")
    
    # Detect input type
    input_type = detect_input_type(input_path)
    logger.info(f"Detected input type: {input_type}")
    
    # Load prompts
    prompts_dir = Path("prompts/extrinsic-eval") / prompt_name
    system_prompt, user_template = load_prompts(prompts_dir)
    
    # Generate output directory name
    if output_dir is None:
        input_name = input_path.name
        hash_params = {
            "model": model,
            "prompt_name": prompt_name,
            "range_spec": range_spec,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        output_hash = generate_output_hash(hash_params)
        output_dir = f"outputs/eval/extrinsic/{input_name}_whodunit_{output_hash}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    items_output_dir = output_path / "items"
    items_output_dir.mkdir(exist_ok=True)
    
    # Get list of items to process
    items_dir = input_path / "items"
    if item_ids:
        item_files = [items_dir / f"{item_id}.json" for item_id in item_ids]
        # Verify all files exist
        for item_file in item_files:
            if not item_file.exists():
                raise WhodunitError(f"Item file not found: {item_file}")
    else:
        item_files = sorted(items_dir.glob("*.json"))
    
    logger.info(f"Processing {len(item_files)} items")
    
    # Process each item
    results = []
    total_cost = 0.0
    total_tokens = 0
    
    for i, item_file in enumerate(item_files, 1):
        item_id = item_file.stem
        logger.info(f"[{i}/{len(item_files)}] Processing item: {item_id}")
        
        try:
            # Load segments and reveal
            segments, reveal_segment = load_item_segments(item_file, input_type)
            
            # Select text based on range
            selected_text, selected_indices = select_text_segments(segments, range_spec)
            
            # Print reveal preview
            reveal_preview = reveal_segment[:300] + "..." if len(reveal_segment) > 300 else reveal_segment
            logger.info(f"Using reveal segment (first 300 chars): {reveal_preview}")
            logger.info(f"Selected range: {range_spec} -> segments {selected_indices} of {len(segments)}")
            
            # Call LLM (template substitution handled by call_llm)
            
            llm_result = call_llm(
                text=selected_text,
                model=model,
                system_and_user_prompt={
                    "system": system_prompt,
                    "user": user_template
                },
                template_vars={
                    "text": selected_text,
                    "reveal_chunk": reveal_segment
                },
                temperature=temperature,
                max_completion_tokens=max_tokens,
                ask_user_confirmation=ask_user_confirmation
            )
            
            # Parse response
            parsed_sections = parse_llm_response(llm_result["response"])
            

            
            # Create result
            item_result = {
                "item_metadata": {
                    "item_id": item_id,
                    "input_type": input_type,
                    "selected_range": range_spec,
                    "selected_indices": selected_indices,
                    "selected_text_length": len(selected_text),
                    "reveal_segment_length": len(reveal_segment),
                    "reveal_preview": reveal_preview,
                    "whodunit_timestamp": datetime.now().isoformat()
                },
                "evaluation_metadata": {
                    "prompt_name": prompt_name,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "prompts_used": {
                        "system": system_prompt,
                        "user": user_template
                    },
                    "command_run": command_run,
                    "processing_time": llm_result.get("processing_time", 0),
                    "usage": llm_result["usage"]
                },
                "parsed_response": parsed_sections
            }
            
            # Save item result
            item_output_file = items_output_dir / f"{item_id}.json"
            with open(item_output_file, 'w') as f:
                json.dump(item_result, f, indent=2)
            
            results.append(item_result)
            total_cost += llm_result["usage"]["total_cost"]
            total_tokens += llm_result["usage"]["total_tokens"]
            
        except Exception as e:
            logger.error(f"Error processing item {item_id}: {e}")
            continue
    

    
    # Create collection metadata
    collection_data = {
        "whodunit_evaluation_info": {
            "collection_metadata": {
                "evaluation_function": "run_whodunit_evaluation",
                "content_type": "whodunit_analysis",
                "input_type": input_type,
                "model": model,
                "prompt_name": prompt_name,
                "range_spec": range_spec,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompts_used": {
                    "system": system_prompt,
                    "user": user_template
                },
                "source_collection": str(input_path),
                "command_run": command_run,
                "hash_parameters": hash_params,
                "hash_note": "Directory name contains 6-char MD5 hash of these parameters",
                "hash_value": output_hash
            },
            "timestamp": datetime.now().isoformat(),
            "items_processed": [r["item_metadata"]["item_id"] for r in results],
            "processing_stats": {
                "total_items": len(item_files),
                "successful_items": len(results),
                "failed_items": len(item_files) - len(results),
                "total_cost": total_cost,
                "total_tokens": total_tokens
            }
        }
    }
    
    # Save collection metadata
    collection_file = output_path / "collection.json"
    with open(collection_file, 'w') as f:
        json.dump(collection_data, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to: {output_path}")
    logger.info(f"Processed {len(results)}/{len(item_files)} items successfully")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Total tokens: {total_tokens}")
    
    return {
        "output_dir": str(output_path),
        "results": results,
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "collection_metadata": collection_data
    }