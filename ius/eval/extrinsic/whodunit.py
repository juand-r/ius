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
        range_spec: Range specification ('all', 'last', 'penultimate', 'all-but-last', '1', '1-3', etc.)
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
    elif range_spec == "all-but-last":
        if total_segments < 2:
            raise WhodunitError("Cannot use 'all-but-last' - need at least 2 segments")
        return list(range(1, total_segments))
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



def extract_ground_truth(item_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract ground truth culprit and accomplice information from item metadata.
    
    Args:
        item_data: Item JSON data
        
    Returns:
        Dictionary with ground truth information
    """
    ground_truth = {
        "culprits": None,
        "culprits_post_reveal": None,
        "accomplices": None
    }
    
    try:
        metadata = item_data["documents"][0]["metadata"]
        
        # Check if original_metadata exists (for BMDS, but not DetectiveQA)
        if "original_metadata" in metadata:
            original_metadata = metadata["original_metadata"]
            
            # Extract culprits
            if "culprit(s), human annotated" in original_metadata:
                ground_truth["culprits"] = original_metadata["culprit(s), human annotated"]
            
            # Extract post-reveal culprits
            if "culprit(s), human annotated--post-reveal" in original_metadata:
                ground_truth["culprits_post_reveal"] = original_metadata["culprit(s), human annotated--post-reveal"]
            
            # Extract accomplices
            if "accomplice(s), human annotated" in original_metadata:
                ground_truth["accomplices"] = original_metadata["accomplice(s), human annotated"]
                
        else:
            logger.info("No original_metadata found in item - ground truth will be None")
            
    except (KeyError, IndexError) as e:
        logger.warning(f"Could not extract ground truth: {e}")
    
    return ground_truth


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


def parse_scoring_response(response: str) -> Dict[str, Any]:
    """
    Parse LLM scoring response into structured assessment.
    
    Args:
        response: Raw LLM scoring response
        
    Returns:
        Dictionary with parsed assessment
    """
    try:
        # Extract JSON from <ASSESSMENT> tags
        match = re.search(r"<ASSESSMENT>(.*?)</ASSESSMENT>", response, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)
        else:
            logger.warning("Could not find <ASSESSMENT> tags in scoring response")
            return None
    except json.JSONDecodeError as e:
        logger.warning(f"Could not parse scoring response as JSON: {e}")
        return None


def score_whodunit_solution(
    llm_solution: Dict[str, str],
    ground_truth: Dict[str, Any],
    scoring_prompt_name: str,
    range_spec: str = "all",
    model: str = "gpt-4.1-mini",
    temperature: float = 0.1,
    max_completion_tokens: int = 1000,
    ask_user_confirmation: bool = False
) -> Dict[str, Any]:
    """
    Score a whodunit solution against ground truth using LLM.
    
    Args:
        llm_solution: Parsed LLM solution
        ground_truth: Ground truth data
        range_spec: Range specification used for text selection
        model: LLM model for scoring
        temperature: LLM temperature
        max_completion_tokens: Max tokens for scoring
        ask_user_confirmation: Whether to ask for confirmation
        
    Returns:
        Dictionary with scoring results
    """
    from ius.utils import call_llm
    
    # Load scoring prompts
    scoring_prompt_dir = Path(f"prompts/extrinsic-eval/{scoring_prompt_name}")
    
    try:
        system_prompt, user_prompt_template = load_prompts(scoring_prompt_dir)
    except Exception as e:
        logger.error(f"Could not load scoring prompts: {e}")
        return {
            "assessment": None,
            "error": f"Could not load scoring prompts: {e}",
            "raw_response": None,
            "usage": {"total_cost": 0, "total_tokens": 0}
        }
    
    # Simple ground truth selection: use post-reveal if range includes the end and it exists
    if range_spec in ["last", "all"] and ground_truth.get("culprits_post_reveal"):
        selected_culprits = ground_truth.get("culprits_post_reveal")
    else:
        selected_culprits = ground_truth.get("culprits")
    
    # Prepare template variables
    template_vars = {
        "suspects": llm_solution.get("suspects", "None"),
        "ground_truth_culprits": selected_culprits or "None",
        "ground_truth_accomplices": ground_truth.get("accomplices", "None"),
        "llm_main_culprits": llm_solution.get("main_culprits", "None"),
        "llm_accomplices": llm_solution.get("accomplices", "None")
    }
    
    # Fill in the user prompt
    user_prompt = user_prompt_template.format(**template_vars)
    
    logger.info("Making LLM call for solution scoring...")
    
    try:
        # Make LLM call for scoring
        scoring_result = call_llm(
            text="",  # Not used for this prompt style
            system_and_user_prompt={
                "system": system_prompt,
                "user": user_prompt
            },
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            ask_user_confirmation=ask_user_confirmation
        )
        
        # Parse the scoring response
        parsed_assessment = parse_scoring_response(scoring_result["response"])
        
        return {
            "assessment": parsed_assessment,
            "raw_response": scoring_result["response"],
            "usage": scoring_result["usage"],
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Error during solution scoring: {e}")
        return {
            "assessment": None,
            "error": str(e),
            "raw_response": None,
            "usage": {"total_cost": 0, "total_tokens": 0}
        }


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
    scoring_prompt_name: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_completion_tokens: int = 2000,
    item_ids: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    rescore: bool = False,
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
        max_completion_tokens: Maximum tokens for LLM response
        item_ids: Specific item IDs to process (None for all)
        output_dir: Custom output directory (None for auto-generated)
        overwrite: Whether to overwrite existing item results
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
            "max_completion_tokens": max_completion_tokens
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
    skipped_items = []
    total_cost = 0.0
    total_tokens = 0
    
    for i, item_file in enumerate(item_files, 1):
        item_id = item_file.stem
        logger.info(f"[{i}/{len(item_files)}] Processing item: {item_id}")
        
        item_output_file = items_output_dir / f"{item_id}.json"
        
        try:
            # PHASE 1: SOLVE (if needed)
            if not item_output_file.exists() or overwrite:
                logger.info(f"🔍 Solving mystery for {item_id}...")
                
                # Load segments and reveal
                segments, reveal_segment = load_item_segments(item_file, input_type)
                
                # Select text based on range
                selected_text, selected_indices = select_text_segments(segments, range_spec)
                
                # Print reveal preview
                reveal_preview = reveal_segment[:300] + "..." if len(reveal_segment) > 300 else reveal_segment
                logger.info(f"Using reveal segment (first 300 chars): {reveal_preview}")
                logger.info(f"Selected range: {range_spec} -> segments {selected_indices} of {len(segments)}")
                
                # Build the actual user prompt with template substitution
                actual_user_prompt = user_template.format(
                    text=selected_text,
                    reveal_chunk=reveal_segment
                )
                
                # Call LLM for solving
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
                    max_completion_tokens=max_completion_tokens,
                    ask_user_confirmation=ask_user_confirmation
                )
                
                # Parse response
                parsed_sections = parse_llm_response(llm_result["response"])
                
                # Extract ground truth from original item
                with open(item_file) as f:
                    original_item_data = json.load(f)
                ground_truth = extract_ground_truth(original_item_data)
                
                # Create result with solution_correctness_assessment set to None
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
                        "max_completion_tokens": max_completion_tokens,
                        "prompts_used": {
                            "system": system_prompt,
                            "user": actual_user_prompt
                        },
                        "command_run": command_run,
                        "processing_time": llm_result.get("processing_time", 0),
                        "usage": llm_result["usage"]
                    },
                    "ground_truth": ground_truth,
                    "raw_response": llm_result["response"],
                    "parsed_response": parsed_sections,
                    "solution_correctness_assessment": None
                }
                
                # Save item result
                with open(item_output_file, 'w') as f:
                    json.dump(item_result, f, indent=2)
                
                total_cost += llm_result["usage"]["total_cost"]
                total_tokens += llm_result["usage"]["total_tokens"]
            
            # PHASE 2: SCORE (if needed)
            if item_output_file.exists():
                # Load existing item
                with open(item_output_file) as f:
                    item_result = json.load(f)
                
                # Check if scoring is needed and scoring prompt is provided
                if scoring_prompt_name and (item_result.get("solution_correctness_assessment") is None or rescore):
                    # Check if ground truth is available for scoring
                    ground_truth = item_result.get("ground_truth", {})
                    if not ground_truth.get("culprits"):
                        logger.info(f"⏭️  No ground truth available for {item_id} - skipping scoring (can be done later)")
                    else:
                        if rescore and item_result.get("solution_correctness_assessment") is not None:
                            logger.info(f"🔄 Re-scoring solution for {item_id}...")
                        else:
                            logger.info(f"📊 Scoring solution for {item_id}...")
                        
                        # Score the solution
                        scoring_result = score_whodunit_solution(
                            llm_solution=item_result["parsed_response"],
                            ground_truth=item_result["ground_truth"],
                            scoring_prompt_name=scoring_prompt_name,
                            range_spec=range_spec,
                            model=model,
                            temperature=temperature,
                            max_completion_tokens=1000,
                            ask_user_confirmation=ask_user_confirmation
                        )
                        
                        # Update item with scoring results
                        item_result["solution_correctness_assessment"] = scoring_result["assessment"]
                        item_result["scoring_metadata"] = {
                            "scoring_model": model,
                            "scoring_temperature": temperature,
                            "scoring_raw_response": scoring_result["raw_response"],
                            "scoring_usage": scoring_result["usage"],
                            "scoring_error": scoring_result["error"],
                            "scoring_timestamp": datetime.now().isoformat()
                        }
                        
                        # Save updated item
                        with open(item_output_file, 'w') as f:
                            json.dump(item_result, f, indent=2)
                        
                        total_cost += scoring_result["usage"]["total_cost"]
                        total_tokens += scoring_result["usage"]["total_tokens"]
                elif not scoring_prompt_name:
                    logger.info(f"⏭️  No scoring prompt provided - skipping Phase 2 scoring for {item_id}")
                else:
                    logger.info(f"⏭️  Scoring already completed for {item_id}")
                
                results.append(item_result)
            
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
                "max_completion_tokens": max_completion_tokens,
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
                "skipped_items": len(skipped_items),
                "failed_items": len(item_files) - len(results) - len(skipped_items),
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