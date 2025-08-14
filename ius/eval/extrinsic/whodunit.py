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
import pandas as pd

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



def get_ground_truth_from_sheets(story_id: str) -> Optional[Dict[str, Any]]:
    """
    Get ground truth data from Google Sheets as a fallback.
    
    Args:
        story_id: The story ID to look up
        
    Returns:
        Dictionary with ground truth data or None if not found/error
    """
    try:
        # Google Sheet configuration
        SHEET_ID = "1awnPbTUjIfVOqqhd8vWXQm8iwPXRMXJ4D1-MWfwLNwM"
        GID = "0"
        
        # Construct CSV export URL
        csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
        
        logger.debug(f"Fetching ground truth from Google Sheets for story_id: {story_id}")
        
        # Read the sheet (keep "N/A" as string, don't convert to NaN)
        df = pd.read_csv(csv_url, keep_default_na=False, na_values=[''])
        
        # Look up the story
        result = df[df['story_id'] == story_id]
        if result.empty:
            logger.debug(f"Story ID '{story_id}' not found in Google Sheets")
            return None
        
        row = result.iloc[0]
        
        # Extract ground truth data
        ground_truth = {
            "culprits": row.get('Gold label (human): main culprit(s)  (PRE-REVEAL)', None),
            "culprits_post_reveal": row.get('Gold label (human): main culprit(s), POST-REVEAL INFORMATION', None),
            "accomplices": row.get('Gold label (human): accomplice(s)', None)
        }
        
        # Convert pandas NaN, empty strings, and "N/A" to None
        for key, value in ground_truth.items():
            if pd.isna(value) or value == "" or value == "N/A":
                ground_truth[key] = "None"
        
        logger.info(f"✅ Found ground truth in Google Sheets for {story_id}")
        return ground_truth
        
    except Exception as e:
        logger.warning(f"Failed to get ground truth from Google Sheets for {story_id}: {e}")
        return None


def extract_ground_truth(item_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract ground truth culprit and accomplice information from item metadata.
    Falls back to Google Sheets if metadata is missing or empty.
    
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
    
    # Get story_id for potential Google Sheets fallback
    story_id = None
    try:
        # For whodunit evaluation items, the story_id is in item_metadata.item_id
        if "item_metadata" in item_data and "item_id" in item_data["item_metadata"]:
            story_id = item_data["item_metadata"]["item_id"]
        # Fallback: try to get from documents metadata
        elif "documents" in item_data and item_data["documents"]:
            story_id = item_data["documents"][0]["metadata"].get("id")
        # Last resort: try top-level id
        else:
            story_id = item_data.get("id")
    except (KeyError, IndexError, TypeError):
        pass
    
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
            logger.info("No original_metadata found in item - will try Google Sheets fallback")
            
    except (KeyError, IndexError) as e:
        logger.warning(f"Could not extract ground truth from metadata: {e}")
    
    # Check if we have any ground truth data, if not try Google Sheets fallback
    has_ground_truth = any(value not in [None, "", []] for value in ground_truth.values())
    
    if not has_ground_truth and story_id:
        logger.info(f"🔄 No ground truth found in metadata for {story_id}, trying Google Sheets fallback...")
        sheets_ground_truth = get_ground_truth_from_sheets(story_id)
        if sheets_ground_truth:
            # Update ground_truth with data from sheets
            for key, value in sheets_ground_truth.items():
                if value not in [None, "", []]:
                    ground_truth[key] = value
            logger.info(f"📊 Using Google Sheets ground truth for {story_id}")
        else:
            logger.warning(f"❌ No ground truth found in Google Sheets for {story_id}")
    elif not story_id:
        logger.warning("Cannot use Google Sheets fallback - no story_id found")
    
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
    Handles the new flattened format with multiple JSON sections.
    
    Args:
        response: Raw LLM scoring response
        
    Returns:
        Dictionary with parsed assessment combining all sections
    """
    import re
    import json
    
    try:
        # Initialize result structure
        result = {
            "culprit": {},
            "accomplice": {}
        }
        
        # Define section patterns and their target locations
        # Handle both direct JSON and markdown code blocks
        sections = {
            r"CULPRIT ASSESSMENT:\s*(?:```json\s*)?(\{[^}]*\})(?:\s*```)?": ("culprit", "assessment"),
            r"CULPRIT MINOR ERRORS:\s*(?:```json\s*)?(\{[^}]*\})(?:\s*```)?": ("culprit", "minor_errors"),
            r"CULPRIT MAJOR ERRORS:\s*(?:```json\s*)?(\{[^}]*\})(?:\s*```)?": ("culprit", "major_errors"),
            r"ACCOMPLICE ASSESSMENT:\s*(?:```json\s*)?(\{[^}]*\})(?:\s*```)?": ("accomplice", "assessment"),
            r"ACCOMPLICE MINOR ERRORS:\s*(?:```json\s*)?(\{[^}]*\})(?:\s*```)?": ("accomplice", "minor_errors"),
            r"ACCOMPLICE MAJOR ERRORS:\s*(?:```json\s*)?(\{[^}]*\})(?:\s*```)?": ("accomplice", "major_errors")
        }
        
        sections_found = 0
        
        # Extract each section
        for pattern, (category, section_name) in sections.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    json_str = match.group(1).strip()
                    parsed_section = json.loads(json_str)
                    
                    if section_name == "assessment":
                        # Merge assessment directly into the category
                        result[category].update(parsed_section)
                    else:
                        # Add as a subsection
                        result[category][section_name] = parsed_section
                    
                    sections_found += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse {section_name} section: {e}")
        
        if sections_found == 0:
            logger.warning("Could not find any JSON sections in scoring response")
            return None
        
        logger.info(f"Successfully parsed {sections_found} sections from scoring response")
        return result
        
    except Exception as e:
        logger.warning(f"Error parsing scoring response: {e}")
        return None


def score_whodunit_solution(
    llm_solution: Dict[str, str],
    ground_truth: Dict[str, Any],
    scoring_prompt_name: str,
    range_spec: str = "all",
    scoring_model: str = "gpt-4o",
    temperature: float = 0.1,
    max_completion_tokens: int = 2000,
    ask_user_confirmation: bool = False
) -> Dict[str, Any]:
    """
    Score a whodunit solution against ground truth using LLM.
    
    Args:
        llm_solution: Parsed LLM solution
        ground_truth: Ground truth data
        range_spec: Range specification used for text selection
        scoring_model: LLM model for scoring
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
    
    logger.info("Making LLM call for solution scoring...")
    
    try:
        # Make LLM call for scoring
        logger.debug(f"Making scoring LLM call with model: {scoring_model}")
        logger.debug(f"Template vars: {template_vars}")

        scoring_result = call_llm(
            text="",  # Not used for this prompt style
            system_and_user_prompt={
                "system": system_prompt,
                "user": user_prompt_template  # Pass template, not filled-in prompt
            },
            template_vars=template_vars,  # Pass template variables
            model=scoring_model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            ask_user_confirmation=ask_user_confirmation
        )
        
        logger.debug(f"Scoring result type: {type(scoring_result)}")
        logger.debug(f"Scoring result keys: {list(scoring_result.keys()) if scoring_result else 'None'}")
        
        # Check if scoring_result is valid
        if not scoring_result:
            raise ValueError("LLM call returned None")
        
        if "response" not in scoring_result:
            raise ValueError(f"LLM call result missing 'response' key. Keys: {list(scoring_result.keys())}")
        
        # Parse the scoring response
        raw_response = scoring_result["response"]
        logger.debug(f"Raw scoring response type: {type(raw_response)}")
        logger.debug(f"Raw scoring response (first 200 chars): {raw_response[:200] if raw_response else 'None'}")
        
        parsed_assessment = parse_scoring_response(raw_response)
        logger.debug(f"Parsed assessment type: {type(parsed_assessment)}")
        logger.debug(f"Parsed assessment: {parsed_assessment}")
        
        return {
            "assessment": parsed_assessment,
            "raw_response": scoring_result["response"],
            "finish_reason": scoring_result.get("finish_reason"),
            "usage": scoring_result.get("usage", {"total_cost": 0, "total_tokens": 0}),
            "final_prompts_used": scoring_result.get("final_prompts_used", {}),
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Error during solution scoring: {e}")
        return {
            "assessment": None,
            "error": str(e),
            "raw_response": None,
            "finish_reason": None,
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


def _update_collection_metadata(collection_file: Path, item_id: str, cost: float, tokens: int, success: bool, skipped: bool = False):
    """Update collection.json with results from processing one item."""
    with open(collection_file, 'r') as f:
        collection_data = json.load(f)
    
    stats = collection_data["whodunit_evaluation_info"]["processing_stats"]
    
    # Update items_processed list
    if item_id not in collection_data["whodunit_evaluation_info"]["items_processed"]:
        collection_data["whodunit_evaluation_info"]["items_processed"].append(item_id)
    
    # Update stats
    if success:
        stats["successful_items"] += 1
    elif skipped:
        stats["skipped_items"] += 1
    else:
        stats["failed_items"] += 1
    
    stats["total_cost"] += cost
    stats["total_tokens"] += tokens
    
    # Save updated collection
    with open(collection_file, 'w') as f:
        json.dump(collection_data, f, indent=2)


def run_whodunit_evaluation(
    input_dir: str,
    range_spec: str = "all",
    prompt_name: str = "default-whodunit-culprits-and-accomplices",
    scoring_prompt_name: Optional[str] = None,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.1,
    max_completion_tokens: int = 100000, # Absurdly high value to account for reasoning tokens
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
        model: LLM model to use for solving
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
    
    # Create initial collection metadata
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
            "items_processed": [],
            "processing_stats": {
                "total_items": 0,  # Will be updated after we count item_files
                "successful_items": 0,
                "skipped_items": 0,
                "failed_items": 0,
                "total_cost": 0.0,
                "total_tokens": 0
            }
        }
    }
    
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
    
    # Update total_items count and save initial collection.json
    collection_data["whodunit_evaluation_info"]["processing_stats"]["total_items"] = len(item_files)
    collection_file = output_path / "collection.json"
    with open(collection_file, 'w') as f:
        json.dump(collection_data, f, indent=2)
    
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
            # Track whether we executed Phase 1 (solving)
            phase1_executed = False
            llm_result = None
            
            # PHASE 1: SOLVE (if needed)
            if not item_output_file.exists() or overwrite:
                phase1_executed = True
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
                    "finish_reason": llm_result["finish_reason"],
                    "parsed_response": parsed_sections,
                    "solution_correctness_assessment": None
                }
                
                # Save item result
                with open(item_output_file, 'w') as f:
                    json.dump(item_result, f, indent=2)
                
                total_cost += llm_result["usage"]["total_cost"]
                total_tokens += llm_result["usage"]["total_tokens"]
            
            # Handle skipped items (log message)
            if not phase1_executed:
                logger.info(f"⏭️  Skipping {item_id} (already exists, use --overwrite to regenerate)")
            
            # PHASE 2: SCORE (if needed)
            if item_output_file.exists():
                # Use a different model for scoring
                scoring_model = "gpt-4o"

                # Load existing item
                with open(item_output_file) as f:
                    item_result = json.load(f)
                
                # Check if scoring is needed and scoring prompt is provided
                if scoring_prompt_name and (item_result.get("solution_correctness_assessment") is None or rescore):
                    # Check if ground truth is available for scoring
                    ground_truth = item_result.get("ground_truth", {})
                    parsed_response = item_result.get("parsed_response")
                    
                    # If no ground truth in item result, try to extract it again (including Google Sheets fallback)
                    if not ground_truth.get("culprits"):
                        logger.info(f"🔄 No ground truth found in item result for {item_id}, trying to extract again...")
                        with open(item_file) as f:
                            original_item_data = json.load(f)
                        ground_truth = extract_ground_truth(original_item_data)
                        # Update the item result with the newly extracted ground truth
                        item_result["ground_truth"] = ground_truth
                    
                    if not ground_truth.get("culprits"):
                        logger.info(f"⏭️  No ground truth available for {item_id} - skipping scoring (can be done later)")
                    elif not parsed_response or not isinstance(parsed_response, dict):
                        logger.warning(f"⏭️  No valid parsed response for {item_id} - skipping scoring")
                    elif not parsed_response.get("suspects") or not parsed_response.get("main_culprits"):
                        logger.info(f"⏭️  LLM solution incomplete for {item_id} (missing suspects or main_culprits) - skipping scoring")
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
                            scoring_model=scoring_model,
                            temperature=temperature,
                            max_completion_tokens=1000,
                            ask_user_confirmation=ask_user_confirmation
                        )
                        
                        # Update item with scoring results
                        item_result["solution_correctness_assessment"] = scoring_result["assessment"]
                        item_result["scoring_metadata"] = {
                            "scoring_model": scoring_model,
                            "scoring_temperature": temperature,
                            "scoring_raw_response": scoring_result["raw_response"],
                            "scoring_finish_reason": scoring_result.get("finish_reason"),
                            "scoring_usage": scoring_result["usage"],
                            "scoring_error": scoring_result["error"],
                            "scoring_timestamp": datetime.now().isoformat(),
                            "scoring_prompts_used": scoring_result.get("final_prompts_used", {})
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
            
            # Update collection metadata incrementally after both phases
            item_cost = 0
            item_tokens = 0
            
            # Add main LLM costs if Phase 1 was executed
            if phase1_executed and llm_result:
                item_cost += llm_result["usage"]["total_cost"]
                item_tokens += llm_result["usage"]["total_tokens"]
            
            # Add scoring costs if applicable (regardless of whether Phase 1 was executed)
            if scoring_prompt_name and item_result.get("scoring_metadata"):
                item_cost += item_result["scoring_metadata"]["scoring_usage"]["total_cost"]
                item_tokens += item_result["scoring_metadata"]["scoring_usage"]["total_tokens"]
            
            # Determine if this was a skip or success
            was_skipped = not phase1_executed
            
            _update_collection_metadata(
                collection_file, 
                item_id, 
                item_cost,
                item_tokens,
                success=not was_skipped,
                skipped=was_skipped
            )
            
        except Exception as e:
            logger.error(f"Error processing item {item_id}: {e}")
            # Update collection metadata for failed item
            _update_collection_metadata(collection_file, item_id, 0, 0, success=False, skipped=False)
            continue
    # Load final collection metadata for return
    with open(collection_file, 'r') as f:
        final_collection_data = json.load(f)
    
    final_stats = final_collection_data["whodunit_evaluation_info"]["processing_stats"]
    
    logger.info(f"Evaluation completed. Results saved to: {output_path}")
    logger.info(f"Processed {final_stats['successful_items']}/{final_stats['total_items']} items successfully")
    logger.info(f"Skipped: {final_stats['skipped_items']}, Failed: {final_stats['failed_items']}")
    logger.info(f"Total cost: ${final_stats['total_cost']:.4f}")
    logger.info(f"Total tokens: {final_stats['total_tokens']}")
    
    return {
        "output_dir": str(output_path),
        "results": results,
        "total_cost": final_stats['total_cost'],
        "total_tokens": final_stats['total_tokens'],
        "collection_metadata": final_collection_data
    }