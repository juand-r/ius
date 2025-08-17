#!/usr/bin/env python3
"""
Extrinsic evaluation for detective stories using whodunit prompts.
Evaluates summaries or chunks by asking an LLM to identify the culprit.
"""

import json
import os
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
            "accomplices": row.get('Gold label (human): accomplice(s)', None),
            "suspects": row.get('Gold suspects pre-reveal', None)
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


def extract_ground_truth(item_data: Dict[str, Any], input_path: str = None) -> Dict[str, Any]:
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
        "accomplices": None,
        "suspects": None
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
    
    # Detect dataset from input path - much cleaner than checking story_id prefixes
    is_bmds = False
    if input_path:
        # Extract dataset name from path like "outputs/chunks/bmds_fixed_size2_8000/"
        path_lower = input_path.lower()
        if "/bmds" in path_lower or "bmds_" in path_lower:
            is_bmds = True
    
    if is_bmds and story_id:
        # This is BMDS - use Google Sheets as primary source
        logger.info(f"📊 BMDS dataset detected from input path, using Google Sheets as primary source for {story_id}...")
        sheets_ground_truth = get_ground_truth_from_sheets(story_id)
        if sheets_ground_truth:
            # Replace ground_truth with data from sheets
            ground_truth = sheets_ground_truth
            logger.info(f"✅ Using Google Sheets ground truth for BMDS story {story_id}")
        else:
            logger.warning(f"❌ No ground truth found in Google Sheets for BMDS story {story_id}")
    else:
        # Non-BMDS dataset - extract suspects from dataset-specific locations
        dataset_name = "unknown"
        if input_path:
            # Get last directory and split on underscore, take first part
            last_dir = os.path.basename(input_path.rstrip("/"))
            dataset_name = last_dir.split("_")[0]
        
        logger.info(f"📁 Non-BMDS dataset ({dataset_name}), using JSON metadata for ground truth")
        
        # Extract suspects for non-BMDS datasets
        try:
            metadata = item_data["documents"][0]["metadata"]
            # True Detective dataset - extract suspects from answer_options  
            if "true-detective" in dataset_name.lower():
                puzzle_data = metadata.get("original_metadata", {}).get("original_metadata", {}).get("puzzle_data", {})
                answer_options = puzzle_data.get("answer_options", "")
                if answer_options:
                    ground_truth["suspects"] = answer_options
                    logger.info(f"✅ Extracted suspects for true-detective from answer_options: {answer_options}")
                else:
                    logger.warning(f"❌ No answer_options found for true-detective item {story_id}")
            else:
                # Other non-BMDS datasets
                raise NotImplementedError(f"Dataset '{dataset_name}' suspect extraction not implemented yet. Only true-detective is supported for non-BMDS datasets.")
            
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Could not extract suspects for non-BMDS dataset {dataset_name}: {e}")
        
        if not story_id:
            logger.warning("Cannot determine dataset type - no story_id found")
    
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


def normalize_field_names(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize field names by replacing hyphens with underscores.
    
    Args:
        data: Dictionary with potentially unnormalized field names
        
    Returns:
        Dictionary with normalized field names
    """
    if not isinstance(data, dict):
        return data
    
    normalized = {}
    for key, value in data.items():
        # Normalize the key by replacing hyphens with underscores
        normalized_key = key.replace('-', '_')
        
        # Recursively normalize nested dictionaries
        if isinstance(value, dict):
            normalized[normalized_key] = normalize_field_names(value)
        else:
            normalized[normalized_key] = value
    
    return normalized


def validate_scoring_response_fields(parsed_response: Dict[str, Any]) -> None:
    """
    Validate that the parsed scoring response contains exactly the expected fields.
    The prompt asks for separate JSON objects, so we validate the flattened structure.
    
    Args:
        parsed_response: Parsed scoring response from LLM (after combining separate JSONs)
        
    Raises:
        ValueError: If any expected fields are missing or unexpected fields are present
    """
    # Define expected fields for each separate JSON section as they appear in the prompt
    expected_sections = {
        # CULPRIT ASSESSMENT JSON
        "culprit_assessment": {"culprit_correct"},
        
        # CULPRIT MINOR ERRORS JSON  
        "culprit_minor_errors": {
            "culprit_missing_or_wrong_alias",
            "hallucinated_part_of_name", 
            "missing_part_of_name",
            "included_accomplice"
        },
        
        # CULPRIT MAJOR ERRORS JSON
        "culprit_major_errors": {
            "different_suspect_not_accomplice",
            "confused_swapped_culprit_and_accomplice",
            "missing_real_name_only_has_alias",
            "included_other_non_accomplice_suspects"
        },
        
        # ACCOMPLICE ASSESSMENT JSON
        "accomplice_assessment": {"accomplice_correct"},
        
        # ACCOMPLICE MINOR ERRORS JSON
        "accomplice_minor_errors": {
            "accomplice_missing_or_wrong_alias",
            "hallucinated_part_of_name",
            "missing_part_of_name", 
            "included_culprit"
        },
        
        # ACCOMPLICE MAJOR ERRORS JSON
        "accomplice_major_errors": {
            "different_suspect_not_culprit",
            "confused_swapped_accomplice_and_culprit",
            "missing_real_name_only_has_alias",
            "included_other_non_culprit_suspects"
        }
    }
    
    errors = []
    
    # Check that we have the expected nested structure from parse_scoring_response
    if "culprit" not in parsed_response or "accomplice" not in parsed_response:
        errors.append("Missing culprit or accomplice categories in parsed response")
        
    if errors:
        error_msg = "Scoring response validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate culprit fields
    culprit_data = parsed_response["culprit"]
    
    # Check culprit_correct (main assessment)
    if "culprit_correct" not in culprit_data:
        errors.append("Missing culprit_correct field")
    
    # Check culprit minor errors
    if "minor_errors" not in culprit_data:
        errors.append("Missing culprit minor_errors section")
    else:
        expected_minor = expected_sections["culprit_minor_errors"]
        actual_minor = set(culprit_data["minor_errors"].keys())
        
        missing_minor = expected_minor - actual_minor
        extra_minor = actual_minor - expected_minor
        
        if missing_minor:
            errors.append(f"Missing culprit minor_errors fields: {missing_minor}")
        if extra_minor:
            errors.append(f"Unexpected culprit minor_errors fields: {extra_minor}")
    
    # Check culprit major errors
    if "major_errors" not in culprit_data:
        errors.append("Missing culprit major_errors section")
    else:
        expected_major = expected_sections["culprit_major_errors"]
        actual_major = set(culprit_data["major_errors"].keys())
        
        missing_major = expected_major - actual_major
        extra_major = actual_major - expected_major
        
        if missing_major:
            errors.append(f"Missing culprit major_errors fields: {missing_major}")
        if extra_major:
            errors.append(f"Unexpected culprit major_errors fields: {extra_major}")
    
    # Validate accomplice fields
    accomplice_data = parsed_response["accomplice"]
    
    # Check accomplice_correct (main assessment)
    if "accomplice_correct" not in accomplice_data:
        errors.append("Missing accomplice_correct field")
    
    # Check accomplice minor errors
    if "minor_errors" not in accomplice_data:
        errors.append("Missing accomplice minor_errors section")
    else:
        expected_minor = expected_sections["accomplice_minor_errors"]
        actual_minor = set(accomplice_data["minor_errors"].keys())
        
        missing_minor = expected_minor - actual_minor
        extra_minor = actual_minor - expected_minor
        
        if missing_minor:
            errors.append(f"Missing accomplice minor_errors fields: {missing_minor}")
        if extra_minor:
            errors.append(f"Unexpected accomplice minor_errors fields: {extra_minor}")
    
    # Check accomplice major errors
    if "major_errors" not in accomplice_data:
        errors.append("Missing accomplice major_errors section")
    else:
        expected_major = expected_sections["accomplice_major_errors"]
        actual_major = set(accomplice_data["major_errors"].keys())
        
        missing_major = expected_major - actual_major
        extra_major = actual_major - expected_major
        
        if missing_major:
            errors.append(f"Missing accomplice major_errors fields: {missing_major}")
        if extra_major:
            errors.append(f"Unexpected accomplice major_errors fields: {extra_major}")
    
    if errors:
        error_msg = "Scoring response validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Scoring response validation passed - all expected fields present")


def parse_single_category_response(response: str, prompt_name: str) -> Dict[str, Any]:
    """
    Parse LLM response for a single category (culprit or accomplice only).
    
    Args:
        response: Raw LLM response
        prompt_name: Name of the prompt used (to determine category)
        
    Returns:
        Dictionary with parsed assessment for the single category
    """
    import re
    import json
    
    try:
        # Determine category from prompt name
        if "culprit" in prompt_name.lower():
            category = "culprit"
            main_field = "culprit_correct"
        elif "accomplice" in prompt_name.lower():
            category = "accomplice"
            main_field = "accomplice_correct"
        else:
            raise ValueError(f"Cannot determine category from prompt name: {prompt_name}")
        
        # Initialize result structure
        result = {
            category: {}
        }
        
        # Define section patterns for single category
        # Use more flexible regex to capture multi-line JSON objects
        # Try multiple patterns to be more robust
        if category == "culprit":
            sections = {
                # Primary patterns with exact headers
                r"CULPRIT ASSESSMENT:\s*(?:```json\s*)?(\{.*?\})(?:\s*```)?": "assessment",
                r"CULPRIT MINOR ERRORS:\s*(?:```json\s*)?(\{.*?\})(?:\s*```)?": "minor_errors", 
                r"CULPRIT MAJOR ERRORS:\s*(?:```json\s*)?(\{.*?\})(?:\s*```)?": "major_errors",
                # Fallback patterns with variations
                r"(?:CULPRIT\s+)?ASSESSMENT:\s*(?:```json\s*)?(\{.*?\})(?:\s*```)?": "assessment",
                r"(?:CULPRIT\s+)?MINOR ERRORS:\s*(?:```json\s*)?(\{.*?\})(?:\s*```)?": "minor_errors",
                r"(?:CULPRIT\s+)?MAJOR ERRORS:\s*(?:```json\s*)?(\{.*?\})(?:\s*```)?": "major_errors",
                # Even more flexible - just look for JSON with expected fields
                r'(\{[^}]*"culprit_correct"[^}]*\})': "assessment",
                r'(\{[^}]*"culprit_missing_or_wrong_alias"[^}]*\})': "minor_errors",
                r'(\{[^}]*"different_suspect_not_accomplice"[^}]*\})': "major_errors"
            }
        else:  # accomplice
            sections = {
                # Primary patterns with exact headers
                r"ACCOMPLICE ASSESSMENT:\s*(?:```json\s*)?(\{.*?\})(?:\s*```)?": "assessment",
                r"ACCOMPLICE MINOR ERRORS:\s*(?:```json\s*)?(\{.*?\})(?:\s*```)?": "minor_errors",
                r"ACCOMPLICE MAJOR ERRORS:\s*(?:```json\s*)?(\{.*?\})(?:\s*```)?": "major_errors",
                # Fallback patterns with variations
                r"(?:ACCOMPLICE\s+)?ASSESSMENT:\s*(?:```json\s*)?(\{.*?\})(?:\s*```)?": "assessment",
                r"(?:ACCOMPLICE\s+)?MINOR ERRORS:\s*(?:```json\s*)?(\{.*?\})(?:\s*```)?": "minor_errors",
                r"(?:ACCOMPLICE\s+)?MAJOR ERRORS:\s*(?:```json\s*)?(\{.*?\})(?:\s*```)?": "major_errors",
                # Even more flexible - just look for JSON with expected fields
                r'(\{[^}]*"accomplice_correct"[^}]*\})': "assessment",
                r'(\{[^}]*"accomplice_missing_or_wrong_alias"[^}]*\})': "minor_errors",
                r'(\{[^}]*"different_suspect_not_culprit"[^}]*\})': "major_errors"
            }
        
        sections_found = 0
        
        # Extract each section
        for pattern, section_name in sections.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    json_str = match.group(1).strip()
                    #####
                    # Remove comments from JSON (LLM sometimes includes them despite instructions)
                    json_str = re.sub(r'#.*?(?=\n|$)', '', json_str)  # Remove # comments
                    json_str = re.sub(r'//.*?(?=\n|$)', '', json_str)  # Remove // comments
                    # Clean up any extra whitespace/commas that might be left
                    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas before }
                    json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas before ]
                    #####
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
            logger.warning(f"Could not find any JSON sections in {category} response")
            return None
        
        logger.info(f"Successfully parsed {sections_found} sections from {category} response")
        
        # Normalize field names (replace hyphens with underscores)
        result = normalize_field_names(result)
        
        # Validate that all expected fields are present for this category
        validate_single_category_fields(result, category)
        
        return result
        
    except Exception as e:
        logger.warning(f"Error parsing {category} response: {e}")
        return None


def validate_single_category_fields(parsed_response: Dict[str, Any], category: str) -> None:
    """
    Validate that a single category response contains all expected fields.
    
    Args:
        parsed_response: Parsed response for single category
        category: Either "culprit" or "accomplice"
        
    Raises:
        ValueError: If any expected fields are missing or unexpected fields are present
    """
    if category == "culprit":
        expected_main = {"culprit_correct"}
        expected_minor = {
            "culprit_missing_or_wrong_alias",
            "hallucinated_part_of_name", 
            "missing_part_of_name",
            "included_accomplice"
        }
        expected_major = {
            "different_suspect_not_accomplice",
            "confused_swapped_culprit_and_accomplice",
            "missing_real_name_only_has_alias",
            "included_other_non_accomplice_suspects"
        }
    else:  # accomplice
        expected_main = {"accomplice_correct"}
        expected_minor = {
            "accomplice_missing_or_wrong_alias",
            "hallucinated_part_of_name",
            "missing_part_of_name", 
            "included_culprit"
        }
        expected_major = {
            "different_suspect_not_culprit",
            "confused_swapped_accomplice_and_culprit",
            "missing_real_name_only_has_alias",
            "included_other_non_culprit_suspects"
        }
    
    errors = []
    
    if category not in parsed_response:
        errors.append(f"Missing {category} category in parsed response")
        
    if errors:
        error_msg = f"Single category validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    category_data = parsed_response[category]
    
    # Check main assessment fields
    category_main_fields = {k for k in category_data.keys() if k not in ["minor_errors", "major_errors"]}
    
    missing_main = expected_main - category_main_fields
    extra_main = category_main_fields - expected_main
    
    if missing_main:
        errors.append(f"Missing {category} main fields: {missing_main}")
    if extra_main:
        errors.append(f"Unexpected {category} main fields: {extra_main}")
    
    # Check minor and major error fields
    for error_type, expected_fields in [("minor_errors", expected_minor), ("major_errors", expected_major)]:
        if error_type not in category_data:
            errors.append(f"Missing {category} {error_type} section")
            continue
            
        actual_fields = set(category_data[error_type].keys())
        
        missing_fields = expected_fields - actual_fields
        extra_fields = actual_fields - expected_fields
        
        if missing_fields:
            errors.append(f"Missing {category} {error_type} fields: {missing_fields}")
        if extra_fields:
            errors.append(f"Unexpected {category} {error_type} fields: {extra_fields}")
    
    if errors:
        error_msg = f"Single category validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Single category validation passed for {category}")


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
        
        # Normalize field names (replace hyphens with underscores)
        result = normalize_field_names(result)
        
        # Validate that all expected fields are present
        validate_scoring_response_fields(result)
        
        return result
        
    except Exception as e:
        logger.warning(f"Error parsing scoring response: {e}")
        return None


def score_whodunit_solution_two_calls(
    llm_solution: Dict[str, str],
    ground_truth: Dict[str, Any],
    range_spec: str = "all",
    scoring_model: str = "gpt-5",
    max_completion_tokens: int = 100000,
    ask_user_confirmation: bool = False
) -> Dict[str, Any]:
    """
    Score a whodunit solution using two separate LLM calls - one for culprits, one for accomplices.
    
    Args:
        llm_solution: Parsed LLM solution
        ground_truth: Ground truth data
        range_spec: Range specification used for text selection
        scoring_model: LLM model for scoring
        max_completion_tokens: Max tokens for scoring
        ask_user_confirmation: Whether to ask for confirmation
        
    Returns:
        Dictionary with scoring results from both calls
    """
    from ius.utils import call_llm
    
    # Simple ground truth selection: use post-reveal if range includes the end and it exists
    if range_spec in ["last", "all"] and ground_truth.get("culprits_post_reveal"):
        selected_culprits = ground_truth.get("culprits_post_reveal")
    else:
        selected_culprits = ground_truth.get("culprits")
    
    # Prepare template variables (same for both calls) - use ground truth suspects from Google Sheets
    template_vars = {
        "suspects": ground_truth.get("suspects", "None"),  # Use ground truth suspects instead of LLM suspects
        "ground_truth_culprits": selected_culprits or "None",
        "ground_truth_accomplices": ground_truth.get("accomplices", "None"),
        "llm_main_culprits": llm_solution.get("main_culprits", "None"),
        "llm_accomplices": llm_solution.get("accomplices", "None")
    }
    
    logger.info("Making two separate LLM calls for culprit and accomplice scoring...")
    
    # Call 1: Score culprits
    logger.info("Making LLM call for culprit scoring...")
    culprit_result = _score_single_category(
        "whodunit-scoring-culprits", 
        template_vars, 
        scoring_model, 
        max_completion_tokens, 
        ask_user_confirmation
    )
    
    # Call 2: Score accomplices
    logger.info("Making LLM call for accomplice scoring...")
    accomplice_result = _score_single_category(
        "whodunit-scoring-accomplices", 
        template_vars, 
        scoring_model, 
        max_completion_tokens, 
        ask_user_confirmation
    )

    # Check if either call failed parsing (but preserve raw responses for debugging)
    culprit_failed = culprit_result.get("error") is not None
    accomplice_failed = accomplice_result.get("error") is not None
    
    if culprit_failed or accomplice_failed:
        # Return error but preserve raw responses for debugging
        error_msg = []
        if culprit_failed:
            error_msg.append(f"Culprit: {culprit_result['error']}")
        if accomplice_failed:
            error_msg.append(f"Accomplice: {accomplice_result['error']}")
        
        return {
            "assessment": None,
            "raw_response": {
                "culprit_raw_response": culprit_result.get("raw_response"),
                "accomplice_raw_response": accomplice_result.get("raw_response")
            },
            "usage": {
                "input_tokens": culprit_result["usage"]["input_tokens"] + accomplice_result["usage"]["input_tokens"],
                "output_tokens": culprit_result["usage"]["output_tokens"] + accomplice_result["usage"]["output_tokens"],
                "total_tokens": culprit_result["usage"]["total_tokens"] + accomplice_result["usage"]["total_tokens"],
                "input_cost": culprit_result["usage"]["input_cost"] + accomplice_result["usage"]["input_cost"],
                "output_cost": culprit_result["usage"]["output_cost"] + accomplice_result["usage"]["output_cost"],
                "total_cost": culprit_result["usage"]["total_cost"] + accomplice_result["usage"]["total_cost"]
            },
            "error": "Two-call scoring failed: " + "; ".join(error_msg),
            "final_prompts_used": {
                "culprit_prompts": culprit_result.get("final_prompts_used", {}),
                "accomplice_prompts": accomplice_result.get("final_prompts_used", {})
            },
            "finish_reason": f"culprit: {culprit_result.get('finish_reason', 'unknown')}, accomplice: {accomplice_result.get('finish_reason', 'unknown')}"
        }
    
    # Combine results - extract the nested category data to avoid double nesting
    combined_assessment = {}
    
    # Extract culprit data (culprit_result["assessment"] is {"culprit": {...}})
    if culprit_result["assessment"] and "culprit" in culprit_result["assessment"]:
        combined_assessment["culprit"] = culprit_result["assessment"]["culprit"]
    
    # Extract accomplice data (accomplice_result["assessment"] is {"accomplice": {...}})
    if accomplice_result["assessment"] and "accomplice" in accomplice_result["assessment"]:
        combined_assessment["accomplice"] = accomplice_result["assessment"]["accomplice"]
    
    # Combine metadata
    combined_usage = {
        "input_tokens": culprit_result["usage"]["input_tokens"] + accomplice_result["usage"]["input_tokens"],
        "output_tokens": culprit_result["usage"]["output_tokens"] + accomplice_result["usage"]["output_tokens"],
        "total_tokens": culprit_result["usage"]["total_tokens"] + accomplice_result["usage"]["total_tokens"],
        "input_cost": culprit_result["usage"]["input_cost"] + accomplice_result["usage"]["input_cost"],
        "output_cost": culprit_result["usage"]["output_cost"] + accomplice_result["usage"]["output_cost"],
        "total_cost": culprit_result["usage"]["total_cost"] + accomplice_result["usage"]["total_cost"]
    }
    
    return {
        "assessment": combined_assessment,
        "raw_response": {
            "culprit_raw_response": culprit_result['raw_response'],
            "accomplice_raw_response": accomplice_result['raw_response']
        },
        "usage": combined_usage,
        "error": None,  # No error for successful results
        "final_prompts_used": {
            "culprit_prompts": culprit_result.get("final_prompts_used", {}),
            "accomplice_prompts": accomplice_result.get("final_prompts_used", {})
        },
        "finish_reason": f"culprit: {culprit_result.get('finish_reason', 'unknown')}, accomplice: {accomplice_result.get('finish_reason', 'unknown')}"
    }


def _score_single_category(
    prompt_name: str,
    template_vars: Dict[str, str],
    scoring_model: str,
    max_completion_tokens: int,
    ask_user_confirmation: bool
) -> Dict[str, Any]:
    """
    Score a single category (culprit or accomplice) using LLM.
    
    Args:
        prompt_name: Name of the prompt directory (e.g., "whodunit-scoring-culprits")
        template_vars: Template variables for the prompt
        scoring_model: LLM model for scoring
        max_completion_tokens: Max tokens for scoring
        ask_user_confirmation: Whether to ask for confirmation
        
    Returns:
        Dictionary with scoring results for this category
    """
    from ius.utils import call_llm
    
    # Load prompts for this category
    prompt_dir = Path(f"prompts/extrinsic-eval/{prompt_name}")
    
    try:
        system_prompt, user_prompt_template = load_prompts(prompt_dir)
    except Exception as e:
        logger.error(f"Could not load {prompt_name} prompts: {e}")
        raise

    # Make LLM call
    scoring_result = call_llm(
        text="",  # Not used for this prompt style
        system_and_user_prompt={
            "system": system_prompt,
            "user": user_prompt_template
        },
        template_vars=template_vars,
        model=scoring_model,
        temperature=1.0,  # gpt-5 only supports temperature=1.0
        max_completion_tokens=max_completion_tokens,
        ask_user_confirmation=ask_user_confirmation
    )
    
    if not scoring_result:
        raise ValueError(f"LLM call for {prompt_name} returned None")
    
    # Parse the response (single category)
    parsed_assessment = parse_single_category_response(scoring_result["response"], prompt_name)

    if not parsed_assessment:
        # Return the raw response even when parsing fails for debugging
        return {
            "assessment": None,
            "raw_response": scoring_result["response"],
            "usage": scoring_result["usage"],
            "final_prompts_used": scoring_result.get("final_prompts_used", {}),
            "finish_reason": scoring_result.get("finish_reason", "unknown"),
            "error": f"Could not parse {prompt_name} response"
        }
    
    return {
        "assessment": parsed_assessment,
        "raw_response": scoring_result["response"],
        "usage": scoring_result["usage"],
        "final_prompts_used": scoring_result.get("final_prompts_used", {}),
        "finish_reason": scoring_result.get("finish_reason", "unknown"),
        "error": None
    }


def score_whodunit_solution(
    llm_solution: Dict[str, str],
    ground_truth: Dict[str, Any],
    scoring_prompt_name: str,
    range_spec: str = "all",
    scoring_model: str = "gpt-5",
    max_completion_tokens: int = 100000,
    ask_user_confirmation: bool = False,
    input_path: str = None
) -> Dict[str, Any]:
    """
    Score a whodunit solution against ground truth using LLM.
    
    Args:
        llm_solution: Parsed LLM solution
        ground_truth: Ground truth data
        scoring_prompt_name: Prompt name - if "two-calls", uses separate culprit/accomplice calls
        range_spec: Range specification used for text selection
        scoring_model: LLM model for scoring
        max_completion_tokens: Max tokens for scoring
        ask_user_confirmation: Whether to ask for confirmation
        input_path: Input path to detect dataset type (for non-BMDS handling)
        
    Returns:
        Dictionary with scoring results
    """
    # Detect if this is a non-BMDS dataset
    is_bmds = False
    if input_path:
        path_lower = input_path.lower()
        is_bmds = "/bmds" in path_lower or "bmds_" in path_lower
    
    # For non-BMDS datasets, force culprits-only scoring

    if not is_bmds:
        logger.info("Non-BMDS dataset detected - using culprits-only scoring")
        score_output = _score_single_category(
            prompt_name="whodunit-scoring-culprits",
            template_vars={
                "suspects": ground_truth.get("suspects", "None"),
                "ground_truth_culprits": ground_truth.get("culprits") or "None",
                "ground_truth_accomplices": "None",  # Not used for culprits-only
                "llm_main_culprits": llm_solution.get("main_culprits", "None"),
                "llm_accomplices": "None"  # Not used for culprits-only
            },
            scoring_model=scoring_model,
            max_completion_tokens=max_completion_tokens,
            ask_user_confirmation=ask_user_confirmation
        )
        return score_output
    
    # Check if we should use the two-call approach (BMDS only)
    if scoring_prompt_name == "two-calls":
        logger.info("Using two-call scoring approach (separate culprit and accomplice calls)")
        return score_whodunit_solution_two_calls(
            llm_solution=llm_solution,
            ground_truth=ground_truth,
            range_spec=range_spec,
            scoring_model=scoring_model,
            max_completion_tokens=max_completion_tokens,
            ask_user_confirmation=ask_user_confirmation
        )
    
    # Original single-call approach
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
    
    # Prepare template variables - use ground truth suspects from Google Sheets
    # NOTE the  LM generated ground truth suspects were not reliable and causing errors in scoring
    # so I used GPT-5 to fix those errors (removing extra suspects who were clearly wrong and fixing
    # wrong names)
    template_vars = {
        "suspects": ground_truth.get("suspects", "None"),  # Use ground truth suspects instead of LLM suspects
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
            temperature=1.0,  # gpt-5 only supports temperature=1.0
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
    model: str = "o3",
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
            "temperature": 0.1,#temperature,
            "max_completion_tokens": max_completion_tokens
        }
        output_hash = generate_output_hash(hash_params)
        output_dir = f"outputs/eval/extrinsic/{input_name}_whodunit_{output_hash}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    items_output_dir = output_path / "items"
    items_output_dir.mkdir(exist_ok=True)
    
    # Create initial collection metadata
    collection_metadata = {
        "evaluation_function": "run_whodunit_evaluation",
        "content_type": "whodunit_analysis",
        "input_type": input_type,
        "model": model,
        "prompt_name": prompt_name,
        "range_spec": range_spec,
        "temperature": 0.1,#temperature,
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
    }
    
    # If input is summaries, include summarization info from source collection
    summarization_model = None
    summarization_prompt_name = None
    if input_type == "summaries":
        source_collection_file = input_path / "collection.json"
        if source_collection_file.exists():
            try:
                with open(source_collection_file, 'r') as f:
                    source_data = json.load(f)
                
                # Extract summarization_info from source collection
                if "summarization_info" in source_data:
                    collection_metadata["summarization_info"] = source_data["summarization_info"]
                    logger.info(f"Added summarization_info from source collection: {source_collection_file}")
                    
                    # Also extract collection-level metadata for individual items
                    try:
                        source_collection_metadata = source_data["summarization_info"]["collection_metadata"]
                        summarization_model = source_collection_metadata.get("model")
                        summarization_prompt_name = source_collection_metadata.get("prompt_name")
                    except Exception as e:
                        logger.debug(f"Could not extract collection-level summarization metadata: {e}")
                else:
                    logger.warning(f"No summarization_info found in source collection: {source_collection_file}")
            except Exception as e:
                logger.warning(f"Failed to load summarization_info from {source_collection_file}: {e}")
        else:
            logger.warning(f"Source collection file not found: {source_collection_file}")
    
    collection_data = {
        "whodunit_evaluation_info": {
            "collection_metadata": collection_metadata,
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

            # Extract ground truth from original item
            # NOTE this is done here because phase 1 is not always executed.

            with open(item_file) as f:
                original_item_data = json.load(f)
            ground_truth = extract_ground_truth(original_item_data, input_dir)
            
            # Extract summarization metadata from item (for summaries input)
            item_optional_summary_length = None
            item_strategy_function = None
            item_summary_content_type = None
            item_step_k_inputs = None
            
            if input_type == "summaries":
                try:
                    documents = original_item_data.get("documents", [])
                    if documents:
                        item_metadata_dict = documents[0].get("metadata", {}).get("item_experiment_metadata", {})
                        
                        # From template_vars
                        template_vars = item_metadata_dict.get("template_vars", {})
                        item_optional_summary_length = template_vars.get("optional_summary_length")
                        
                        # From item_experiment_metadata
                        item_strategy_function = item_metadata_dict.get("strategy_function")
                        item_summary_content_type = item_metadata_dict.get("summary_content_type")
                        item_step_k_inputs = item_metadata_dict.get("step_k_inputs")
                except Exception as e:
                    logger.debug(f"Could not extract item-level summarization metadata for {item_id}: {e}")

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
                    temperature=0.1,#temperature,
                    max_completion_tokens=max_completion_tokens,
                    ask_user_confirmation=ask_user_confirmation
                )
                
                
                # Parse response
                parsed_sections = parse_llm_response(llm_result["response"])
                

                # Extract puzzle_data for true-detective metadata 
                puzzle_metadata = {}
                try:
                    # Detect if this is true-detective
                    dataset_name = input_dir.split("/")[-2] if "/" in input_dir else input_dir
                    if "true-detective" in dataset_name.lower():
                        with open(item_file) as f:
                            original_item_data = json.load(f)
                        
                        puzzle_data = original_item_data["documents"][0]["metadata"].get("original_metadata", {}).get("original_metadata", {}).get("puzzle_data", {})
                        if puzzle_data:
                            # Copy puzzle_data excluding mystery_text and outcome
                            puzzle_metadata = {k: v for k, v in puzzle_data.items() if k not in ["mystery_text", "outcome"]}
                            logger.info(f"✅ Added puzzle_data metadata for true-detective item {item_id}")
                except Exception as e:
                    logger.warning(f"Could not extract puzzle_data metadata for {item_id}: {e}")

                # Create result with solution_correctness_assessment set to None
                item_metadata = {
                    "item_id": item_id,
                    "input_type": input_type,
                    "selected_range": range_spec,
                    "selected_indices": selected_indices,
                    "selected_text_length": len(selected_text),
                    "total_chunks": len(segments),
                    "reveal_segment_length": len(reveal_segment),
                    "reveal_preview": reveal_preview,
                    "evaluation_timestamp": datetime.now().isoformat()
                }
                
                # Add summarization metadata if available (for summaries input)
                if input_type == "summaries":
                    item_metadata.update({
                        "optional_summary_length": item_optional_summary_length,
                        "strategy_function": item_strategy_function,
                        "summary_content_type": item_summary_content_type,
                        "step_k_inputs": item_step_k_inputs,
                        "summarization_model": summarization_model,
                        "summarization_prompt_name": summarization_prompt_name
                    })
                
                # Add puzzle_data for true-detective datasets
                if puzzle_metadata:
                    item_metadata["puzzle_data"] = puzzle_metadata
                
                item_result = {
                    "item_metadata": item_metadata,
                    "evaluation_metadata": {
                        "prompt_name": prompt_name,
                        "model": model,
                        "temperature": 0.1,#temperature,
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
                scoring_model = "gpt-5"

                # Load existing item
                with open(item_output_file) as f:
                    item_result = json.load(f)

                # Check if scoring is needed and scoring prompt is provided
                if scoring_prompt_name and (item_result.get("solution_correctness_assessment") is None or rescore):
                    # Check if ground truth is available for scoring
                    #ground_truth = item_result.get("ground_truth", {})
                    parsed_response = item_result.get("parsed_response")
                    
                    # If no ground truth in item result, try to extract it again (including Google Sheets fallback)
#                    if not ground_truth.get("culprits"):
#                        logger.info(f"🔄 No ground truth found in item result for {item_id}, trying to extract again...")
#                        with open(item_file) as f:
#                            original_item_data = json.load(f)
#                        ground_truth = extract_ground_truth(original_item_data, input_dir)
#                        # Update the item result with the newly extracted ground truth
#                        item_result["ground_truth"] = ground_truth
#                    
#                    if not ground_truth.get("culprits"):
#                        logger.info(f"⏭️  No ground truth available for {item_id} - skipping scoring (can be done later)")
#                    elif not parsed_response or not isinstance(parsed_response, dict):
#                        logger.warning(f"⏭️  No valid parsed response for {item_id} - skipping scoring")
#                    elif not parsed_response.get("suspects") or not parsed_response.get("main_culprits"):
#                        logger.info(f"⏭️  LLM solution incomplete for {item_id} (missing suspects or main_culprits) - skipping #scoring")
#                    else:
#                        if rescore and item_result.get("solution_correctness_assessment") is not None:
#                            logger.info(f"🔄 Re-scoring solution for {item_id}...")
#                        else:
#                            logger.info(f"📊 Scoring solution for {item_id}...")
                #if True:
                if scoring_prompt_name and (item_result.get("solution_correctness_assessment") is None or rescore):

                        # Score the solution
                        scoring_result = score_whodunit_solution(
                            llm_solution=item_result["parsed_response"],
                            ground_truth=ground_truth, #item_result["ground_truth"],
                            scoring_prompt_name=scoring_prompt_name,
                            range_spec=range_spec,
                            scoring_model=scoring_model,
                            max_completion_tokens=100000,
                            ask_user_confirmation=ask_user_confirmation,
                            input_path=input_dir
                        )
                        # Update item with scoring results
                        item_result["solution_correctness_assessment"] = scoring_result["assessment"]
                        
                        # Handle different raw_response formats (single-call vs two-call)
                        raw_response = scoring_result["raw_response"]
                        if isinstance(raw_response, dict):
                            # Two-call approach: separate responses
                            scoring_raw_response = raw_response
                        else:
                            # Single-call approach: string response
                            scoring_raw_response = raw_response
                        
                        # Handle different prompt formats (single-call vs two-call)
                        prompts_used = scoring_result.get("final_prompts_used", {})
                        if scoring_prompt_name == "two-calls":
                            # Two-call approach: already properly structured
                            scoring_prompts_used = prompts_used
                        else:
                            # Single-call approach: already properly structured
                            scoring_prompts_used = prompts_used

                        item_result["scoring_metadata"] = {
                            "scoring_model": scoring_model,
                            "scoring_raw_response": scoring_raw_response,
                            "scoring_finish_reason": scoring_result.get("finish_reason"),
                            "scoring_usage": scoring_result["usage"],
                            "scoring_error": scoring_result.get("error", None),
                            "scoring_timestamp": datetime.now().isoformat(),
                            "scoring_prompts_used": scoring_prompts_used
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