#!/usr/bin/env python3
"""
Standalone script to update whodunit evaluation JSON files with corrected assessments from Google Sheets.

This script reads corrected assessment data from a Google Sheet and updates the corresponding
JSON files in the outputs/eval/extrinsic directories with the corrected values for:
- culprit_correct, accomplice_correct
- All major and minor error fields for both culprits and accomplices

Usage:
    python update_whodunit_assessments_from_sheets.py [--dry-run] [--target-dir DIR]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Google Sheet configuration
SHEET_ID = "1awnPbTUjIfVOqqhd8vWXQm8iwPXRMXJ4D1-MWfwLNwM"
GID = "736907744"  # The specific sheet tab with the corrected data

# Fields to update from the Google Sheet
ASSESSMENT_FIELDS = [
    "culprit_correct",
    "accomplice_correct",
    "culprit_missing_or_wrong_alias",
    "culprit_hallucinated_part_of_name", 
    "culprit_missing_part_of_name",
    "culprit_included_accomplice",
    "culprit_different_suspect_not_accomplice",
    "culprit_confused_swapped_culprit_and_accomplice",
    "culprit_missing_real_name_only_has_alias",
    "culprit_included_other_non_accomplice_suspects",
    "accomplice_missing_or_wrong_alias",
    "accomplice_hallucinated_part_of_name",
    "accomplice_missing_part_of_name", 
    "accomplice_included_culprit",
    "accomplice_different_suspect_not_culprit",
    "accomplice_confused_swapped_accomplice_and_culprit",
    "accomplice_missing_real_name_only_has_alias",
    "accomplice_included_other_non_culprit_suspects"
]

# Mapping from Google Sheet field names to JSON field names
FIELD_MAPPING = {
    # Culprit assessments
    "culprit_correct": ("culprit", "culprit_correct"),
    
    # Culprit minor errors - these have mixed naming in JSON
    "culprit_missing_or_wrong_alias": ("culprit", "minor_errors", "culprit_missing_or_wrong_alias"),
    "culprit_hallucinated_part_of_name": ("culprit", "minor_errors", "hallucinated_part_of_name"),
    "culprit_missing_part_of_name": ("culprit", "minor_errors", "missing_part_of_name"),
    "culprit_included_accomplice": ("culprit", "minor_errors", "included_accomplice"),
    
    # Culprit major errors - these drop the culprit_ prefix in JSON
    "culprit_different_suspect_not_accomplice": ("culprit", "major_errors", "different_suspect_not_accomplice"),
    "culprit_confused_swapped_culprit_and_accomplice": ("culprit", "major_errors", "confused_swapped_culprit_and_accomplice"),
    "culprit_missing_real_name_only_has_alias": ("culprit", "major_errors", "missing_real_name_only_has_alias"),
    "culprit_included_other_non_accomplice_suspects": ("culprit", "major_errors", "included_other_non_accomplice_suspects"),
    
    # Accomplice assessments
    "accomplice_correct": ("accomplice", "accomplice_correct"),
    
    # Accomplice minor errors - these have mixed naming in JSON
    "accomplice_missing_or_wrong_alias": ("accomplice", "minor_errors", "accomplice_missing_or_wrong_alias"),
    "accomplice_hallucinated_part_of_name": ("accomplice", "minor_errors", "hallucinated_part_of_name"),
    "accomplice_missing_part_of_name": ("accomplice", "minor_errors", "missing_part_of_name"),
    "accomplice_included_culprit": ("accomplice", "minor_errors", "included_culprit"),
    
    # Accomplice major errors - these drop the accomplice_ prefix in JSON
    "accomplice_different_suspect_not_culprit": ("accomplice", "major_errors", "different_suspect_not_culprit"),
    "accomplice_confused_swapped_accomplice_and_culprit": ("accomplice", "major_errors", "confused_swapped_accomplice_and_culprit"),
    "accomplice_missing_real_name_only_has_alias": ("accomplice", "major_errors", "missing_real_name_only_has_alias"),
    "accomplice_included_other_non_culprit_suspects": ("accomplice", "major_errors", "included_other_non_culprit_suspects"),
}


def load_corrected_assessments() -> pd.DataFrame:
    """
    Load corrected assessment data from Google Sheets.
    
    Returns:
        DataFrame with corrected assessment data
    """
    csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
    
    logger.info(f"Loading corrected assessments from Google Sheets...")
    logger.debug(f"CSV URL: {csv_url}")
    
    try:
        # Read the sheet, preserving string values
        df = pd.read_csv(csv_url, keep_default_na=False, na_values=[''])
        logger.info(f"✅ Loaded {len(df)} rows from Google Sheets")
        
        # Log the columns we found
        logger.debug(f"Available columns: {list(df.columns)}")
        
        # Check if we have the required columns
        missing_cols = [col for col in ['item_id'] + ASSESSMENT_FIELDS if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in Google Sheet: {missing_cols}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data from Google Sheets: {e}")
        raise


def find_whodunit_json_files(target_dir: Optional[str] = None) -> List[Path]:
    """
    Find all whodunit evaluation JSON files.
    
    Args:
        target_dir: Optional specific directory to search in
        
    Returns:
        List of paths to JSON files
    """
    # SAFETY CHECK: Only allow the specific directory
    ALLOWED_DIR = "outputs/eval/extrinsic/bmds_fixed_size2_8000_whodunit_7fdb57"
    
    if target_dir:
        # Normalize the path to handle different formats
        normalized_target = str(Path(target_dir).resolve())
        normalized_allowed = str(Path(ALLOWED_DIR).resolve())
        
        if normalized_target != normalized_allowed:
            raise ValueError(f"SAFETY ERROR: This script is only allowed to modify {ALLOWED_DIR}. "
                           f"Attempted to access: {target_dir}")
        
        search_path = Path(target_dir)
        if not search_path.exists():
            logger.error(f"Target directory does not exist: {target_dir}")
            return []
    else:
        # Default to the allowed directory only
        search_path = Path(ALLOWED_DIR)
        if not search_path.exists():
            logger.error(f"Default directory does not exist: {ALLOWED_DIR}")
            return []
    
    json_files = []
    
    # Find all JSON files in items/ subdirectories
    for json_file in search_path.rglob("items/*.json"):
        json_files.append(json_file)
    
    logger.info(f"Found {len(json_files)} JSON files to potentially update")
    return json_files


def update_json_file(json_path: Path, corrections_df: pd.DataFrame, dry_run: bool = False) -> bool:
    """
    Update a single JSON file with corrected assessments.
    
    Args:
        json_path: Path to the JSON file
        corrections_df: DataFrame with corrected assessments
        dry_run: If True, don't actually write changes
        
    Returns:
        True if file was updated, False otherwise
    """
    try:
        # Load the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract item_id from the JSON
        item_id = None
        if "item_metadata" in data and "item_id" in data["item_metadata"]:
            item_id = data["item_metadata"]["item_id"]
        elif "documents" in data and data["documents"]:
            item_id = data["documents"][0]["metadata"].get("id")
        else:
            item_id = data.get("id")
        
        if not item_id:
            logger.warning(f"Could not extract item_id from {json_path}")
            return False
        
        # Look up corrections for this item
        corrections = corrections_df[corrections_df['item_id'] == item_id]
        if corrections.empty:
            logger.debug(f"No corrections found for item {item_id}")
            return False
        
        correction_row = corrections.iloc[0]
        
        # Update the assessment fields using the field mapping
        updated_fields = []
        
        if "solution_correctness_assessment" not in data or not data["solution_correctness_assessment"]:
            logger.warning(f"No solution_correctness_assessment found in {item_id}")
            return False
            
        assessment = data["solution_correctness_assessment"]
        
        # Process each field in the mapping
        for sheet_field, json_path_tuple in FIELD_MAPPING.items():
            if sheet_field in correction_row and pd.notna(correction_row[sheet_field]):
                new_value = correction_row[sheet_field]
                
                # Navigate to the correct location in the JSON
                current_location = assessment
                json_field_path = []
                
                try:
                    # Navigate through the nested structure
                    for path_element in json_path_tuple[:-1]:  # All but the last element
                        if path_element not in current_location:
                            logger.warning(f"Path element '{path_element}' not found in {item_id}")
                            break
                        current_location = current_location[path_element]
                        json_field_path.append(path_element)
                    else:
                        # Get the final field name
                        final_field = json_path_tuple[-1]
                        json_field_path.append(final_field)
                        
                        # Get the current value
                        old_value = current_location.get(final_field)
                        
                        # Only update if values are different
                        if old_value != new_value:
                            current_location[final_field] = new_value
                            field_path_str = " -> ".join(json_field_path)
                            updated_fields.append(f"{field_path_str}: {old_value} → {new_value}")
                            
                except Exception as e:
                    logger.error(f"Error navigating to field {sheet_field} in {item_id}: {e}")
                    continue
        
        if updated_fields:
            if not dry_run:
                # Write the updated JSON back to file
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"\n✅ UPDATED {item_id}:")
                for field_change in updated_fields:
                    print(f"   • {field_change}")
            else:
                print(f"\n🔍 [DRY RUN] Would update {item_id}:")
                for field_change in updated_fields:
                    print(f"   • {field_change}")
            return True
        else:
            logger.debug(f"No updates needed for {item_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating {json_path}: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Update whodunit evaluation JSON files with corrected assessments from Google Sheets"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--target-dir",
        help="Specific directory to search for JSON files (default: outputs/eval/extrinsic)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load corrected assessments from Google Sheets
        corrections_df = load_corrected_assessments()
        
        if corrections_df.empty:
            logger.error("No data loaded from Google Sheets")
            return 1
        
        # Find JSON files to update
        json_files = find_whodunit_json_files(args.target_dir)
        
        if not json_files:
            logger.error("No JSON files found to update")
            return 1
        
        # Update each JSON file
        updated_count = 0
        print(f"\n{'='*60}")
        print(f"PROCESSING {len(json_files)} JSON FILES")
        print(f"{'='*60}")
        
        for json_file in json_files:
            if update_json_file(json_file, corrections_df, args.dry_run):
                updated_count += 1
        
        print(f"\n{'='*60}")
        if args.dry_run:
            print(f"🔍 DRY RUN SUMMARY: Would update {updated_count} out of {len(json_files)} files")
        else:
            print(f"✅ FINAL SUMMARY: Updated {updated_count} out of {len(json_files)} files")
        print(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())