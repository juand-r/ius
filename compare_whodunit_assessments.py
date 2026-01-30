#!/usr/bin/env python3
"""
Compare whodunit assessment results between two directories to identify differences.

This script compares the solution_correctness_assessment fields between the original
and FIXED evaluation results to identify where assessments differ.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

def load_assessment_data(directory: Path) -> Dict[str, Dict]:
    """Load assessment data from all JSON files in a directory."""
    assessments = {}
    items_dir = directory / "items"
    
    if not items_dir.exists():
        print(f"Warning: Items directory not found: {items_dir}")
        return assessments
    
    for json_file in items_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            item_id = json_file.stem
            assessment = data.get('solution_correctness_assessment', {})
            
            if assessment:
                assessments[item_id] = assessment
            else:
                print(f"Warning: No assessment found for {item_id}")
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return assessments

def flatten_assessment(assessment: Dict) -> Dict[str, str]:
    """Flatten nested assessment structure for easier comparison."""
    flattened = {}
    
    # Culprit assessments - handle double nesting
    if 'culprit' in assessment:
        culprit_outer = assessment['culprit']
        # Check if there's a nested 'culprit' key (new structure)
        if 'culprit' in culprit_outer:
            culprit = culprit_outer['culprit']
        else:
            culprit = culprit_outer
            
        flattened['culprit_correct'] = culprit.get('culprit_correct', '')
        
        # Minor errors
        if 'minor_errors' in culprit:
            for key, value in culprit['minor_errors'].items():
                flattened[f'culprit_minor_{key}'] = value
        
        # Major errors
        if 'major_errors' in culprit:
            for key, value in culprit['major_errors'].items():
                flattened[f'culprit_major_{key}'] = value
    
    # Accomplice assessments - handle double nesting
    if 'accomplice' in assessment:
        accomplice_outer = assessment['accomplice']
        # Check if there's a nested 'accomplice' key (new structure)
        if 'accomplice' in accomplice_outer:
            accomplice = accomplice_outer['accomplice']
        else:
            accomplice = accomplice_outer
            
        flattened['accomplice_correct'] = accomplice.get('accomplice_correct', '')
        
        # Minor errors
        if 'minor_errors' in accomplice:
            for key, value in accomplice['minor_errors'].items():
                flattened[f'accomplice_minor_{key}'] = value
        
        # Major errors
        if 'major_errors' in accomplice:
            for key, value in accomplice['major_errors'].items():
                flattened[f'accomplice_major_{key}'] = value
    
    return flattened

def compare_assessments(original_dir: Path, fixed_dir: Path) -> None:
    """Compare assessments between two directories."""
    
    print(f"Comparing assessments:")
    print(f"  Original: {original_dir}")
    print(f"  Fixed:    {fixed_dir}")
    print("=" * 80)
    
    # Load data from both directories
    original_data = load_assessment_data(original_dir)
    fixed_data = load_assessment_data(fixed_dir)
    
    print(f"Loaded {len(original_data)} items from original directory")
    print(f"Loaded {len(fixed_data)} items from fixed directory")
    
    # Find common items
    common_items = set(original_data.keys()) & set(fixed_data.keys())
    only_original = set(original_data.keys()) - set(fixed_data.keys())
    only_fixed = set(fixed_data.keys()) - set(original_data.keys())
    
    if only_original:
        print(f"\\nItems only in original: {sorted(only_original)}")
    if only_fixed:
        print(f"\\nItems only in fixed: {sorted(only_fixed)}")
    
    print(f"\\nComparing {len(common_items)} common items...")
    print("=" * 80)
    
    # Compare each common item and collect all differences
    differences_found = 0
    total_field_differences = 0
    all_culprit_diffs = []
    all_accomplice_diffs = []
    
    for item_id in sorted(common_items):
        original_flat = flatten_assessment(original_data[item_id])
        fixed_flat = flatten_assessment(fixed_data[item_id])
        
        # Find differences
        all_fields = set(original_flat.keys()) | set(fixed_flat.keys())
        item_differences = []
        
        for field in sorted(all_fields):
            original_value = original_flat.get(field, 'MISSING')
            fixed_value = fixed_flat.get(field, 'MISSING')
            
            # Normalize N/A and No to be equivalent
            def normalize_value(val):
                if val in ['N/A', 'No']:
                    return 'No/N/A'
                return val
            
            normalized_original = normalize_value(original_value)
            normalized_fixed = normalize_value(fixed_value)
            
            if normalized_original != normalized_fixed:
                item_differences.append((field, original_value, fixed_value))
        
        if item_differences:
            differences_found += 1
            total_field_differences += len(item_differences)
            
            # Separate culprit and accomplice differences
            culprit_diffs = [(item_id, field, orig, fixed) for field, orig, fixed in item_differences if field.startswith('culprit_')]
            accomplice_diffs = [(item_id, field, orig, fixed) for field, orig, fixed in item_differences if field.startswith('accomplice_')]
            
            all_culprit_diffs.extend(culprit_diffs)
            all_accomplice_diffs.extend(accomplice_diffs)
    
    # Display ALL culprit differences first
    if all_culprit_diffs:
        print("\\n" + "=" * 80)
        print("👤 ALL CULPRIT DIFFERENCES")
        print("=" * 80)
        
        current_item = None
        for item_id, field, orig_val, fixed_val in all_culprit_diffs:
            if item_id != current_item:
                print(f"\\n📋 ITEM: {item_id}")
                print("-" * 40)
                current_item = item_id
            
            clean_field = field.replace('culprit_', '')
            print(f"  🔄 {clean_field}:")
            print(f"     Original: {orig_val}")
            print(f"     Fixed:    {fixed_val}")
    
    # Display ALL accomplice differences second
    # if all_accomplice_diffs:
    #     print("\\n" + "=" * 80)
    #     print("👥 ALL ACCOMPLICE DIFFERENCES")
    #     print("=" * 80)
    #     
    #     current_item = None
    #     for item_id, field, orig_val, fixed_val in all_accomplice_diffs:
    #         if item_id != current_item:
    #             print(f"\\n📋 ITEM: {item_id}")
    #             print("-" * 40)
    #             current_item = item_id
    #         
    #         clean_field = field.replace('accomplice_', '')
    #         print(f"  🔄 {clean_field}:")
    #         print(f"     Original: {orig_val}")
    #         print(f"     Fixed:    {fixed_val}")
    
    # Summary
    print("\\n" + "=" * 80)
    print("📊 SUMMARY:")
    print(f"  • Total items compared: {len(common_items)}")
    print(f"  • Items with differences: {differences_found}")
    print(f"  • Total field differences: {total_field_differences}")
    
    if differences_found == 0:
        print("  ✅ No differences found - assessments are identical!")
    else:
        accuracy_rate = (len(common_items) - differences_found) / len(common_items) * 100
        print(f"  • Agreement rate: {accuracy_rate:.1f}%")
        print(f"  • Avg differences per differing item: {total_field_differences/differences_found:.1f}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare whodunit assessment results between two directories")
    parser.add_argument("--original", 
                       default="outputs/eval/extrinsic/bmds_fixed_size2_8000_whodunit_7fdb57",
                       help="Path to original evaluation directory")
    parser.add_argument("--fixed", 
                       default="outputs/eval/extrinsic/bmds_fixed_size2_8000_whodunit_7fdb57FIXED",
                       help="Path to fixed evaluation directory")
    
    args = parser.parse_args()
    
    original_path = Path(args.original)
    fixed_path = Path(args.fixed)
    
    if not original_path.exists():
        print(f"Error: Original directory does not exist: {original_path}")
        return 1
    
    if not fixed_path.exists():
        print(f"Error: Fixed directory does not exist: {fixed_path}")
        return 1
    
    compare_assessments(original_path, fixed_path)
    return 0

if __name__ == "__main__":
    exit(main())