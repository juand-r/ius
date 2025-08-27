#!/usr/bin/env python3
"""
Entity Coverage Items Counter

Counts the number of directories in items/ subdirectory for each 
true-detective entity-coverage evaluation directory.
"""

import os
from pathlib import Path
import sys

def count_items_directories(base_path: str = "outputs/eval/intrinsic/entity-coverage") -> None:
    """
    Count the number of directories in items/ for each true-detective entity-coverage directory.
    
    Args:
        base_path: Path to the entity-coverage evaluation results directory
    """
    
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_path}")
        return
    
    # Find all directories starting with "true-detective"
    true_detective_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("true-detective"):
            true_detective_dirs.append(item)
    
    if not true_detective_dirs:
        print(f"No directories starting with 'true-detective' found in {base_path}")
        return
    
    # Sort directories by name for consistent output
    true_detective_dirs.sort(key=lambda x: x.name)
    
    print("Entity Coverage Items Directory and JSON File Count")
    print("=" * 70)
    print(f"Base directory: {base_path}")
    print()
    
    total_directories = 0
    total_json_files = 0
    
    for eval_dir in true_detective_dirs:
        items_dir = eval_dir / "items"
        
        if not items_dir.exists():
            print(f"{eval_dir.name:60} | No items/ directory")
            continue
        
        if not items_dir.is_dir():
            print(f"{eval_dir.name:60} | items/ is not a directory")
            continue
        
        # Count subdirectories in items/
        subdirs = [item for item in items_dir.iterdir() if item.is_dir()]
        dir_count = len(subdirs)
        total_directories += dir_count
        
        # Count JSON files in all subdirectories
        json_count = 0
        for subdir in subdirs:
            json_files = list(subdir.glob("*.json"))
            json_count += len(json_files)
        
        total_json_files += json_count
        print(f"{eval_dir.name:60} | {dir_count:3d} dirs | {json_count:4d} JSONs")
    
    print("-" * 80)
    print(f"{'Total directories across all true-detective evaluations':60} | {total_directories:3d}")
    print(f"{'Total JSON files across all true-detective evaluations':60} | {total_json_files:3d}")
    print(f"{'Number of true-detective evaluation directories':60} | {len(true_detective_dirs):3d}")
    
    if len(true_detective_dirs) > 0:
        avg_dirs = total_directories / len(true_detective_dirs)
        avg_jsons = total_json_files / len(true_detective_dirs)
        print(f"{'Average directories per evaluation':60} | {avg_dirs:6.1f}")
        print(f"{'Average JSON files per evaluation':60} | {avg_jsons:6.1f}")

def main():
    """Main function."""
    
    if len(sys.argv) > 2:
        print("Usage: python count_entity_coverage_items.py [base_path]")
        print("Example: python count_entity_coverage_items.py outputs/eval/intrinsic/entity-coverage")
        sys.exit(1)
    
    base_path = sys.argv[1] if len(sys.argv) == 2 else "outputs/eval/intrinsic/entity-coverage"
    
    try:
        count_items_directories(base_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()