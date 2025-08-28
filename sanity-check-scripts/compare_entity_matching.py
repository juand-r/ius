#!/usr/bin/env python3
"""
Compare entity matching accuracy between hybrid and trusted baseline results.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def load_intersection(json_file):
    """Load the intersection data from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data.get("entity_analysis", {}).get("intersection", [])

def normalize_intersection(intersection_list):
    """Normalize intersection tuples for comparison."""
    # Convert to set of tuples, normalized to lowercase for comparison
    return set((summary.lower().strip(), source.lower().strip()) for summary, source in intersection_list)

def compare_intersections():
    """Compare intersections between hybrid and trusted baseline."""
    
    trusted_dir = Path("outputs/eval/intrinsic/entity-coverage/bmds-orig-compare/items")
    hybrid_dir = Path("outputs/eval/intrinsic/entity-coverage/bmds_fixed_size2_8000_all_concat_5e8bbe_entity_coverage_01045a/items")
    
    if not trusted_dir.exists():
        print(f"❌ Trusted directory not found: {trusted_dir}")
        return
    
    if not hybrid_dir.exists():
        print(f"❌ Hybrid directory not found: {hybrid_dir}")
        return
    
    # Get all JSON files from trusted directory
    trusted_files = list(trusted_dir.glob("*.json"))
    hybrid_files = list(hybrid_dir.glob("*.json"))
    
    trusted_items = {f.stem for f in trusted_files}
    hybrid_items = {f.stem for f in hybrid_files}
    
    print(f"📁 Trusted baseline: {len(trusted_items)} items")
    print(f"📁 Hybrid results: {len(hybrid_items)} items")
    print(f"📁 Common items: {len(trusted_items & hybrid_items)}")
    
    if not (trusted_items & hybrid_items):
        print("❌ No common items found!")
        return
    
    # Compare each common item
    results = {
        "perfect_matches": 0,
        "differences": 0,
        "missing_in_hybrid": 0,
        "extra_in_hybrid": 0,
        "details": []
    }
    
    for item_id in sorted(trusted_items & hybrid_items):
        trusted_file = trusted_dir / f"{item_id}.json"
        hybrid_file = hybrid_dir / f"{item_id}.json"
        
        try:
            trusted_intersection = load_intersection(trusted_file)
            hybrid_intersection = load_intersection(hybrid_file)
            
            # Normalize for comparison
            trusted_set = normalize_intersection(trusted_intersection)
            hybrid_set = normalize_intersection(hybrid_intersection)
            
            # Compare
            missing_in_hybrid = trusted_set - hybrid_set
            extra_in_hybrid = hybrid_set - trusted_set
            
            if not missing_in_hybrid and not extra_in_hybrid:
                results["perfect_matches"] += 1
                print(f"✅ {item_id}: Perfect match ({len(trusted_intersection)} intersections)")
            else:
                results["differences"] += 1
                
                detail = {
                    "item_id": item_id,
                    "trusted_count": len(trusted_intersection),
                    "hybrid_count": len(hybrid_intersection),
                    "missing_in_hybrid": list(missing_in_hybrid),
                    "extra_in_hybrid": list(extra_in_hybrid)
                }
                results["details"].append(detail)
                
                print(f"❌ {item_id}: Differences found")
                print(f"   Trusted: {len(trusted_intersection)} intersections")
                print(f"   Hybrid:  {len(hybrid_intersection)} intersections")
                
                if missing_in_hybrid:
                    print(f"   Missing in hybrid ({len(missing_in_hybrid)}):")
                    for summary, source in sorted(missing_in_hybrid):
                        print(f"      '{summary}' -> '{source}'")
                
                if extra_in_hybrid:
                    print(f"   Extra in hybrid ({len(extra_in_hybrid)}):")
                    for summary, source in sorted(extra_in_hybrid):
                        print(f"      '{summary}' -> '{source}'")
                
                print()
                
        except Exception as e:
            print(f"❌ Error processing {item_id}: {e}")
    
    # Summary
    total_items = len(trusted_items & hybrid_items)
    accuracy = results["perfect_matches"] / total_items * 100 if total_items > 0 else 0
    
    print("="*60)
    print("🎯 ACCURACY COMPARISON SUMMARY")
    print("="*60)
    print(f"Total items compared: {total_items}")
    print(f"Perfect matches: {results['perfect_matches']}")
    print(f"Items with differences: {results['differences']}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if results["differences"] > 0:
        print(f"\n📊 Detailed breakdown:")
        total_missing = sum(len(d["missing_in_hybrid"]) for d in results["details"])
        total_extra = sum(len(d["extra_in_hybrid"]) for d in results["details"])
        print(f"Total missing intersections: {total_missing}")
        print(f"Total extra intersections: {total_extra}")
    
    return results

if __name__ == "__main__":
    results = compare_intersections()