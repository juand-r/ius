#!/usr/bin/env python3
"""
Analyze the types of differences in entity matching to assess if they're reasonable.
"""

import json
from pathlib import Path

def categorize_difference(trusted_pair, hybrid_pair, difference_type):
    """Categorize the type of difference between matches."""
    if difference_type == "missing":
        trusted_summary, trusted_source = trusted_pair
        # Look for similar matches in hybrid that could be equivalent
        return {
            "type": "missing",
            "trusted": trusted_pair,
            "category": "unknown",
            "description": f"Trusted had '{trusted_summary}' -> '{trusted_source}' but hybrid didn't"
        }
    elif difference_type == "extra":
        hybrid_summary, hybrid_source = hybrid_pair
        return {
            "type": "extra", 
            "hybrid": hybrid_pair,
            "category": "unknown",
            "description": f"Hybrid had '{hybrid_summary}' -> '{hybrid_source}' but trusted didn't"
        }

def analyze_matching_quality():
    """Analyze if the differences represent errors or reasonable variations."""
    
    trusted_dir = Path("outputs/eval/intrinsic/entity-coverage/bmds-orig-compare/items")
    hybrid_dir = Path("outputs/eval/intrinsic/entity-coverage/bmds_fixed_size2_8000_all_concat_5e8bbe_entity_coverage_01045a/items")
    
    reasonable_differences = 0
    concerning_differences = 0
    
    print("🔍 DETAILED DIFFERENCE ANALYSIS")
    print("="*60)
    
    for item_file in trusted_dir.glob("*.json"):
        item_id = item_file.stem
        hybrid_file = hybrid_dir / f"{item_id}.json"
        
        if not hybrid_file.exists():
            continue
            
        # Load both files
        with open(item_file) as f:
            trusted_data = json.load(f)
        with open(hybrid_file) as f:
            hybrid_data = json.load(f)
            
        trusted_intersection = trusted_data.get("entity_analysis", {}).get("intersection", [])
        hybrid_intersection = hybrid_data.get("entity_analysis", {}).get("intersection", [])
        
        # Normalize for comparison
        trusted_set = set((s.lower().strip(), t.lower().strip()) for s, t in trusted_intersection)
        hybrid_set = set((s.lower().strip(), t.lower().strip()) for s, t in hybrid_intersection)
        
        missing = trusted_set - hybrid_set
        extra = hybrid_set - trusted_set
        
        if not missing and not extra:
            continue  # Perfect match
            
        print(f"\n📄 {item_id}:")
        print(f"   Trusted: {len(trusted_intersection)} intersections")
        print(f"   Hybrid:  {len(hybrid_intersection)} intersections")
        
        # Analyze missing matches
        if missing:
            print(f"   🟡 Missing in hybrid ({len(missing)}):")
            for summary, source in sorted(missing):
                # Check if there's a similar match in hybrid
                similar_in_hybrid = [
                    (h_sum, h_src) for h_sum, h_src in hybrid_set
                    if source in h_src or h_src in source or summary in h_sum or h_sum in summary
                ]
                
                if similar_in_hybrid:
                    print(f"      ✅ REASONABLE: '{summary}' -> '{source}' (hybrid has similar: {similar_in_hybrid[0]})")
                    reasonable_differences += 1
                else:
                    print(f"      ❌ CONCERNING: '{summary}' -> '{source}' (no similar match found)")
                    concerning_differences += 1
        
        # Analyze extra matches  
        if extra:
            print(f"   🟢 Extra in hybrid ({len(extra)}):")
            for summary, source in sorted(extra):
                # Check if this could be a more specific version of a trusted match
                similar_in_trusted = [
                    (t_sum, t_src) for t_sum, t_src in trusted_set
                    if source in t_src or t_src in source or summary in t_sum or t_sum in summary
                ]
                
                if similar_in_trusted:
                    print(f"      ✅ REASONABLE: '{summary}' -> '{source}' (more specific than trusted: {similar_in_trusted[0]})")
                    reasonable_differences += 1
                else:
                    # Check if this is a valid standalone match
                    if summary.lower() in source.lower() or source.lower() in summary.lower():
                        print(f"      ✅ REASONABLE: '{summary}' -> '{source}' (valid new match)")
                        reasonable_differences += 1
                    else:
                        print(f"      ❌ CONCERNING: '{summary}' -> '{source}' (unclear if valid)")
                        concerning_differences += 1
    
    total_differences = reasonable_differences + concerning_differences
    if total_differences > 0:
        reasonable_pct = reasonable_differences / total_differences * 100
        concerning_pct = concerning_differences / total_differences * 100
    else:
        reasonable_pct = concerning_pct = 0
        
    print("\n" + "="*60)
    print("🎯 DIFFERENCE QUALITY ANALYSIS")
    print("="*60)
    print(f"Total differences analyzed: {total_differences}")
    print(f"Reasonable differences: {reasonable_differences} ({reasonable_pct:.1f}%)")
    print(f"Concerning differences: {concerning_differences} ({concerning_pct:.1f}%)")
    print(f"\n💡 Assessment: The hybrid system appears to be making {'reasonable' if reasonable_pct > 70 else 'questionable'} matching decisions overall.")
    
    return {
        "total": total_differences,
        "reasonable": reasonable_differences,
        "concerning": concerning_differences,
        "reasonable_pct": reasonable_pct
    }

if __name__ == "__main__":
    analyze_matching_quality()