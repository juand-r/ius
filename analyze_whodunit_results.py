#!/usr/bin/env python3
"""
Script to analyze and compare whodunit evaluation results.
Compares results with and without reveal segments.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

def load_whodunit_results(directory_path):
    """Load all whodunit results from a directory."""
    results = {}
    items_dir = Path(directory_path) / "items"
    
    if not items_dir.exists():
        print(f"Warning: {items_dir} does not exist")
        return results
    
    for item_file in items_dir.glob("*.json"):
        item_id = item_file.stem
        try:
            with open(item_file, 'r') as f:
                data = json.load(f)
                results[item_id] = data
        except Exception as e:
            print(f"Error loading {item_file}: {e}")
    
    return results

def extract_parsed_response_fields(results):
    """Extract key fields from parsed responses."""
    extracted = {}
    
    for item_id, data in results.items():
        parsed = data.get("parsed_response", {})
        extracted[item_id] = {
            "thought_process": parsed.get("thought_process", "").strip(),
            "suspects": parsed.get("suspects", "").strip(),
            "main_culprits": parsed.get("main_culprits", "").strip(),
            "accomplices": parsed.get("accomplices", "").strip(),
            "event_reconstruction": parsed.get("event_reconstruction", "").strip(),
            "why_others_innocent": parsed.get("why_others_innocent", "").strip(),
            "raw_response_length": len(data.get("raw_response", "")),
            "total_cost": data.get("usage", {}).get("total_cost", 0),
            "total_tokens": data.get("usage", {}).get("total_tokens", 0),
        }
    
    return extracted

def analyze_culprit_identification(results):
    """Analyze culprit identification patterns."""
    analysis = {
        "total_items": len(results),
        "has_main_culprits": 0,
        "has_accomplices": 0,
        "has_neither": 0,
        "culprit_patterns": Counter(),
        "accomplice_patterns": Counter(),
    }
    
    for item_id, data in results.items():
        main_culprits = data["main_culprits"].lower()
        accomplices = data["accomplices"].lower()
        
        has_culprits = main_culprits and main_culprits not in ["none", "n/a", ""]
        has_accomplices = accomplices and accomplices not in ["none", "n/a", ""]
        
        if has_culprits:
            analysis["has_main_culprits"] += 1
            analysis["culprit_patterns"][main_culprits] += 1
        
        if has_accomplices:
            analysis["has_accomplices"] += 1
            analysis["accomplice_patterns"][accomplices] += 1
        
        if not has_culprits and not has_accomplices:
            analysis["has_neither"] += 1
    
    return analysis

def compare_results(without_reveal, with_reveal):
    """Compare results between with and without reveal."""
    comparison = {
        "common_items": set(without_reveal.keys()) & set(with_reveal.keys()),
        "only_without_reveal": set(without_reveal.keys()) - set(with_reveal.keys()),
        "only_with_reveal": set(with_reveal.keys()) - set(without_reveal.keys()),
        "culprit_agreement": 0,
        "accomplice_agreement": 0,
        "different_culprits": [],
        "different_accomplices": [],
        "cost_comparison": {},
        "token_comparison": {},
    }
    
    common_items = comparison["common_items"]
    
    for item_id in common_items:
        without = without_reveal[item_id]
        with_r = with_reveal[item_id]
        
        # Compare culprits
        without_culprits = without["main_culprits"].lower().strip()
        with_culprits = with_r["main_culprits"].lower().strip()
        
        if without_culprits == with_culprits:
            comparison["culprit_agreement"] += 1
        else:
            comparison["different_culprits"].append({
                "item_id": item_id,
                "without_reveal": without_culprits,
                "with_reveal": with_culprits
            })
        
        # Compare accomplices
        without_acc = without["accomplices"].lower().strip()
        with_acc = with_r["accomplices"].lower().strip()
        
        if without_acc == with_acc:
            comparison["accomplice_agreement"] += 1
        else:
            comparison["different_accomplices"].append({
                "item_id": item_id,
                "without_reveal": without_acc,
                "with_reveal": with_acc
            })
        
        # Cost and token comparison
        comparison["cost_comparison"][item_id] = {
            "without_reveal": without["total_cost"],
            "with_reveal": with_r["total_cost"]
        }
        
        comparison["token_comparison"][item_id] = {
            "without_reveal": without["total_tokens"],
            "with_reveal": with_r["total_tokens"]
        }
    
    return comparison

def print_analysis_summary(name, analysis):
    """Print analysis summary."""
    print(f"\n{'='*50}")
    print(f"ANALYSIS: {name}")
    print(f"{'='*50}")
    print(f"Total items: {analysis['total_items']}")
    print(f"Items with main culprits: {analysis['has_main_culprits']} ({analysis['has_main_culprits']/analysis['total_items']*100:.1f}%)")
    print(f"Items with accomplices: {analysis['has_accomplices']} ({analysis['has_accomplices']/analysis['total_items']*100:.1f}%)")
    print(f"Items with neither: {analysis['has_neither']} ({analysis['has_neither']/analysis['total_items']*100:.1f}%)")
    
    print(f"\nTop 5 most common culprit identifications:")
    for culprit, count in analysis['culprit_patterns'].most_common(5):
        print(f"  '{culprit}': {count}")
    
    if analysis['accomplice_patterns']:
        print(f"\nTop 5 most common accomplice identifications:")
        for acc, count in analysis['accomplice_patterns'].most_common(5):
            print(f"  '{acc}': {count}")

def print_comparison_summary(comparison):
    """Print comparison summary."""
    print(f"\n{'='*50}")
    print(f"COMPARISON: With vs Without Reveal")
    print(f"{'='*50}")
    
    common_count = len(comparison["common_items"])
    print(f"Common items: {common_count}")
    print(f"Only without reveal: {len(comparison['only_without_reveal'])}")
    print(f"Only with reveal: {len(comparison['only_with_reveal'])}")
    
    if common_count > 0:
        culprit_agreement_pct = comparison["culprit_agreement"] / common_count * 100
        accomplice_agreement_pct = comparison["accomplice_agreement"] / common_count * 100
        
        print(f"\nCulprit agreement: {comparison['culprit_agreement']}/{common_count} ({culprit_agreement_pct:.1f}%)")
        print(f"Accomplice agreement: {comparison['accomplice_agreement']}/{common_count} ({accomplice_agreement_pct:.1f}%)")
        
        if comparison["different_culprits"]:
            print(f"\nItems with different culprit identifications ({len(comparison['different_culprits'])}):")
            for diff in comparison["different_culprits"][:10]:  # Show first 10
                print(f"  {diff['item_id']}: '{diff['without_reveal']}' vs '{diff['with_reveal']}'")
        
        if comparison["different_accomplices"]:
            print(f"\nItems with different accomplice identifications ({len(comparison['different_accomplices'])}):")
            for diff in comparison["different_accomplices"][:10]:  # Show first 10
                print(f"  {diff['item_id']}: '{diff['without_reveal']}' vs '{diff['with_reveal']}'")
        
        # Cost and token analysis
        total_cost_without = sum(comparison["cost_comparison"][item]["without_reveal"] for item in comparison["common_items"])
        total_cost_with = sum(comparison["cost_comparison"][item]["with_reveal"] for item in comparison["common_items"])
        
        total_tokens_without = sum(comparison["token_comparison"][item]["without_reveal"] for item in comparison["common_items"])
        total_tokens_with = sum(comparison["token_comparison"][item]["with_reveal"] for item in comparison["common_items"])
        
        print(f"\nCost Analysis:")
        print(f"  Without reveal: ${total_cost_without:.4f}")
        print(f"  With reveal: ${total_cost_with:.4f}")
        print(f"  Difference: ${total_cost_with - total_cost_without:.4f}")
        
        print(f"\nToken Analysis:")
        print(f"  Without reveal: {total_tokens_without:,} tokens")
        print(f"  With reveal: {total_tokens_with:,} tokens")
        print(f"  Difference: {total_tokens_with - total_tokens_without:,} tokens")

def main():
    """Main analysis function."""
    # Define directories
    without_reveal_dir = "outputs/eval/extrinsic/bmds_fixed_size2_8000_whodunit_41c9c6"
    with_reveal_dir = "outputs/eval/extrinsic/bmds_fixed_size2_8000_whodunit_b4ca75"
    
    print("Loading whodunit evaluation results...")
    
    # Load results
    without_reveal_results = load_whodunit_results(without_reveal_dir)
    with_reveal_results = load_whodunit_results(with_reveal_dir)
    
    print(f"Loaded {len(without_reveal_results)} items without reveal")
    print(f"Loaded {len(with_reveal_results)} items with reveal")
    
    # Extract parsed response fields
    without_reveal_extracted = extract_parsed_response_fields(without_reveal_results)
    with_reveal_extracted = extract_parsed_response_fields(with_reveal_results)
    
    # Analyze each set
    without_reveal_analysis = analyze_culprit_identification(without_reveal_extracted)
    with_reveal_analysis = analyze_culprit_identification(with_reveal_extracted)
    
    # Print individual analyses
    print_analysis_summary("WITHOUT REVEAL (all-but-last)", without_reveal_analysis)
    print_analysis_summary("WITH REVEAL (all)", with_reveal_analysis)
    
    # Compare results
    comparison = compare_results(without_reveal_extracted, with_reveal_extracted)
    print_comparison_summary(comparison)
    
    # Save detailed results to CSV for further analysis
    if comparison["common_items"]:
        print(f"\nSaving detailed comparison to 'whodunit_comparison.csv'...")
        
        comparison_data = []
        for item_id in comparison["common_items"]:
            without = without_reveal_extracted[item_id]
            with_r = with_reveal_extracted[item_id]
            
            comparison_data.append({
                "item_id": item_id,
                "without_reveal_culprits": without["main_culprits"],
                "with_reveal_culprits": with_r["main_culprits"],
                "culprits_match": without["main_culprits"].lower().strip() == with_r["main_culprits"].lower().strip(),
                "without_reveal_accomplices": without["accomplices"],
                "with_reveal_accomplices": with_r["accomplices"],
                "accomplices_match": without["accomplices"].lower().strip() == with_r["accomplices"].lower().strip(),
                "without_reveal_cost": without["total_cost"],
                "with_reveal_cost": with_r["total_cost"],
                "without_reveal_tokens": without["total_tokens"],
                "with_reveal_tokens": with_r["total_tokens"],
            })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv("whodunit_comparison.csv", index=False)
        print("Detailed comparison saved!")

if __name__ == "__main__":
    main()