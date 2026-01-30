#!/usr/bin/env python3
"""Analyze distributions of entity coverage metrics to explore relationship with solve rates."""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_entity_coverage_data(constraint_filter="<500", dataset="bmds") -> Dict[str, List[float]]:
    """
    Load entity coverage data for a specific constraint and extract individual item metrics.
    
    Args:
        constraint_filter: Length constraint to filter by (e.g., "<500", "<200", "summary", "long")
        dataset: Dataset name (e.g., "bmds")
        
    Returns:
        Dictionary with method -> list of metric values for individual items
    """
    base_path = Path("outputs/eval/intrinsic/entity-coverage")
    results = {
        'concat_precision': [],
        'concat_recall': [],
        'concat_jaccard': [],
        'iterative_precision': [],
        'iterative_recall': [],
        'iterative_jaccard': []
    }
    
    print(f"Loading entity coverage data for {constraint_filter} constraint...")
    
    for eval_dir in base_path.iterdir():
        if not eval_dir.is_dir():
            continue
        
        # Filter by dataset
        if dataset.lower() not in eval_dir.name.lower():
            continue
        
        try:
            # Load collection metadata to get constraint info
            collection_file = eval_dir / "collection.json"
            if not collection_file.exists():
                continue
                
            with open(collection_file, 'r') as f:
                collection = json.load(f)
            
            eval_info = collection.get('entity_coverage_evaluation_info', {})
            meta = eval_info.get('collection_metadata', {})
            
            # Skip if no successful items
            processing_stats = eval_info.get('processing_stats', {})
            if processing_stats.get('successful_items', 0) == 0:
                continue
            
            # Get source collection to extract constraint
            source_collection = meta.get('source_collection', '')
            if not source_collection:
                continue
            
            # Load source collection metadata to get summary length constraint
            source_collection_file = Path(source_collection) / "collection.json"
            if not source_collection_file.exists():
                continue
                
            with open(source_collection_file, 'r') as f:
                source_data = json.load(f)
            source_meta = source_data.get('summarization_info', {}).get('collection_metadata', {})
            length_constraint = source_meta.get('optional_summary_length', 'unknown')
            
            # Convert constraint to short form
            short_constraint = get_short_constraint(length_constraint)
            
            # Skip if not the constraint we're looking for
            if short_constraint != constraint_filter:
                continue
            
            # Determine method
            if 'concat' in eval_dir.name:
                method = 'concat'
            elif 'iterative' in eval_dir.name:
                method = 'iterative'
            else:
                continue
            
            print(f"  Found {method} results: {eval_dir.name}")
            
            # Load individual item metrics
            items_dir = eval_dir / "items"
            if not items_dir.exists():
                continue
            
            item_count = 0
            for item_file in items_dir.glob("*.json"):
                try:
                    with open(item_file, 'r') as f:
                        item_data = json.load(f)
                    
                    entity_analysis = item_data.get('entity_analysis', {})
                    metrics = entity_analysis.get('metrics', {})
                    
                    if metrics:
                        precision = metrics.get('precision', 0)
                        recall = metrics.get('recall', 0)
                        jaccard = metrics.get('jaccard_similarity', 0)
                        
                        results[f'{method}_precision'].append(precision)
                        results[f'{method}_recall'].append(recall)
                        results[f'{method}_jaccard'].append(jaccard)
                        item_count += 1
                
                except Exception as e:
                    print(f"    Error processing {item_file}: {e}")
                    continue
            
            print(f"    Loaded {item_count} items")
                        
        except Exception as e:
            print(f"  Error processing {eval_dir.name}: {e}")
            continue
    
    print(f"\nSummary:")
    for key, values in results.items():
        if values:
            print(f"  {key}: {len(values)} items, mean={np.mean(values):.3f}, std={np.std(values):.3f}")
    
    return results

def get_short_constraint(constraint):
    """Convert full constraint to short form, based on create_entity_coverage_charts.py."""
    if not constraint:
        return "unknown"
    
    constraint_lower = constraint.lower()
    
    # Handle specific word count constraints (order matters - check smaller numbers first)
    if "less than 50 words" in constraint_lower:
        return "<50"
    elif "less than 100 words" in constraint_lower:
        return "<100"
    elif "less than 200 words" in constraint_lower:
        return "<200"
    elif "less than 500 words" in constraint_lower:
        return "<500"
    elif "less than 1000 words" in constraint_lower:
        return "<1000"
    elif "very long" in constraint_lower:
        return "long"
    elif "summary" in constraint_lower and "word" not in constraint_lower:
        return "summary"
    else:
        # Try to extract any word count
        import re
        match = re.search(r'(\d+)\s*words?', constraint_lower)
        if match:
            return f"<{match.group(1)}"
        return constraint[:10] + "..." if len(constraint) > 10 else constraint

def plot_entity_coverage_histograms(constraint="<500", dataset="bmds"):
    """Create histograms of entity coverage metrics for a specific constraint."""
    
    # Load data
    data = load_entity_coverage_data(constraint, dataset)
    
    if not any(data.values()):
        print(f"No data found for constraint '{constraint}' in dataset '{dataset}'")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Entity Coverage Distributions: {constraint} constraint ({dataset.upper()})', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['precision', 'recall', 'jaccard']
    colors = ['orange', 'purple']
    method_names = ['Concat', 'Iterative']
    
    for i, metric in enumerate(metrics):
        for j, (method, color, method_name) in enumerate(zip(['concat', 'iterative'], colors, method_names)):
            ax = axes[j, i]
            
            values = data[f'{method}_{metric}']
            
            if not values:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method_name}: {metric.title()}')
                continue
            
            # Create histogram
            ax.hist(values, bins=20, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
            
            # Add statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            median_val = np.median(values)
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            
            ax.set_title(f'{method_name}: {metric.title()}')
            ax.set_xlabel(f'{metric.title()} Score')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text box with stats
            stats_text = f'n={len(values)}\nμ={mean_val:.3f}\nσ={std_val:.3f}\nMin={min(values):.3f}\nMax={max(values):.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8), fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'entity_coverage_distributions_{constraint.replace("<", "lt")}_{dataset}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Histogram saved as: {filename}")
    
    # Print detailed statistics
    print(f"\nDetailed Statistics for {constraint} constraint:")
    print("=" * 60)
    
    for method in ['concat', 'iterative']:
        print(f"\n{method.upper()} METHOD:")
        for metric in metrics:
            values = data[f'{method}_{metric}']
            if values:
                values_array = np.array(values)
                print(f"  {metric.title()}: n={len(values)}, "
                      f"mean={np.mean(values):.3f} ± {np.std(values):.3f}, "
                      f"median={np.median(values):.3f}, "
                      f"range=[{np.min(values):.3f}, {np.max(values):.3f}], "
                      f"Q1={np.percentile(values, 25):.3f}, Q3={np.percentile(values, 75):.3f}")
            else:
                print(f"  {metric.title()}: No data")

def analyze_multiple_constraints(constraints=["<200", "<500", "summary", "long"], dataset="bmds"):
    """Create a comparison plot showing recall distributions across different constraints."""
    
    fig, axes = plt.subplots(2, len(constraints), figsize=(5*len(constraints), 10))
    fig.suptitle(f'Recall Distributions Across Length Constraints ({dataset.upper()})', 
                 fontsize=16, fontweight='bold')
    
    colors = ['orange', 'purple']
    method_names = ['Concat', 'Iterative']
    
    all_data = {}
    
    for constraint in constraints:
        print(f"\nLoading data for {constraint} constraint...")
        data = load_entity_coverage_data(constraint, dataset)
        all_data[constraint] = data
    
    for i, constraint in enumerate(constraints):
        data = all_data[constraint]
        
        for j, (method, color, method_name) in enumerate(zip(['concat', 'iterative'], colors, method_names)):
            if len(constraints) == 1:
                ax = axes[j]
            else:
                ax = axes[j, i]
            
            recall_values = data[f'{method}_recall']
            
            if not recall_values:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method_name}: {constraint}')
                continue
            
            # Create histogram
            ax.hist(recall_values, bins=15, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
            
            # Add statistics
            mean_val = np.mean(recall_values)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'μ={mean_val:.3f}')
            
            ax.set_title(f'{method_name}: {constraint}')
            if i == 0:  # Only add y-label to leftmost plots
                ax.set_ylabel('Frequency')
            if j == 1:  # Only add x-label to bottom plots
                ax.set_xlabel('Recall Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'recall_distributions_comparison_{dataset}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plot saved as: {filename}")

if __name__ == "__main__":
    import sys
    
    # Default to <500 constraint for BMDS
    constraint = sys.argv[1] if len(sys.argv) > 1 else "<500"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "bmds"
    
    if constraint == "all":
        # Analyze multiple constraints
        analyze_multiple_constraints(dataset=dataset)
    else:
        # Analyze single constraint
        plot_entity_coverage_histograms(constraint, dataset)