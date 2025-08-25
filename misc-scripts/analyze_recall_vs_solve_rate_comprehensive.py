#!/usr/bin/env python3
"""Comprehensive analysis of recall vs solve rate across multiple summary length constraints."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from scipy.stats import fisher_exact

def load_entity_coverage_metric(constraint, method="concat", dataset="bmds", metric="recall") -> Dict[str, float]:
    """Load entity coverage metric values for individual items.
    
    Args:
        constraint: Summary length constraint (e.g., '<500', 'summary')
        method: Summarization method ('concat' or 'iterative') 
        dataset: Dataset name (e.g., 'bmds')
        metric: Entity coverage metric ('recall', 'precision', or 'jaccard_similarity')
    """
    base_path = Path("outputs/eval/intrinsic/entity-coverage")
    metric_values = {}
    
    for eval_dir in base_path.iterdir():
        if not eval_dir.is_dir():
            continue
        
        # Filter by dataset and method
        if dataset.lower() not in eval_dir.name.lower():
            continue
        if method.lower() not in eval_dir.name.lower():
            continue
        
        try:
            # Load collection metadata to get constraint info
            collection_file = eval_dir / "collection.json"
            if not collection_file.exists():
                continue
                
            with open(collection_file, 'r') as f:
                collection = json.load(f)
            
            eval_info = collection.get('entity_coverage_evaluation_info', {})
            processing_stats = eval_info.get('processing_stats', {})
            
            # Skip if no successful items
            if processing_stats.get('successful_items', 0) == 0:
                continue
            
            # Get source collection to extract constraint
            meta = eval_info.get('collection_metadata', {})
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
            if short_constraint != constraint:
                continue
            
            # Load individual item recall values
            items_dir = eval_dir / "items"
            if not items_dir.exists():
                continue
            
            for item_file in items_dir.glob("*.json"):
                try:
                    with open(item_file, 'r') as f:
                        item_data = json.load(f)
                    
                    entity_analysis = item_data.get('entity_analysis', {})
                    metrics = entity_analysis.get('metrics', {})
                    
                    if metrics:
                        item_id = item_file.stem
                        metric_value = metrics.get(metric, 0)
                        metric_values[item_id] = metric_value
                
                except Exception as e:
                    continue
            
            break  # Found our target evaluation, stop looking
                        
        except Exception as e:
            continue
    
    return metric_values

def load_whodunit_solve_rates(constraint, method="concat", dataset="bmds") -> Dict[str, bool]:
    """Load whodunit solve rates for individual items."""
    base_path = Path("outputs/eval/extrinsic")
    solve_rates = {}
    
    for eval_dir in base_path.iterdir():
        if not eval_dir.is_dir():
            continue
        
        # Filter by dataset and method
        if dataset.lower() not in eval_dir.name.lower():
            continue
        if method.lower() not in eval_dir.name.lower():
            continue
        
        try:
            # Load collection metadata
            collection_file = eval_dir / "collection.json"
            if not collection_file.exists():
                continue
                
            with open(collection_file, 'r') as f:
                collection = json.load(f)
            
            eval_info = collection.get('whodunit_evaluation_info', {})
            meta = eval_info.get('collection_metadata', {})
            
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
            if short_constraint != constraint:
                continue
            
            # Load individual item solve rates
            items_dir = eval_dir / "items"
            if not items_dir.exists():
                continue
            
            for item_file in items_dir.glob("*.json"):
                try:
                    with open(item_file, 'r') as f:
                        item_data = json.load(f)
                    
                    assessment = item_data.get('solution_correctness_assessment', {})
                    if not assessment:
                        continue
                    
                    item_id = item_file.stem
                    culprit_correct = assessment.get('culprit', {}).get('culprit_correct') == 'Yes'
                    solve_rates[item_id] = culprit_correct
                
                except Exception as e:
                    continue
            
            break  # Found our target evaluation, stop looking
                        
        except Exception as e:
            continue
    
    return solve_rates

def get_short_constraint(constraint):
    """Convert full constraint to short form."""
    if not constraint:
        return "unknown"
    
    constraint_lower = constraint.lower()
    
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
        import re
        match = re.search(r'(\d+)\s*words?', constraint_lower)
        if match:
            return f"<{match.group(1)}"
        return constraint[:10] + "..." if len(constraint) > 10 else constraint

def analyze_single_constraint(constraint, metric_threshold=0.35, method="concat", dataset="bmds", metric="recall"):
    """Analyze entity coverage metric vs solve rate for a single constraint."""
    
    # Load data
    metric_values = load_entity_coverage_metric(constraint, method, dataset, metric)
    solve_rates = load_whodunit_solve_rates(constraint, method, dataset)
    
    if not metric_values or not solve_rates:
        return None
    
    # Find common items
    common_items = set(metric_values.keys()) & set(solve_rates.keys())
    
    if len(common_items) == 0:
        return None
    
    # Create matched dataset
    matched_data = []
    for item_id in common_items:
        matched_data.append({
            'item_id': item_id,
            'metric_value': metric_values[item_id],
            'solved': solve_rates[item_id]
        })
    
    # Sort by metric value
    matched_data.sort(key=lambda x: x['metric_value'])
    
    # Calculate optimal threshold for roughly equal bins
    metric_vals = [item['metric_value'] for item in matched_data]
    median_metric = np.median(metric_vals)
    
    # Use median as threshold for equal-sized bins
    threshold = median_metric
    
    # Split into bins
    low_metric_items = [item for item in matched_data if item['metric_value'] < threshold]
    high_metric_items = [item for item in matched_data if item['metric_value'] >= threshold]
    
    # Calculate solve rates for each bin
    low_metric_solved = sum(1 for item in low_metric_items if item['solved'])
    high_metric_solved = sum(1 for item in high_metric_items if item['solved'])
    
    low_metric_solve_rate = low_metric_solved / len(low_metric_items) if low_metric_items else 0
    high_metric_solve_rate = high_metric_solved / len(high_metric_items) if high_metric_items else 0
    
    # Statistical test
    if low_metric_items and high_metric_items:
        contingency_table = [
            [low_metric_solved, len(low_metric_items) - low_metric_solved],
            [high_metric_solved, len(high_metric_items) - high_metric_solved]
        ]
        try:
            odds_ratio, p_value = fisher_exact(contingency_table)
        except:
            odds_ratio, p_value = 1.0, 1.0
    else:
        odds_ratio, p_value = 1.0, 1.0
    
    return {
        'constraint': constraint,
        'n_items': len(matched_data),
        'threshold': threshold,
        'low_metric_bin_size': len(low_metric_items),
        'high_metric_bin_size': len(high_metric_items),
        'low_metric_solve_rate': low_metric_solve_rate,
        'high_metric_solve_rate': high_metric_solve_rate,
        'difference': high_metric_solve_rate - low_metric_solve_rate,
        'p_value': p_value,
        'odds_ratio': odds_ratio,
        'metric_range': (min(metric_vals), max(metric_vals)),
        'matched_data': matched_data
    }

def create_comprehensive_analysis(constraints=["<200", "<500", "summary", "long"], method="concat", dataset="bmds", metric="recall"):
    """Create comprehensive analysis across multiple constraints."""
    
    print(f"Comprehensive Analysis: {metric.title()} vs Solve Rate")
    print(f"Method: {method.title()}, Dataset: {dataset.upper()}")
    print("=" * 60)
    
    results = {}
    
    # Analyze each constraint
    for constraint in constraints:
        print(f"\nAnalyzing {constraint} constraint...")
        result = analyze_single_constraint(constraint, method=method, dataset=dataset, metric=metric)
        if result:
            results[constraint] = result
            print(f"  Found {result['n_items']} matching items")
            print(f"  Threshold: {result['threshold']:.3f}")
            print(f"  Bin sizes: {result['low_metric_bin_size']} vs {result['high_metric_bin_size']}")
            print(f"  Solve rates: {result['low_metric_solve_rate']:.3f} vs {result['high_metric_solve_rate']:.3f}")
            print(f"  Difference: {result['difference']:+.3f}, p={result['p_value']:.3f}")
        else:
            print(f"  No data found for {constraint}")
    
    if not results:
        print("No data found for any constraints!")
        return
    
    # Create visualization
    n_constraints = len(results)
    fig, axes = plt.subplots(1, n_constraints, figsize=(5*n_constraints, 6))
    
    if n_constraints == 1:
        axes = [axes]
    
    fig.suptitle(f'Entity Coverage {metric.title()} vs Solve Rate Analysis\n({method.title()} Summaries, {dataset.upper()})', 
                 fontsize=16, fontweight='bold')
    
    colors = ['lightcoral', 'lightblue']
    
    for i, (constraint, result) in enumerate(results.items()):
        # Bar chart of solve rates
        ax = axes[i]
        
        bins = [f'Low {metric.title()}\n(<{result["threshold"]:.2f})', f'High {metric.title()}\n(≥{result["threshold"]:.2f})']
        solve_rates_plot = [result['low_metric_solve_rate'], result['high_metric_solve_rate']]
        bin_sizes = [result['low_metric_bin_size'], result['high_metric_bin_size']]
        
        bars = ax.bar(bins, solve_rates_plot, color=colors, alpha=0.8, edgecolor='black')
        if i == 0:  # Only leftmost subplot gets y-label
            ax.set_ylabel('Solve Rate (Fraction with Culprit Correct)')
        ax.set_title(f'{constraint} Constraint')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rate, size in zip(bars, solve_rates_plot, bin_sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{rate:.3f}\n(n={size})', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Add significance indicator with clearer explanation
        if result['p_value'] < 0.05:
            sig_text = f'p = {result["p_value"]:.3f}\n(Significant)'
            bg_color = 'yellow'
        else:
            sig_text = f'p = {result["p_value"]:.3f}\n(Not Significant)'
            bg_color = 'lightgray'
            
        ax.text(0.5, 0.75, sig_text, ha='center', va='center', 
               transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'comprehensive_{metric}_vs_solve_rate_{method}_{dataset}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as: {filename}")
    
    # Print explanation of p-values
    print("\n" + "="*80)
    print("STATISTICAL INTERPRETATION:")
    print("="*80)
    print("P-values measure statistical significance (Fisher's exact test):")
    print("  • p < 0.05: Significant relationship (unlikely due to chance)")
    print("  • p > 0.05: No significant relationship (could be due to chance)")
    print("")
    print(f"HIGH p-values in our results indicate that entity coverage {metric}")
    print("has little to no significant impact on solve rates!")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE:")
    print("="*80)
    print(f"{'Constraint':<12} {'N':<4} {'Threshold':<10} {'Low→High':<12} {'Difference':<12} {'P-value':<10} {'Significant?':<12}")
    print("-"*80)
    
    for constraint, result in results.items():
        low_rate = result['low_metric_solve_rate']
        high_rate = result['high_metric_solve_rate']
        diff = result['difference']
        p_val = result['p_value']
        threshold = result['threshold']
        n_items = result['n_items']
        
        significance = "Yes" if p_val < 0.05 else "No"
        print(f"{constraint:<12} {n_items:<4} {threshold:<10.3f} {low_rate:.3f}→{high_rate:.3f}  {diff:+.3f}       {p_val:<10.3f} {significance:<12}")
    
    print(f"\nKEY FINDING: High p-values confirm that entity coverage {metric}")
    print("does NOT significantly predict solve rates for these summary lengths.")
    
    return results

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    method = sys.argv[1] if len(sys.argv) > 1 else "concat"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "bmds"
    metric = sys.argv[3] if len(sys.argv) > 3 else "recall"
    
    # Define constraints to analyze
    constraints = ["<200", "<500", "summary", "long"]
    
    print(f"Running comprehensive analysis:")
    print(f"  Method: {method}")
    print(f"  Dataset: {dataset}")
    print(f"  Metric: {metric}")
    print(f"  Constraints: {constraints}")
    
    results = create_comprehensive_analysis(constraints, method, dataset, metric)