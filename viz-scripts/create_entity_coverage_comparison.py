#!/usr/bin/env python3
"""Create side-by-side comparison charts for entity coverage analysis."""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Import the analysis functions from the main script
from create_entity_coverage_charts import load_entity_coverage_results

def create_entity_coverage_comparison_chart(dataset_filter="bmds", use_last_only=True, add_error_bars=False):
    """Create side-by-side comparison charts for entity coverage analysis."""
    
    # Load evaluation results
    results = load_entity_coverage_results(dataset_filter, use_last_only)
    
    if not results:
        print(f"No {dataset_filter} entity coverage results found!")
        return []
    
    print(f"Found {len(results)} {dataset_filter} entity coverage evaluation results")
    
    # Group results by method
    concat_results = [r for r in results if r['method'] == 'concat']
    iterative_results = [r for r in results if r['method'] == 'iterative']
    
    # Sort by short constraint for consistent ordering
    concat_results.sort(key=lambda x: x['short_constraint'])
    iterative_results.sort(key=lambda x: x['short_constraint'])
    
    # Define constraint order based on dataset
    if dataset_filter.lower() in ['bmds', 'detectiveqa']:
        # For bmds and detectiveqa: summary goes after <500
        expected_constraints = ['<50', '<100', '<200', '<500', 'summary', '<1000', 'long']
    if dataset_filter.lower() in ['detectiveqa']:
        # For bmds and detectiveqa: summary goes after <500
        expected_constraints = ['<200', '<500', 'summary', '<1000', 'long']
    elif dataset_filter.lower() == 'true-detective':
        # For true-detective: summary goes after <200
        expected_constraints = ['<50', '<100', '<200', 'summary', '<500', '<1000', 'long']
    else:
        # Default order
        expected_constraints = ['<50', '<100', '<200', '<500', 'summary', '<1000', 'long']
    
    metrics = ['precision', 'recall', 'jaccard']
    metric_titles = ['Precision', 'Recall', 'Jaccard Similarity']
    
    # Colors for different methods
    concat_color = '#FF7F50'  # Orange
    iterative_color = '#9370DB'  # Purple
    
    chart_files = []
    
    # Create separate chart for each metric
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        # Create individual figure for this metric
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(f'Entity Coverage {title}: Concat vs Iterative ({dataset_filter.upper()})', fontsize=16, fontweight='bold')
        
        # Prepare data for grouped bar chart
        x = np.arange(len(expected_constraints))
        width = 0.35
        
        concat_means = []
        concat_sems = []
        iterative_means = []
        iterative_sems = []
        x_labels = []
        
        for constraint in expected_constraints:
            # Find concat result for this constraint
            concat_result = next((r for r in concat_results if r['short_constraint'] == constraint), None)
            if concat_result:
                concat_means.append(concat_result[metric]['mean'] * 100)
                concat_sems.append(concat_result[metric]['sem'] * 100)
                concat_word_count = concat_result['avg_word_count']
            else:
                concat_means.append(0)
                concat_sems.append(0)
                concat_word_count = None
            
            # Find iterative result for this constraint
            iterative_result = next((r for r in iterative_results if r['short_constraint'] == constraint), None)
            if iterative_result:
                iterative_means.append(iterative_result[metric]['mean'] * 100)
                iterative_sems.append(iterative_result[metric]['sem'] * 100)
                iterative_word_count = iterative_result['avg_word_count']
            else:
                iterative_means.append(0)
                iterative_sems.append(0)
                iterative_word_count = None
            
            # Create x-axis label without word count
            # Replace "summary" with "unconstrained" for display
            display_constraint = "unconstrained" if constraint == "summary" else constraint
            x_labels.append(display_constraint)
        
        # Create grouped bar chart (only show bars with data > 0)
        bars1_list = []
        bars2_list = []
        concat_labeled = False
        iterative_labeled = False
        
        for j, (concat_mean, concat_sem, iter_mean, iter_sem) in enumerate(zip(concat_means, concat_sems, iterative_means, iterative_sems)):
            if concat_mean > 0:
                bar1 = ax.bar(j - width/2, concat_mean, width, 
                             yerr=concat_sem if add_error_bars else None,
                             label='Concat' if not concat_labeled else "", color=concat_color, alpha=0.8, 
                             capsize=5 if add_error_bars else 0)
                bars1_list.extend(bar1)
                concat_labeled = True
            
            if iter_mean > 0:
                bar2 = ax.bar(j + width/2, iter_mean, width, 
                             yerr=iter_sem if add_error_bars else None,
                             label='Iterative' if not iterative_labeled else "", color=iterative_color, alpha=0.8, 
                             capsize=5 if add_error_bars else 0)
                bars2_list.extend(bar2)
                iterative_labeled = True
        
        ax.set_xlabel('Summary Length Constraint (word count)', fontsize=12, labelpad=15)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        if concat_labeled or iterative_labeled:  # Only show legend if we have at least one method
            ax.legend(loc='upper left')
        ax.set_ylim(0, 100.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for j, (concat_mean, concat_sem, iter_mean, iter_sem) in enumerate(zip(concat_means, concat_sems, iterative_means, iterative_sems)):
            if concat_mean > 0:
                y_offset = (concat_sem + 2.0) if add_error_bars else 2.0
                ax.text(j - width/2, concat_mean + y_offset,
                       f'{concat_mean:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            if iter_mean > 0:
                y_offset = (iter_sem + 2.0) if add_error_bars else 2.0
                ax.text(j + width/2, iter_mean + y_offset,
                       f'{iter_mean:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Save and show individual chart
        plt.tight_layout()
        chart_filename = f'../plots/entity_coverage_{metric}_{dataset_filter}.png'
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        chart_files.append(chart_filename)
        plt.show()
        plt.close()  # Close the figure to free memory
    
    # Print comparative analysis
    print("\nComparative Analysis:")
    print("=" * 50)
    
    for constraint in expected_constraints:
        concat_result = next((r for r in concat_results if r['short_constraint'] == constraint), None)
        iterative_result = next((r for r in iterative_results if r['short_constraint'] == constraint), None)
        
        print(f"\n{constraint} constraint:")
        if concat_result:
            print(f"  Concat    ({concat_result['avg_word_count']}w): P:{concat_result['precision']['mean']*100:.1f}±{concat_result['precision']['sem']*100:.1f}%, "
                  f"R:{concat_result['recall']['mean']*100:.1f}±{concat_result['recall']['sem']*100:.1f}%, "
                  f"J:{concat_result['jaccard']['mean']*100:.1f}±{concat_result['jaccard']['sem']*100:.1f}%")
        else:
            print("  Concat:    No data")
        
        if iterative_result:
            print(f"  Iterative ({iterative_result['avg_word_count']}w): P:{iterative_result['precision']['mean']*100:.1f}±{iterative_result['precision']['sem']*100:.1f}%, "
                  f"R:{iterative_result['recall']['mean']*100:.1f}±{iterative_result['recall']['sem']*100:.1f}%, "
                  f"J:{iterative_result['jaccard']['mean']*100:.1f}±{iterative_result['jaccard']['sem']*100:.1f}%")
        else:
            print("  Iterative: No data")
        
        # Show differences if both methods have data
        if concat_result and iterative_result:
            prec_diff = (iterative_result['precision']['mean'] - concat_result['precision']['mean']) * 100
            rec_diff = (iterative_result['recall']['mean'] - concat_result['recall']['mean']) * 100
            jacc_diff = (iterative_result['jaccard']['mean'] - concat_result['jaccard']['mean']) * 100
            
            print(f"  Difference (Iterative - Concat): P:{prec_diff:+.1f}%, R:{rec_diff:+.1f}%, J:{jacc_diff:+.1f}%")
    
    return chart_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create entity coverage comparison charts")
    parser.add_argument("dataset", nargs="?", default="bmds", 
                       help="Dataset to analyze (bmds or true-detective)")
    parser.add_argument("--add-error-bars", action="store_true",
                       help="Add error bars to the charts")
    parser.add_argument("--use-all-summaries", action="store_true",
                       help="Use all summaries instead of just the last one for each item")
    
    args = parser.parse_args()
    
    dataset = args.dataset.lower()
    if dataset not in ["bmds", "true-detective", "detectiveqa"]:
        print(f"Error: Unknown dataset '{dataset}'. Use 'bmds', 'true-detective', or 'detectiveqa'")
        exit(1)
    
    use_last_only = not args.use_all_summaries
    
    print(f"Creating entity coverage comparison charts for dataset: {dataset}")
    print(f"Using {'last summary only' if use_last_only else 'all summaries'} for each item")
    print(f"Error bars: {'enabled' if args.add_error_bars else 'disabled'}")
    
    chart_files = create_entity_coverage_comparison_chart(dataset, use_last_only, args.add_error_bars)
    if chart_files:
        print(f"Comparison charts saved as:")
        for chart_file in chart_files:
            print(f"  - {chart_file}")
    else:
        print("No charts were created.")