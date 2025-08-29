#!/usr/bin/env python3
"""Create side-by-side comparison charts for faithfulness analysis."""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import re
import scipy.stats as stats

def load_faithfulness_results(dataset_filter="bmds", use_last_only=True, less_strict=False, less_strict_also=False):
    """Load and analyze all faithfulness evaluation results."""
    base_path = Path("../outputs/eval/intrinsic/faithfulness")
    results = []
    
    for eval_dir in base_path.iterdir():
        if not eval_dir.is_dir():
            continue
        
        # Filter by dataset
        if dataset_filter.lower() not in eval_dir.name.lower():
            continue
        
        try:
            result = analyze_faithfulness_directory(eval_dir, use_last_only, less_strict, less_strict_also)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing {eval_dir.name}: {e}")
    
    return results

def analyze_faithfulness_directory(eval_dir, use_last_only=True, less_strict=False, less_strict_also=False):
    """Analyze a single faithfulness evaluation directory."""
    
    # Load collection metadata
    collection_file = eval_dir / "collection.json"
    if not collection_file.exists():
        return None
    
    with open(collection_file, 'r') as f:
        collection_data = json.load(f)
    
    # Extract input path (claims directory) and trace back to summaries
    input_path = collection_data.get('input_path', '')
    if not input_path:
        return None
    
    # The input_path points to claims directory, we need to get the summaries directory
    # Example: outputs/summaries-claims/bmds_fixed_size2_8000_all_concat_131eac_claims_default-claim-extraction
    # We need: outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac
    
    # Extract the summary collection name by removing the claims suffix
    claims_dir_name = Path(input_path).name
    # Remove _claims_default-claim-extraction suffix to get summary collection name
    summary_collection_name = claims_dir_name.replace('_claims_default-claim-extraction', '')
    
    # Build path to summary collection
    source_collection_file = Path("..") / "outputs" / "summaries" / summary_collection_name / "collection.json"
    if not source_collection_file.exists():
        return None
    
    with open(source_collection_file, 'r') as f:
        source_data = json.load(f)
    
    summarization_info = source_data.get('summarization_info', {})
    source_metadata = summarization_info.get('collection_metadata', {})
    constraint = source_metadata.get('optional_summary_length', 'unknown')
    
    # Determine method from directory name
    method = 'concat' if 'concat' in eval_dir.name else 'iterative'
    
    # Shorten constraint for display
    short_constraint = shorten_length_constraint(constraint)
    
    # Load faithfulness results from items
    items_dir = eval_dir / "items"
    if not items_dir.exists():
        return None
    
    faithfulness_scores = []
    faithfulness_scores_less_strict = []  # For --less-strict-also mode
    word_counts = []
    
    for item_dir in items_dir.iterdir():
        if not item_dir.is_dir():
            continue
        
        # Find the highest numbered JSON file (or only file if use_last_only)
        json_files = list(item_dir.glob("*.json"))
        if not json_files:
            continue
        
        if use_last_only:
            # Get the highest numbered file
            json_files.sort(key=lambda x: int(x.stem))
            target_file = json_files[-1]
        else:
            # Process all files (for future extension)
            target_file = json_files[0]  # For now, just use the first
        
        # Get item_id first
        item_id = item_dir.name
        
        # Load faithfulness results
        with open(target_file, 'r') as f:
            item_data = json.load(f)
        
        faithfulness_results = item_data.get('faithfulness_results', [])
        if not faithfulness_results:
            continue
        
        # Calculate faithfulness score for this item
        for result in faithfulness_results:
            claim_evaluations = result.get('claim_evaluations', [])
            if claim_evaluations:
                # Log "No" predictions
                for eval in claim_evaluations:
                    if eval.get('predicted_label') == 'No':
                        print(f"LOG: 'No' prediction found - Directory: {eval_dir.name}, Item: {item_id}")
                        print(f"  Full evaluation data: {eval}")
                
                total_claims = len(claim_evaluations)
                if total_claims > 0:
                    if less_strict_also:
                        # Calculate both strict and less-strict scores
                        faithful_count = sum(1 for eval in claim_evaluations if eval.get('faithful', False))
                        most_likely_faithful_claims = sum(1 for eval in claim_evaluations 
                                                        if eval.get('predicted_label') in ['Yes', 'PartialSupport'])
                        
                        strict_score = (faithful_count / total_claims) * 100
                        less_strict_score = (most_likely_faithful_claims / total_claims) * 100
                        
                        faithfulness_scores.append(strict_score)
                        faithfulness_scores_less_strict.append(less_strict_score)
                    elif less_strict:
                        # Less strict: count "Yes" and "PartialSupport" as faithful
                        most_likely_faithful_claims = sum(1 for eval in claim_evaluations 
                                                        if eval.get('predicted_label') in ['Yes', 'PartialSupport'])
                        faithfulness_score = (most_likely_faithful_claims / total_claims) * 100
                        faithfulness_scores.append(faithfulness_score)
                    else:
                        # Standard: use the 'faithful' field
                        faithful_count = sum(1 for eval in claim_evaluations if eval.get('faithful', False))
                        faithfulness_score = (faithful_count / total_claims) * 100
                        faithfulness_scores.append(faithfulness_score)
        
        # Get word count from source summary
        source_items_dir = Path("..") / "outputs" / "summaries" / summary_collection_name / "items"
        source_item_file = source_items_dir / f"{item_id}.json"
        
        if source_item_file.exists():
            with open(source_item_file, 'r') as f:
                source_item = json.load(f)
            
            documents = source_item.get('documents', [])
            if documents:
                summaries = documents[0].get('summaries', [])
                if summaries:
                    if use_last_only and len(summaries) > 0:
                        summary_text = summaries[-1]  # Last summary
                    else:
                        summary_text = summaries[0]  # First summary
                    
                    if summary_text:
                        word_count = len(summary_text.split())
                        word_counts.append(word_count)
    
    if not faithfulness_scores:
        return None
    
    # Calculate statistics
    mean_faithfulness = np.mean(faithfulness_scores)
    sem_faithfulness = stats.sem(faithfulness_scores) if len(faithfulness_scores) > 1 else 0
    avg_word_count = np.mean(word_counts) if word_counts else 0
    
    result = {
        'directory': eval_dir.name,
        'method': method,
        'constraint': constraint,
        'short_constraint': short_constraint,
        'faithfulness': {
            'mean': mean_faithfulness,
            'sem': sem_faithfulness,
            'values': faithfulness_scores
        },
        'avg_word_count': avg_word_count
    }
    
    # Add less-strict scores if in less_strict_also mode
    if less_strict_also and faithfulness_scores_less_strict:
        mean_faithfulness_less_strict = np.mean(faithfulness_scores_less_strict)
        sem_faithfulness_less_strict = stats.sem(faithfulness_scores_less_strict) if len(faithfulness_scores_less_strict) > 1 else 0
        result['faithfulness_less_strict'] = {
            'mean': mean_faithfulness_less_strict,
            'sem': sem_faithfulness_less_strict,
            'values': faithfulness_scores_less_strict
        }
    
    return result

def shorten_length_constraint(constraint):
    """Shorten the length constraint for display."""
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
    elif "summary" in constraint_lower and "unconstrained" not in constraint_lower:
        return "summary"
    elif "unconstrained" in constraint_lower:
        return "summary"
    
    # Try to extract number from constraint
    match = re.search(r'(\d+)\s*words?', constraint_lower)
    if match:
        return f"<{match.group(1)}"
    return constraint[:10] + "..." if len(constraint) > 10 else constraint

def create_faithfulness_comparison_chart(dataset_filter="bmds", use_last_only=True, add_error_bars=False, less_strict=False, less_strict_also=False):
    """Create side-by-side comparison charts for faithfulness analysis."""
    
    # Load evaluation results
    results = load_faithfulness_results(dataset_filter, use_last_only, less_strict, less_strict_also)
    
    if not results:
        print(f"No {dataset_filter} faithfulness results found!")
        return []
    
    print(f"Found {len(results)} {dataset_filter} faithfulness evaluation results")
    
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
    
    metrics = ['faithfulness']
    metric_titles = ['Faithfulness']
    
    # Colors for different methods
    concat_color = '#FF7F50'  # Orange
    iterative_color = '#9370DB'  # Purple
    
    chart_files = []
    
    # Create separate chart for each metric
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        # Create individual figure for this metric
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(f'Faithfulness {title}: Concat vs Iterative ({dataset_filter.upper()})', fontsize=16, fontweight='bold')
        
        # Prepare data for grouped bar chart
        x = np.arange(len(expected_constraints))
        width = 0.35
        
        concat_means = []
        concat_sems = []
        iterative_means = []
        iterative_sems = []
        concat_less_strict_means = []
        iterative_less_strict_means = []
        x_labels = []
        
        for constraint in expected_constraints:
            # Find concat result for this constraint
            concat_result = next((r for r in concat_results if r['short_constraint'] == constraint), None)
            if concat_result:
                concat_means.append(concat_result[metric]['mean'])
                concat_sems.append(concat_result[metric]['sem'])
                concat_word_count = concat_result['avg_word_count']
                # Add less-strict data if available
                if less_strict_also and 'faithfulness_less_strict' in concat_result:
                    concat_less_strict_means.append(concat_result['faithfulness_less_strict']['mean'])
                else:
                    concat_less_strict_means.append(0)
            else:
                concat_means.append(0)
                concat_sems.append(0)
                concat_less_strict_means.append(0)
                concat_word_count = None
            
            # Find iterative result for this constraint
            iterative_result = next((r for r in iterative_results if r['short_constraint'] == constraint), None)
            if iterative_result:
                iterative_means.append(iterative_result[metric]['mean'])
                iterative_sems.append(iterative_result[metric]['sem'])
                iterative_word_count = iterative_result['avg_word_count']
                # Add less-strict data if available
                if less_strict_also and 'faithfulness_less_strict' in iterative_result:
                    iterative_less_strict_means.append(iterative_result['faithfulness_less_strict']['mean'])
                else:
                    iterative_less_strict_means.append(0)
            else:
                iterative_means.append(0)
                iterative_sems.append(0)
                iterative_less_strict_means.append(0)
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
            if less_strict_also:
                # Stacked bars: strict (bottom) + additional less-strict (top)
                concat_less_strict_mean = concat_less_strict_means[j]
                iter_less_strict_mean = iterative_less_strict_means[j]
                
                if concat_mean > 0:
                    # Bottom bar (strict score)
                    bar1_bottom = ax.bar(j - width/2, concat_mean, width, 
                                       color=concat_color, alpha=0.8,
                                       label='Concat (Strict)' if not concat_labeled else "")
                    # Top bar (additional less-strict score)
                    additional_concat = max(0, concat_less_strict_mean - concat_mean)
                    if additional_concat > 0:
                        bar1_top = ax.bar(j - width/2, additional_concat, width,
                                        bottom=concat_mean, color=concat_color, alpha=0.4,
                                        label='Concat (+Less Strict)' if not concat_labeled else "")
                    bars1_list.extend(bar1_bottom)
                    concat_labeled = True
                
                if iter_mean > 0:
                    # Bottom bar (strict score)
                    bar2_bottom = ax.bar(j + width/2, iter_mean, width,
                                       color=iterative_color, alpha=0.8,
                                       label='Iterative (Strict)' if not iterative_labeled else "")
                    # Top bar (additional less-strict score)
                    additional_iter = max(0, iter_less_strict_mean - iter_mean)
                    if additional_iter > 0:
                        bar2_top = ax.bar(j + width/2, additional_iter, width,
                                        bottom=iter_mean, color=iterative_color, alpha=0.4,
                                        label='Iterative (+Less Strict)' if not iterative_labeled else "")
                    bars2_list.extend(bar2_bottom)
                    iterative_labeled = True
            else:
                # Regular bars (original logic)
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
            ax.legend(loc='lower right')
        ax.set_ylim(0, 100.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for j, (concat_mean, concat_sem, iter_mean, iter_sem) in enumerate(zip(concat_means, concat_sems, iterative_means, iterative_sems)):
            if less_strict_also:
                # For stacked bars, show the total (less-strict) value at the top
                concat_less_strict_mean = concat_less_strict_means[j]
                iter_less_strict_mean = iterative_less_strict_means[j]
                
                if concat_mean > 0:
                    ax.text(j - width/2, concat_less_strict_mean + 1,
                           f'{concat_less_strict_mean:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                if iter_mean > 0:
                    ax.text(j + width/2, iter_less_strict_mean + 1,
                           f'{iter_less_strict_mean:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            else:
                # Regular labels for non-stacked bars
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
        chart_filename = f'../plots/faithfulness_{metric}_{dataset_filter}.png'
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
            print(f"  Concat    ({concat_result['avg_word_count']:.0f}w): F:{concat_result['faithfulness']['mean']:.1f}±{concat_result['faithfulness']['sem']:.1f}%")
        else:
            print("  Concat:    No data")
        
        if iterative_result:
            print(f"  Iterative ({iterative_result['avg_word_count']:.0f}w): F:{iterative_result['faithfulness']['mean']:.1f}±{iterative_result['faithfulness']['sem']:.1f}%")
        else:
            print("  Iterative: No data")
        
        # Show differences if both methods have data
        if concat_result and iterative_result:
            faith_diff = iterative_result['faithfulness']['mean'] - concat_result['faithfulness']['mean']
            print(f"  Difference (Iterative - Concat): F:{faith_diff:+.1f}%")
    
    return chart_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create faithfulness comparison charts")
    parser.add_argument("dataset", nargs="?", default="bmds", 
                       help="Dataset to analyze (bmds or true-detective)")
    parser.add_argument("--add-error-bars", action="store_true",
                       help="Add error bars to the charts")
    parser.add_argument("--use-all-summaries", action="store_true",
                       help="Use all summaries instead of just the last one for each item")
    parser.add_argument("--less-strict", action="store_true",
                       help="Use less strict scoring: count 'Yes' and 'PartialSupport' as faithful instead of just 'faithful' field")
    parser.add_argument("--less-strict-also", action="store_true",
                       help="Show both strict and less-strict scores on same chart (less-strict as lighter shade on top)")
    
    args = parser.parse_args()
    
    dataset = args.dataset.lower()
    if dataset not in ["bmds", "true-detective", "detectiveqa"]:
        print(f"Error: Unknown dataset '{dataset}'. Use 'bmds', 'true-detective', or 'detectiveqa'")
        exit(1)
    
    use_last_only = not args.use_all_summaries
    
    print(f"Creating faithfulness comparison charts for dataset: {dataset}")
    print(f"Using {'last summary only' if use_last_only else 'all summaries'} for each item")
    print(f"Error bars: {'enabled' if args.add_error_bars else 'disabled'}")
    
    if args.less_strict_also:
        print(f"Scoring method: both strict and less-strict (stacked bars)")
    elif args.less_strict:
        print(f"Scoring method: less strict (Yes + PartialSupport)")
    else:
        print(f"Scoring method: standard (faithful field)")
    
    chart_files = create_faithfulness_comparison_chart(dataset, use_last_only, args.add_error_bars, args.less_strict, args.less_strict_also)
    if chart_files:
        print(f"Comparison charts saved as:")
        for chart_file in chart_files:
            print(f"  - {chart_file}")
    else:
        print("No charts were created.")