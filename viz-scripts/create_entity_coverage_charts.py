#!/usr/bin/env python3
"""Create visualizations for entity coverage analysis."""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import re

logger = logging.getLogger(__name__)

def shorten_length_constraint(constraint):
    """Shorten the length constraint for display, based on pattern from create_summary_length_charts.py."""
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
        match = re.search(r'(\d+)\s*words?', constraint_lower)
        if match:
            return f"<{match.group(1)}"
        return constraint[:10] + "..." if len(constraint) > 10 else constraint

def analyze_entity_coverage_directory(dir_path: Path, use_last_only: bool = True) -> Dict:
    """Analyze a single entity coverage evaluation directory."""
    
    # Load collection metadata
    collection_file = dir_path / "collection.json"
    if not collection_file.exists():
        return None
        
    with open(collection_file, 'r') as f:
        collection = json.load(f)
    
    eval_info = collection.get('entity_coverage_multi_evaluation_info', {})
    meta = eval_info.get('collection_metadata', {})
    processing_stats = eval_info.get('processing_stats', {})
    
    # Check for successful items, but don't skip immediately - let's investigate
    if processing_stats.get('successful_items', 0) == 0:
        print(f"Warning: {dir_path.name} reports no successful items, checking actual item files...")
    
    # Extract source collection from both metadata and command_run for validation
    source_collection = meta.get('source_collection', '')
    command_run = meta.get('command_run', '')
    
    if not source_collection:
        print(f"Warning: No source collection for {dir_path.name}")
        return None
    
    # Validate that source_collection matches what's in command_run
    if command_run and '--input' in command_run:
        # Extract input path from command_run
        import shlex
        try:
            args = shlex.split(command_run)
            input_idx = args.index('--input')
            if input_idx + 1 < len(args):
                command_input = args[input_idx + 1]
                # Add trailing slash for comparison if missing
                if not command_input.endswith('/'):
                    command_input += '/'
                if not source_collection.endswith('/'):
                    source_collection += '/'
                    
                if command_input != source_collection:
                    print(f"Warning: source_collection ({source_collection}) doesn't match command_run input ({command_input}) for {dir_path.name}")
        except Exception as e:
            print(f"Warning: Could not parse command_run for validation in {dir_path.name}: {e}")
    
    # Load source collection metadata to get summary length constraint
    # Adjust path to be relative to the parent directory since we're in viz-scripts
    source_collection_file = Path("..") / source_collection / "collection.json"
    length_constraint = "unknown"
    if source_collection_file.exists():
        try:
            with open(source_collection_file, 'r') as f:
                source_data = json.load(f)
            source_meta = source_data.get('summarization_info', {}).get('collection_metadata', {})
            length_constraint = source_meta.get('optional_summary_length', 'unknown')
        except Exception as e:
            print(f"Warning: Could not read source collection for {dir_path.name}: {e}")
            return None
    else:
        print(f"Warning: Source collection file not found: {source_collection_file}")
        return None
    
    # Collect metrics from all items and calculate average text length
    items_dir = dir_path / "items"
    if not items_dir.exists():
        return None
    
    metrics_list = []
    total_items = 0
    total_text_length = 0
    
    # Get list of item files to process
    if use_last_only:
        # For each item directory, get only the highest numbered JSON file
        item_files = []
        for item_dir in items_dir.iterdir():
            if item_dir.is_dir():
                json_files = list(item_dir.glob("*.json"))
                if json_files:
                    # Sort by numeric value in filename and take the last one
                    json_files.sort(key=lambda x: int(x.stem))
                    item_files.append(json_files[-1])
    else:
        # Use all JSON files
        item_files = list(items_dir.glob("*/*.json"))
    
    for item_file in item_files:
        try:
            with open(item_file, 'r') as f:
                item_data = json.load(f)
            
            entity_analysis = item_data.get('entity_analysis', {})
            metrics = entity_analysis.get('metrics', {})
            
            if metrics:
                total_items += 1
                # Get text length from item metadata
                text_length = item_data.get('item_metadata', {}).get('selected_text_length', 0)
                total_text_length += text_length
                
                metrics_list.append({
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'jaccard_similarity': metrics.get('jaccard_similarity', 0),
                    'num_source_entities': metrics.get('num_source_entities', 0),
                    'num_summary_entities': metrics.get('num_summary_entities', 0),
                    'num_matched_entities': metrics.get('num_matched_entities', 0),
                    'text_length': text_length
                })
            else:
                # Check if the item has the raw data but missing final analysis
                if ('source_entities' in item_data and 
                    'summary_entities' in item_data and 
                    'matching_metadata' in item_data):
                    print(f"  Warning: {item_file.name} has matching data but missing entity_analysis section")
                else:
                    print(f"  Warning: {item_file.name} appears incomplete (missing core sections)")
        except Exception as e:
            print(f"Error processing {item_file}: {e}")
            continue
    
    if not metrics_list:
        print(f"Skipping {dir_path.name}: no items with complete entity_analysis metrics found")
        return None
    
    # Calculate aggregated metrics and average word count
    precision_scores = [m['precision'] for m in metrics_list]
    recall_scores = [m['recall'] for m in metrics_list]
    jaccard_scores = [m['jaccard_similarity'] for m in metrics_list]
    avg_text_length = total_text_length / total_items if total_items > 0 else 0
    avg_word_count = int(avg_text_length / 5)  # Approximate words (5 chars per word)
    
    # Determine method (concat vs iterative)
    method = "unknown"
    if 'concat' in dir_path.name:
        method = "concat"
    elif 'iterative' in dir_path.name:
        method = "iterative"
    
    return {
        'directory': dir_path.name,
        'method': method,
        'length_constraint': length_constraint,
        'short_constraint': shorten_length_constraint(length_constraint),
        'source_collection': source_collection,
        'total_items': total_items,
        'avg_text_length': avg_text_length,
        'avg_word_count': avg_word_count,
        'precision': {
            'mean': np.mean(precision_scores),
            'std': np.std(precision_scores),
            'sem': np.std(precision_scores) / np.sqrt(len(precision_scores)),  # Standard error of the mean
            'values': precision_scores
        },
        'recall': {
            'mean': np.mean(recall_scores),
            'std': np.std(recall_scores),
            'sem': np.std(recall_scores) / np.sqrt(len(recall_scores)),
            'values': recall_scores
        },
        'jaccard': {
            'mean': np.mean(jaccard_scores),
            'std': np.std(jaccard_scores),
            'sem': np.std(jaccard_scores) / np.sqrt(len(jaccard_scores)),
            'values': jaccard_scores
        }
    }

def load_entity_coverage_results(dataset_filter="bmds", use_last_only: bool = True) -> List[Dict]:
    """Load and analyze all entity coverage evaluation results."""
    base_path = Path("../outputs/eval/intrinsic/entity-coverage")
    results = []
    
    for eval_dir in base_path.iterdir():
        if not eval_dir.is_dir():
            continue
        
        # Filter by dataset
        if dataset_filter.lower() not in eval_dir.name.lower():
            continue
        
        try:
            result = analyze_entity_coverage_directory(eval_dir, use_last_only)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing {eval_dir.name}: {e}")
    
    return results

def create_entity_coverage_charts(dataset_filter="bmds"):
    """Create charts for entity coverage analysis for specified dataset."""
    
    # Load evaluation results
    results = load_entity_coverage_results(dataset_filter)
    
    if not results:
        print(f"No {dataset_filter} entity coverage results found!")
        return
    
    print(f"Found {len(results)} {dataset_filter} entity coverage evaluation results")
    
    # Group results by method
    concat_results = [r for r in results if r['method'] == 'concat']
    iterative_results = [r for r in results if r['method'] == 'iterative']
    
    # Sort by short constraint for consistent ordering
    concat_results.sort(key=lambda x: x['short_constraint'])
    iterative_results.sort(key=lambda x: x['short_constraint'])
    
    print(f"Concat results: {len(concat_results)}")
    for r in concat_results:
        print(f"  {r['directory']} -> {r['short_constraint']} ({r['avg_word_count']} words) -> P:{r['precision']['mean']:.3f}, R:{r['recall']['mean']:.3f}, J:{r['jaccard']['mean']:.3f}")
    
    print(f"Iterative results: {len(iterative_results)}")
    for r in iterative_results:
        print(f"  {r['directory']} -> {r['short_constraint']} ({r['avg_word_count']} words) -> P:{r['precision']['mean']:.3f}, R:{r['recall']['mean']:.3f}, J:{r['jaccard']['mean']:.3f}")
    
    # Define all expected constraint categories in the order we want to display them
    expected_constraints = ['<200', '<500', 'summary', 'long']
    constraint_order = {constraint: i for i, constraint in enumerate(expected_constraints)}
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Entity Coverage Analysis: {dataset_filter.upper()}', fontsize=16, fontweight='bold')
    
    # Metrics to plot
    metrics = ['precision', 'recall', 'jaccard']
    metric_titles = ['Precision', 'Recall', 'Jaccard Similarity']
    
    # Colors for different methods
    concat_color = '#FF7F50'  # Orange
    iterative_color = '#9370DB'  # Purple
    
    # Prepare data for concat results (ensuring all constraints are represented)
    concat_data_by_constraint = {}
    for r in concat_results:
        constraint = r['short_constraint']
        concat_data_by_constraint[constraint] = r
    
    # Prepare data for iterative results (ensuring all constraints are represented)
    iterative_data_by_constraint = {}
    for r in iterative_results:
        constraint = r['short_constraint']
        iterative_data_by_constraint[constraint] = r
    
    # Top row: Concat results
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[0, i]
        
        x_labels = []
        means = []
        sems = []
        
        for constraint in expected_constraints:
            if constraint in concat_data_by_constraint:
                r = concat_data_by_constraint[constraint]
                x_labels.append(f"{constraint}\n({r['avg_word_count']}w)")
                means.append(r[metric]['mean'])
                sems.append(r[metric]['sem'])
            else:
                x_labels.append(f"{constraint}\n(N/A)")
                means.append(0)
                sems.append(0)
        
        # Create bars, but only show those with data
        bars = []
        for j, (mean, sem) in enumerate(zip(means, sems)):
            if mean > 0:  # Only show bars with data
                bar = ax.bar(j, mean, yerr=sem, color=concat_color, alpha=0.8, capsize=5, width=0.6)
                bars.extend(bar)
                # Add value labels on bars
                ax.text(j, mean + sem + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'Concat: {title}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Bottom row: Iterative results
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[1, i]
        
        x_labels = []
        means = []
        sems = []
        
        for constraint in expected_constraints:
            if constraint in iterative_data_by_constraint:
                r = iterative_data_by_constraint[constraint]
                x_labels.append(f"{constraint}\n({r['avg_word_count']}w)")
                means.append(r[metric]['mean'])
                sems.append(r[metric]['sem'])
            else:
                x_labels.append(f"{constraint}\n(N/A)")
                means.append(0)
                sems.append(0)
        
        # Create bars, but only show those with data
        bars = []
        for j, (mean, sem) in enumerate(zip(means, sems)):
            if mean > 0:  # Only show bars with data
                bar = ax.bar(j, mean, yerr=sem, color=iterative_color, alpha=0.8, capsize=5, width=0.6)
                bars.extend(bar)
                # Add value labels on bars
                ax.text(j, mean + sem + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'Iterative: {title}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add x-axis label to bottom row
        ax.set_xlabel('Summary Length Constraint (word count)', fontsize=12)
    
    plt.tight_layout()
    chart_filename = f'../plots/entity_coverage_charts_{dataset_filter}.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nEntity Coverage Analysis Summary:")
    print("=" * 50)
    
    if concat_results:
        print(f"\nConcat Summaries ({len(concat_results)} evaluations):")
        for r in concat_results:
            print(f"  {r['short_constraint']:>6}: P:{r['precision']['mean']:.3f}±{r['precision']['sem']:.3f}, "
                  f"R:{r['recall']['mean']:.3f}±{r['recall']['sem']:.3f}, "
                  f"J:{r['jaccard']['mean']:.3f}±{r['jaccard']['sem']:.3f} ({r['total_items']} items)")
        
        # Overall averages for concat
        concat_prec_means = [r['precision']['mean'] for r in concat_results]
        concat_rec_means = [r['recall']['mean'] for r in concat_results]
        concat_jacc_means = [r['jaccard']['mean'] for r in concat_results]
        print(f"  Overall Concat Averages: P:{np.mean(concat_prec_means):.3f}, "
              f"R:{np.mean(concat_rec_means):.3f}, J:{np.mean(concat_jacc_means):.3f}")
    
    if iterative_results:
        print(f"\nIterative Summaries ({len(iterative_results)} evaluations):")
        for r in iterative_results:
            print(f"  {r['short_constraint']:>6}: P:{r['precision']['mean']:.3f}±{r['precision']['sem']:.3f}, "
                  f"R:{r['recall']['mean']:.3f}±{r['recall']['sem']:.3f}, "
                  f"J:{r['jaccard']['mean']:.3f}±{r['jaccard']['sem']:.3f} ({r['total_items']} items)")
        
        # Overall averages for iterative
        iter_prec_means = [r['precision']['mean'] for r in iterative_results]
        iter_rec_means = [r['recall']['mean'] for r in iterative_results]
        iter_jacc_means = [r['jaccard']['mean'] for r in iterative_results]
        print(f"  Overall Iterative Averages: P:{np.mean(iter_prec_means):.3f}, "
              f"R:{np.mean(iter_rec_means):.3f}, J:{np.mean(iter_jacc_means):.3f}")
    
    return chart_filename

if __name__ == "__main__":
    import sys
    
    # Parse command line argument for dataset
    dataset = "bmds"  # Default
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
        if dataset not in ["bmds", "true-detective"]:
            print(f"Error: Unknown dataset '{dataset}'. Use 'bmds' or 'true-detective'")
            sys.exit(1)
    
    print(f"Creating entity coverage charts for dataset: {dataset}")
    chart_file = create_entity_coverage_charts(dataset)
    print(f"Chart saved as: {chart_file}")