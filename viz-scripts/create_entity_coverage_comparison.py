#!/usr/bin/env python3
"""Create side-by-side comparison charts for entity coverage analysis."""

import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def lighten_color(color, amount=0.5):
    """Lighten a color by blending with white."""
    try:
        c = mcolors.to_rgb(color)
        white = np.array([1, 1, 1])
        c = np.array(c)
        lightened = c + (white - c) * amount
        return mcolors.to_hex(lightened)
    except:
        return color


def darken_color(color, amount=0.3):
    """Darken a color by reducing brightness."""
    try:
        c = mcolors.to_rgb(color)
        c = np.array(c) * (1 - amount)
        return mcolors.to_hex(c)
    except:
        return color


def load_individual_entity_coverage_data(dataset_filter="bmds", use_last_only=True):
    """Load individual item-level entity coverage data with summary lengths."""
    base_path = Path("../outputs/eval/intrinsic/entity-coverage")
    
    all_data = []
    
    for eval_dir in base_path.iterdir():
        if not eval_dir.is_dir():
            continue
        
        # Filter by dataset
        if dataset_filter.lower() not in eval_dir.name.lower():
            continue
        
        # Load collection metadata
        collection_file = eval_dir / "collection.json"
        if not collection_file.exists():
            continue
        
        try:
            with open(collection_file, 'r') as f:
                collection = json.load(f)
            
            eval_info = collection.get('entity_coverage_multi_evaluation_info', {})
            meta = eval_info.get('collection_metadata', {})
            source_collection = meta.get('source_collection', '')
            
            if not source_collection:
                continue
            
            # Load source collection metadata to get summary length constraint
            source_collection_file = Path("..") / source_collection / "collection.json"
            if not source_collection_file.exists():
                continue
            
            with open(source_collection_file, 'r') as f:
                source_data = json.load(f)
            
            source_meta = source_data.get('summarization_info', {}).get('collection_metadata', {})
            length_constraint = source_meta.get('optional_summary_length', 'unknown')
            strategy = source_meta.get('strategy_function', 'unknown')
            
            # Determine method
            if 'concat' in strategy:
                method = 'concat'
            elif 'iterative' in strategy:
                method = 'iterative'
            else:
                continue
            
            # Shorten constraint for display
            short_constraint = shorten_length_constraint(length_constraint)
            
            # Load individual item data
            items_dir = eval_dir / "items"
            if not items_dir.exists():
                continue
            
            # Get list of item files to process
            if use_last_only:
                item_files = []
                for item_dir in items_dir.iterdir():
                    if item_dir.is_dir():
                        json_files = list(item_dir.glob("*.json"))
                        if json_files:
                            json_files.sort(key=lambda x: int(x.stem))
                            item_files.append(json_files[-1])
            else:
                item_files = list(items_dir.glob("*/*.json"))
            
            # Process each item
            for item_file in item_files:
                try:
                    with open(item_file, 'r') as f:
                        item_data = json.load(f)
                    
                    # Get metrics
                    entity_analysis = item_data.get('entity_analysis', {})
                    metrics = entity_analysis.get('metrics', {})
                    
                    if not metrics:
                        continue
                    
                    # Get item metadata
                    item_metadata = item_data.get('item_metadata', {})
                    item_id = item_metadata.get('item_id', 'unknown')
                    selected_indices = item_metadata.get('selected_indices', [])
                    total_chunks = item_metadata.get('total_chunks', 0)
                    
                    if not selected_indices:
                        continue
                    
                    # Load the actual summary text from the source collection
                    # The selected_indices tells us which summary to use (0-indexed)
                    summary_index = selected_indices[0] if len(selected_indices) == 1 else selected_indices[-1]
                    
                    # When use_last_only=True, we should be getting the last chunk's summary
                    if use_last_only and total_chunks > 0:
                        # For last summary, the index should be total_chunks - 1 (0-indexed)
                        expected_last_index = total_chunks - 1
                        assert summary_index == expected_last_index, \
                            f"Expected last summary index {expected_last_index} but got {summary_index} for item {item_id}"
                    
                    summary_item_file = Path("..") / source_collection / "items" / f"{item_id}.json"
                    if not summary_item_file.exists():
                        print(f"Warning: Summary file not found: {summary_item_file}")
                        continue
                    
                    with open(summary_item_file, 'r') as f:
                        summary_data = json.load(f)
                    
                    # Navigate to the summary text
                    # Structure: documents[0].summaries[summary_index]
                    documents = summary_data.get('documents', [])
                    if not documents:
                        continue
                    
                    summaries = documents[0].get('summaries', [])
                    if not summaries or summary_index >= len(summaries):
                        continue
                    
                    # Verify we have the expected number of summaries
                    if total_chunks > 0:
                        assert len(summaries) == total_chunks, \
                            f"Expected {total_chunks} summaries but found {len(summaries)} for item {item_id}"
                    
                    # Get the summary at the selected index
                    summary_text = summaries[summary_index]
                    
                    # Additional verification: when use_last_only, we should be getting the last summary
                    if use_last_only and total_chunks > 0:
                        assert summary_index == len(summaries) - 1, \
                            f"use_last_only=True but not using last summary: index {summary_index} vs last {len(summaries)-1} for item {item_id}"
                    
                    # Count actual words
                    word_count = len(summary_text.split())
                    
                    all_data.append({
                        'constraint': short_constraint,
                        'method': method,
                        'word_count': word_count,
                        'precision': metrics.get('precision', 0) * 100,
                        'recall': metrics.get('recall', 0) * 100,
                        'jaccard': metrics.get('jaccard_similarity', 0) * 100,
                        'item_id': item_id
                    })
                    
                except Exception as e:
                    print(f"Warning: Could not process {item_file}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Warning: Could not process {eval_dir.name}: {e}")
            continue
    
    return all_data


def shorten_length_constraint(constraint):
    """Shorten the length constraint for display."""
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
        match = re.search(r'(\d+)\s*words?', constraint_lower)
        if match:
            return f"<{match.group(1)}"
        return constraint[:10] + "..." if len(constraint) > 10 else constraint


def create_entity_coverage_scatterplots(dataset_filter="bmds", use_last_only=True):
    """Create scatterplots showing individual entity coverage scores vs summary length."""
    
    print(f"\nCreating entity coverage scatterplots for {dataset_filter}...")
    
    # Load individual item data
    data = load_individual_entity_coverage_data(dataset_filter, use_last_only)
    
    if not data:
        print(f"No individual entity coverage data found for {dataset_filter}!")
        return []
    
    print(f"Loaded {len(data)} individual data points")
    
    # Define constraint order and base colors
    if dataset_filter.lower() in ['bmds', 'detectiveqa']:
        expected_constraints = ['<50', '<100', '<200', '<500', 'summary', '<1000', 'long']
    elif dataset_filter.lower() == 'true-detective':
        expected_constraints = ['<50', '<100', '<200', 'summary', '<500', '<1000', 'long']
    else:
        expected_constraints = ['<50', '<100', '<200', '<500', 'summary', '<1000', 'long']
    
    # Define color pairs for each constraint: (concat_color, iterative_color)
    # Format: lighter shade for concat, darker shade for iterative
    color_pairs = {
        '<50': ('#808080', '#000000'),           # grey vs black
        '<100': ('#FF69B4', '#8B0000'),          # pink vs dark red
        '<200': ('#87CEEB', '#00008B'),          # light blue vs dark blue
        '<500': ('#90EE90', '#006400'),          # lime green vs dark green
        'summary': ('#FFD700', '#FF4500'),       # light yellow orange vs red orange
        '<1000': ('#DDA0DD', '#4B0082'),         # light purple vs dark purple
        'long': ('#FFFFE0', '#B8860B')           # light yellow vs darkish yellow
    }
    
    # Create color mapping
    colors = {}
    for constraint in expected_constraints:
        if constraint in color_pairs:
            concat_color, iterative_color = color_pairs[constraint]
            colors[(constraint, 'concat')] = concat_color
            colors[(constraint, 'iterative')] = iterative_color
    
    metrics = ['precision', 'recall', 'jaccard']
    metric_titles = ['Precision', 'Recall', 'Jaccard Similarity']
    
    chart_files = []
    
    # Create separate scatterplot for each metric
    for metric, title in zip(metrics, metric_titles):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'Entity Coverage {title} vs Summary Length ({dataset_filter.upper()})', 
                     fontsize=16, fontweight='bold')
        
        # Plot data points for each constraint and method
        for constraint in expected_constraints:
            for method in ['concat', 'iterative']:
                # Filter data for this constraint and method
                filtered_data = [d for d in data 
                                if d['constraint'] == constraint and d['method'] == method]
                
                if not filtered_data:
                    continue
                
                word_counts = [d['word_count'] for d in filtered_data]
                metric_values = [d[metric] for d in filtered_data]
                
                color = colors.get((constraint, method), '#808080')
                marker = 'o' if method == 'concat' else 's'  # circle for concat, square for iterative
                alpha = 0.7
                size = 60
                
                # Create label for legend (only once per constraint-method combo)
                display_constraint = "unconstrained" if constraint == "summary" else constraint
                label = f"{display_constraint} ({method})"
                
                ax.scatter(word_counts, metric_values, 
                          c=color, marker=marker, s=size, alpha=alpha, 
                          label=label, edgecolors='black', linewidths=1.0)
        
        ax.set_xlabel('Summary Length (words)', fontsize=12)
        ax.set_ylabel(f'{title} (%)', fontsize=12)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        
        # Create legend with smaller font and multiple columns
        ax.legend(loc='best', fontsize=8, ncol=2, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = f'../plots/entity_coverage_{metric}_scatter_{dataset_filter}.png'
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        chart_files.append(chart_filename)
        print(f"Saved: {chart_filename}")
        plt.show()
        plt.close()
    
    return chart_files


def create_single_item_line_plot(dataset_filter="bmds", item_id=None, use_last_only=True):
    """Create line plots for a single item showing concat vs iterative across constraints."""
    
    print(f"\nCreating single-item line plots for {dataset_filter}, item: {item_id}...")
    
    # Load individual item data
    data = load_individual_entity_coverage_data(dataset_filter, use_last_only)
    
    if not data:
        print(f"No individual entity coverage data found for {dataset_filter}!")
        return []
    
    # Filter for the specific item
    if item_id:
        item_data = [d for d in data if d['item_id'] == item_id]
        if not item_data:
            print(f"No data found for item {item_id}!")
            available_items = sorted(set(d['item_id'] for d in data))
            print(f"Available items: {', '.join(available_items[:10])}...")
            return []
    else:
        # If no item specified, pick the first one
        item_id = data[0]['item_id']
        item_data = [d for d in data if d['item_id'] == item_id]
        print(f"No item specified, using: {item_id}")
    
    print(f"Found {len(item_data)} data points for item {item_id}")
    
    # Define constraint order
    if dataset_filter.lower() in ['bmds', 'detectiveqa']:
        expected_constraints = ['<50', '<100', '<200', '<500', 'summary', '<1000', 'long']
    elif dataset_filter.lower() == 'true-detective':
        expected_constraints = ['<50', '<100', '<200', 'summary', '<500', '<1000', 'long']
    else:
        expected_constraints = ['<50', '<100', '<200', '<500', 'summary', '<1000', 'long']
    
    # Organize data by method and constraint
    concat_data = {}
    iterative_data = {}
    
    for d in item_data:
        constraint = d['constraint']
        if constraint not in expected_constraints:
            continue
        
        if d['method'] == 'concat':
            concat_data[constraint] = d
        elif d['method'] == 'iterative':
            iterative_data[constraint] = d
    
    metrics = ['precision', 'recall', 'jaccard']
    metric_titles = ['Precision', 'Recall', 'Jaccard Similarity']
    
    chart_files = []
    
    # Create separate line plot for each metric
    for metric, title in zip(metrics, metric_titles):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'{title} vs Summary Length: Item {item_id} ({dataset_filter.upper()})', 
                     fontsize=16, fontweight='bold')
        
        # Prepare data for concat
        concat_constraints = []
        concat_word_counts = []
        concat_metric_values = []
        
        for constraint in expected_constraints:
            if constraint in concat_data:
                concat_constraints.append(constraint)
                concat_word_counts.append(concat_data[constraint]['word_count'])
                concat_metric_values.append(concat_data[constraint][metric])
        
        # Prepare data for iterative
        iterative_constraints = []
        iterative_word_counts = []
        iterative_metric_values = []
        
        for constraint in expected_constraints:
            if constraint in iterative_data:
                iterative_constraints.append(constraint)
                iterative_word_counts.append(iterative_data[constraint]['word_count'])
                iterative_metric_values.append(iterative_data[constraint][metric])
        
        # Plot concat line
        if concat_word_counts:
            ax.plot(concat_word_counts, concat_metric_values, 
                   marker='o', markersize=10, linewidth=2.5, 
                   color='#2E86AB', linestyle='-', 
                   label='Concat', markeredgecolor='black', markeredgewidth=1.5)
            
            # Add constraint labels
            for i, (wc, val, constraint) in enumerate(zip(concat_word_counts, concat_metric_values, concat_constraints)):
                display_constraint = "unc" if constraint == "summary" else constraint
                ax.annotate(display_constraint, 
                           xy=(wc, val), 
                           xytext=(0, 10), 
                           textcoords='offset points',
                           ha='center', 
                           fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Plot iterative line
        if iterative_word_counts:
            ax.plot(iterative_word_counts, iterative_metric_values, 
                   marker='s', markersize=10, linewidth=2.5, 
                   color='#A23B72', linestyle='--', 
                   label='Iterative', markeredgecolor='black', markeredgewidth=1.5)
            
            # Add constraint labels
            for i, (wc, val, constraint) in enumerate(zip(iterative_word_counts, iterative_metric_values, iterative_constraints)):
                display_constraint = "unc" if constraint == "summary" else constraint
                ax.annotate(display_constraint, 
                           xy=(wc, val), 
                           xytext=(0, -15), 
                           textcoords='offset points',
                           ha='center', 
                           fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.set_xlabel('Summary Length (words)', fontsize=12)
        ax.set_ylabel(f'{title} (%)', fontsize=12)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = f'../plots/entity_coverage_{metric}_line_{item_id}_{dataset_filter}.png'
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        chart_files.append(chart_filename)
        print(f"Saved: {chart_filename}")
        plt.show()
        plt.close()
    
    return chart_files


def create_violin_plots(dataset_filter="bmds", use_last_only=True):
    """Create violin plots showing distribution of summary lengths across constraints."""
    
    print(f"\nCreating violin plots for {dataset_filter}...")
    
    # Load individual item data
    data = load_individual_entity_coverage_data(dataset_filter, use_last_only)
    
    if not data:
        print(f"No individual entity coverage data found for {dataset_filter}!")
        return []
    
    print(f"Loaded {len(data)} individual data points")
    
    # Define constraint order
    if dataset_filter.lower() in ['bmds', 'detectiveqa']:
        expected_constraints = ['<50', '<100', '<200', '<500', 'summary', '<1000', 'long']
    elif dataset_filter.lower() == 'true-detective':
        expected_constraints = ['<50', '<100', '<200', 'summary', '<500', '<1000', 'long']
    else:
        expected_constraints = ['<50', '<100', '<200', '<500', 'summary', '<1000', 'long']
    
    # Filter to only constraints that exist in the data
    available_constraints = sorted(set(d['constraint'] for d in data))
    expected_constraints = [c for c in expected_constraints if c in available_constraints]
    
    print(f"Available constraints: {', '.join(expected_constraints)}")
    
    chart_files = []
    
    # Create a single violin plot showing summary length distribution
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle(f'Summary Length Distribution by Constraint ({dataset_filter.upper()})', 
                 fontsize=16, fontweight='bold')
    
    # Prepare data for violin plots
    labels = []
    all_data = []
    all_positions = []
    colors = []
    
    concat_color = '#2E86AB'  # Blue
    iterative_color = '#A23B72'  # Purple
    
    for i, constraint in enumerate(expected_constraints):
        base_pos = i * 3  # Space out constraint groups
        
        # Get concat word counts for this constraint
        concat_word_counts = [d['word_count'] for d in data 
                             if d['constraint'] == constraint and d['method'] == 'concat']
        
        # Get iterative word counts for this constraint
        iterative_word_counts = [d['word_count'] for d in data 
                                if d['constraint'] == constraint and d['method'] == 'iterative']
        
        if concat_word_counts:
            all_data.append(concat_word_counts)
            all_positions.append(base_pos - 0.4)
            colors.append(concat_color)
        
        if iterative_word_counts:
            all_data.append(iterative_word_counts)
            all_positions.append(base_pos + 0.4)
            colors.append(iterative_color)
        
        # Label for x-axis
        if concat_word_counts or iterative_word_counts:
            display_constraint = "unc" if constraint == "summary" else constraint
            labels.append((base_pos, display_constraint))
    
    # Plot violins
    if all_data:
        parts = ax.violinplot(all_data, positions=all_positions, widths=0.7,
                             showmeans=False, showmedians=False, showextrema=False)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Add scatter points on top of violins
        for i, (pos, values, color) in enumerate(zip(all_positions, all_data, colors)):
            # Add jitter to x-coordinates for visibility
            jitter = np.random.normal(0, 0.05, size=len(values))
            x_coords = [pos + j for j in jitter]
            
            ax.scatter(x_coords, values, 
                      alpha=0.6, s=40, color=color, 
                      edgecolors='black', linewidths=0.8, zorder=3)
        
        # Add median lines
        for pos, values, color in zip(all_positions, all_data, colors):
            median = np.median(values)
            ax.plot([pos - 0.3, pos + 0.3], [median, median], 
                   color='white', linewidth=2.5, zorder=4)
            ax.plot([pos - 0.3, pos + 0.3], [median, median], 
                   color='black', linewidth=1.5, zorder=5)
    
    # Set x-axis labels
    ax.set_xticks([pos for pos, _ in labels])
    ax.set_xticklabels([label for _, label in labels], fontsize=11)
    ax.set_xlabel('Length Constraint', fontsize=12)
    ax.set_ylabel('Summary Length (words)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=concat_color, edgecolor='black', alpha=0.6, label='Concat'),
        Patch(facecolor=iterative_color, edgecolor='black', alpha=0.6, label='Iterative')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save chart
    chart_filename = f'../plots/summary_length_violin_{dataset_filter}.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    chart_files.append(chart_filename)
    print(f"Saved: {chart_filename}")
    plt.show()
    plt.close()
    
    return chart_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create entity coverage comparison charts")
    parser.add_argument("dataset", nargs="?", default="bmds", 
                       help="Dataset to analyze (bmds or true-detective)")
    parser.add_argument("--add-error-bars", action="store_true",
                       help="Add error bars to the charts")
    parser.add_argument("--use-all-summaries", action="store_true", default=False,
                       help="Use all summaries instead of just the last one for each item (default: False, use last only)")
    parser.add_argument("--scatterplot", action="store_true",
                       help="Create scatterplots showing individual data points")
    parser.add_argument("--bar-chart", action="store_true",
                       help="Create bar charts showing averages (default if neither specified)")
    parser.add_argument("--single-item", type=str, metavar="ITEM_ID",
                       help="Create line plot for a single item (e.g., ADP02)")
    parser.add_argument("--violin", action="store_true",
                       help="Create violin plots showing distribution across constraints")
    
    args = parser.parse_args()
    
    dataset = args.dataset.lower()
    if dataset not in ["bmds", "true-detective", "detectiveqa"]:
        print(f"Error: Unknown dataset '{dataset}'. Use 'bmds', 'true-detective', or 'detectiveqa'")
        exit(1)
    
    use_last_only = not args.use_all_summaries
    
    # Handle single-item plot separately
    if args.single_item:
        print(f"Creating single-item line plot for dataset: {dataset}, item: {args.single_item}")
        print(f"Using {'last summary only' if use_last_only else 'all summaries'} for each item")
        
        chart_files = create_single_item_line_plot(dataset, args.single_item, use_last_only)
        if chart_files:
            print(f"\nSingle-item line plots saved as:")
            for chart_file in chart_files:
                print(f"  - {chart_file}")
        else:
            print("\nNo charts were created.")
        exit(0)
    
    # Handle violin plots
    if args.violin:
        print(f"Creating violin plots for dataset: {dataset}")
        print(f"Using {'last summary only' if use_last_only else 'all summaries'} for each item")
        
        chart_files = create_violin_plots(dataset, use_last_only)
        if chart_files:
            print(f"\nViolin plots saved as:")
            for chart_file in chart_files:
                print(f"  - {chart_file}")
        else:
            print("\nNo charts were created.")
        exit(0)
    
    # If neither is specified, create both
    create_bar = args.bar_chart or (not args.scatterplot and not args.bar_chart and not args.violin)
    create_scatter = args.scatterplot or (not args.scatterplot and not args.bar_chart and not args.violin)
    
    print(f"Creating entity coverage comparison charts for dataset: {dataset}")
    print(f"Using {'last summary only' if use_last_only else 'all summaries'} for each item")
    print(f"Error bars: {'enabled' if args.add_error_bars else 'disabled'}")
    
    all_chart_files = []
    
    if create_bar:
        print("\n" + "="*60)
        print("Creating bar charts (averages)...")
        print("="*60)
        chart_files = create_entity_coverage_comparison_chart(dataset, use_last_only, args.add_error_bars)
        if chart_files:
            print(f"\nBar charts saved as:")
            for chart_file in chart_files:
                print(f"  - {chart_file}")
            all_chart_files.extend(chart_files)
        else:
            print("No bar charts were created.")
    
    if create_scatter:
        print("\n" + "="*60)
        print("Creating scatterplots (individual data points)...")
        print("="*60)
        scatter_files = create_entity_coverage_scatterplots(dataset, use_last_only)
        if scatter_files:
            print(f"\nScatterplots saved as:")
            for scatter_file in scatter_files:
                print(f"  - {scatter_file}")
            all_chart_files.extend(scatter_files)
        else:
            print("No scatterplots were created.")
    
    if all_chart_files:
        print(f"\n{'='*60}")
        print(f"Total charts created: {len(all_chart_files)}")
        print(f"{'='*60}")
    else:
        print("\nNo charts were created.")