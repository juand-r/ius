#!/usr/bin/env python3
"""Create charts showing ROUGE continuity precision scores vs summary length for concat and iterative methods."""

import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import pandas as pd
from pathlib import Path
import scipy.stats as stats
from collections import defaultdict

def get_continuity_data(dataset_filter="bmds"):
    """Extract ROUGE continuity data from the evaluation results and their source collections."""
    
    base_path = Path("../outputs/eval/intrinsic/rouge-continuity")
    concat_data = []
    iterative_data = []
    
    # Scan for directories matching the dataset filter
    all_eval_dirs = []
    
    for eval_dir in base_path.iterdir():
        if not eval_dir.is_dir():
            continue
        if not eval_dir.name.startswith(dataset_filter):
            continue
        all_eval_dirs.append(eval_dir)
    
    if not all_eval_dirs:
        print(f"No {dataset_filter} rouge-continuity evaluation directories found!")
        return [], []
    
    print(f"Found {len(all_eval_dirs)} {dataset_filter} rouge-continuity evaluation directories")
    
    for eval_dir in all_eval_dirs:
        print(f"Processing: {eval_dir.name}")
        
        # Load evaluation collection metadata
        collection_file = eval_dir / "collection.json"
        if not collection_file.exists():
            print(f"  No collection.json found, skipping")
            continue
            
        with open(collection_file, 'r') as f:
            eval_collection = json.load(f)
        
        # Extract source collection path from input_path
        input_path = eval_collection.get('input_path', '')
        if not input_path:
            print(f"  No input_path found, skipping")
            continue
        
        # Extract range specification from parameters
        continuity_info = eval_collection.get('rouge_continuity_evaluation_info', {})
        range_spec = continuity_info.get('parameters', {}).get('range_spec', '')
        
        if not range_spec:
            print(f"  No range_spec found, skipping")
            continue
            
        print(f"  Input path: {input_path}")
        print(f"  Range spec: {range_spec}")
        
        # Load source collection metadata to get length constraint
        source_collection_file = Path("..") / input_path / "collection.json"
        if not source_collection_file.exists():
            print(f"  Source collection not found: {source_collection_file}, skipping")
            continue
        
        with open(source_collection_file, 'r') as f:
            source_data = json.load(f)
        
        source_meta = source_data.get('summarization_info', {}).get('collection_metadata', {})
        length_constraint = source_meta.get('optional_summary_length', '')
        
        # Load continuity results from individual items
        items_dir = eval_dir / "items"
        if not items_dir.exists():
            print(f"  No items directory found, skipping")
            continue
        
        # Collect continuity scores and calculate word counts from source summaries
        rouge_scores = {'rougeLsum': [], 'rougeL': [], 'rouge2': []}
        total_words = 0
        total_chars = 0
        valid_items = 0
        
        for item_file in items_dir.glob("*.json"):
            with open(item_file, 'r') as f:
                item_data = json.load(f)
            
            # Extract continuity scores
            continuity_results = item_data.get('rouge_continuity_results', {})
            continuity_averages = continuity_results.get('continuity_averages', {})
            
            # Collect scores for each ROUGE metric
            for rouge_metric in ['rougeLsum', 'rougeL', 'rouge2']:
                continuity_key = f"{rouge_metric}-continuity"
                if continuity_key in continuity_averages:
                    precision_score = continuity_averages[continuity_key].get('precision', 0)
                    if precision_score > 0:  # Only include valid scores
                        rouge_scores[rouge_metric].append(precision_score)
            
            # Calculate word counts from source summaries
            item_id = item_file.stem
            source_items_dir = Path("..") / input_path / "items"
            source_item_file = source_items_dir / f"{item_id}.json"
            
            if source_item_file.exists():
                with open(source_item_file, 'r') as f:
                    source_item = json.load(f)
                
                # Get summary text based on range_spec
                documents = source_item.get('documents', [])
                if documents:
                    summaries = documents[0].get('summaries', [])
                    if summaries:
                        summary_texts = []
                        
                        if range_spec == 'all':
                            summary_texts = summaries
                        elif range_spec == 'all-but-last':
                            if len(summaries) >= 2:
                                summary_texts = summaries[:-1]
                        
                        # Count words and chars for the selected summaries
                        for summary_text in summary_texts:
                            if summary_text:
                                word_count = len(summary_text.split())
                                char_count = len(summary_text)
                                total_words += word_count
                                total_chars += char_count
                        
                        valid_items += 1
        
        if valid_items == 0:
            print(f"  No valid items found, skipping")
            continue
        
        # Calculate average word count
        avg_words = total_words / valid_items if valid_items > 0 else 0
        avg_chars = total_chars / valid_items if valid_items > 0 else 0
        
        # Calculate average continuity scores for each ROUGE metric
        avg_rouge_scores = {}
        for rouge_metric in ['rougeLsum', 'rougeL', 'rouge2']:
            if rouge_scores[rouge_metric]:
                avg_rouge_scores[rouge_metric] = np.mean(rouge_scores[rouge_metric])
            else:
                avg_rouge_scores[rouge_metric] = 0.0
        
        # Shorten the length constraint for display
        short_constraint = shorten_length_constraint(length_constraint)
        
        # Categorize by method (concat vs iterative)
        if 'concat' in eval_dir.name:
            for rouge_metric in ['rougeLsum', 'rougeL', 'rouge2']:
                # Store: (avg_words, avg_score, constraint, avg_chars, rouge_metric, individual_scores)
                concat_data.append((avg_words, avg_rouge_scores[rouge_metric], short_constraint, avg_chars, rouge_metric, rouge_scores[rouge_metric]))
                print(f"    -> Added to concat_data ({rouge_metric}): {avg_words:.1f} words, {avg_rouge_scores[rouge_metric]:.1f} precision, constraint: {short_constraint}")
        elif 'iterative' in eval_dir.name:
            for rouge_metric in ['rougeLsum', 'rougeL', 'rouge2']:
                # Store: (avg_words, avg_score, constraint, avg_chars, rouge_metric, individual_scores)
                iterative_data.append((avg_words, avg_rouge_scores[rouge_metric], short_constraint, avg_chars, rouge_metric, rouge_scores[rouge_metric]))
                print(f"    -> Added to iterative_data ({rouge_metric}): {avg_words:.1f} words, {avg_rouge_scores[rouge_metric]:.1f} precision, constraint: {short_constraint}")
        else:
            print(f"    -> NOT CATEGORIZED: dir={eval_dir.name}")
    
    # Sort by word count
    concat_data.sort(key=lambda x: x[0])
    iterative_data.sort(key=lambda x: x[0])
    
    return concat_data, iterative_data

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
    elif "summary" in constraint_lower and "word" not in constraint_lower:
        return "unconstrained"
    else:
        # Try to extract any word count
        import re
        match = re.search(r'(\d+)\s*words?', constraint_lower)
        if match:
            return f"<{match.group(1)}"
        return constraint[:10] + "..." if len(constraint) > 10 else constraint

def create_rouge_continuity_charts(dataset="bmds", rouge_metric="rougeLsum", add_error_bars=False):
    """Create charts showing ROUGE continuity precision vs summary length."""
    
    concat_data, iterative_data = get_continuity_data(dataset)
    
    # Filter data by the specific ROUGE metric
    concat_filtered = [d for d in concat_data if d[4] == rouge_metric]
    iterative_filtered = [d for d in iterative_data if d[4] == rouge_metric]
    
    if not concat_filtered and not iterative_filtered:
        print(f"No data found for {rouge_metric} continuity in dataset {dataset}")
        return None
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Dataset title mapping
    if dataset == "bmds":
        dataset_title = "BMDS"
    elif dataset == "true-detective":
        dataset_title = "True-Detective"
    elif dataset == "detectiveqa":
        dataset_title = "DetectiveQA"
    else:
        dataset_title = dataset.upper()
    
    # ROUGE metric title mapping
    rouge_title_map = {
        'rougeLsum': 'ROUGE-Lsum',
        'rougeL': 'ROUGE-L', 
        'rouge2': 'ROUGE-2'
    }
    rouge_title = rouge_title_map.get(rouge_metric, rouge_metric.upper())
    

    
    # Aggregate data by constraint category for error bar calculation
    
    concat_by_category = defaultdict(list)
    iterative_by_category = defaultdict(list)
    
    # Group individual scores by constraint category
    for data_point in concat_filtered:
        category = data_point[2]
        individual_scores = data_point[5]  # individual_scores is the 6th element
        concat_by_category[category].extend(individual_scores)
    
    for data_point in iterative_filtered:
        category = data_point[2]
        individual_scores = data_point[5]  # individual_scores is the 6th element
        iterative_by_category[category].extend(individual_scores)
    
    # Calculate means and standard errors for each category
    concat_dict = {}
    iterative_dict = {}
    
    for category in concat_by_category:
        scores = concat_by_category[category]
        if scores:
            mean_score = np.mean(scores)
            sem_score = stats.sem(scores) if len(scores) > 1 else 0
            # Find a representative data point for word count
            representative = next(d for d in concat_filtered if d[2] == category)
            concat_dict[category] = (representative[0], mean_score, category, representative[3], representative[4], sem_score)
    
    for category in iterative_by_category:
        scores = iterative_by_category[category]
        if scores:
            mean_score = np.mean(scores)
            sem_score = stats.sem(scores) if len(scores) > 1 else 0
            # Find a representative data point for word count
            representative = next(d for d in iterative_filtered if d[2] == category)
            iterative_dict[category] = (representative[0], mean_score, category, representative[3], representative[4], sem_score)
    
    # Find common categories and sort by word length
    common_categories = set(concat_dict.keys()) & set(iterative_dict.keys())
    if not common_categories:
        print(f"No common constraint categories found between concat and iterative methods for {rouge_metric}!")
        return None
    
    # Sort categories by the concat word length (could use iterative too, should be similar)
    sorted_categories = sorted(common_categories, key=lambda cat: concat_dict[cat][0])
    
    # Prepare data for plotting
    n_pairs = len(sorted_categories)
    
    bar_width = 0.6
    x_base = np.arange(n_pairs) * 2.0
    
    # Colors
    concat_color = '#FF6B6B'  # Orange-red
    iterative_color = '#9370DB'  # Purple
    
    for i, category in enumerate(sorted_categories):
        concat_data_point = concat_dict[category]
        iterative_data_point = iterative_dict[category]
        
        # Position bars side by side
        concat_x = x_base[i] - bar_width/2
        iterative_x = x_base[i] + bar_width/2
        
        # Plot concat bar with optional error bars
        concat_yerr = concat_data_point[5] if add_error_bars else None
        concat_bar = ax.bar(concat_x, concat_data_point[1], 
                          color=concat_color, alpha=0.8, width=bar_width, 
                          yerr=concat_yerr, capsize=5 if add_error_bars else 0,
                          label='Concat' if i == 0 else "")
        
        # Plot iterative bar with optional error bars
        iterative_yerr = iterative_data_point[5] if add_error_bars else None
        iterative_bar = ax.bar(iterative_x, iterative_data_point[1], 
                             color=iterative_color, alpha=0.8, width=bar_width, 
                             yerr=iterative_yerr, capsize=5 if add_error_bars else 0,
                             label='Iterative' if i == 0 else "")
        
        # Add precision score labels above bars
        ax.text(concat_x, concat_data_point[1] + 1, f'{concat_data_point[1]:.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.text(iterative_x, iterative_data_point[1] + 1, f'{iterative_data_point[1]:.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        

        
        # Add constraint label below the pair of bars
        pair_center = x_base[i]
        ax.text(pair_center, -0.02, category, ha='center', va='top', 
                transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold')
    
    # Customize the plot
    ax.set_ylabel(f'{rouge_title} Continuity Precision', fontsize=12)
    ax.set_ylim(0, 100)  # ROUGE precision scores from 0 to 100
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis
    ax.set_xticks(x_base)
    ax.set_xticklabels([''] * len(x_base))  # Hide default x-axis labels since we have custom ones
    
    # Add legend
    ax.legend(loc='upper right')
    
    plt.tight_layout(pad=2.0)  # Add padding for constraint labels
    
    # Add x-axis label at the very bottom of the figure
    plt.figtext(0.5, 0.005, 'Summary Length Constraint', ha='center', fontsize=12)
    
    # Generate filename
    filename = f'rouge_continuity_{rouge_metric}_{dataset}.png'
    output_path = f'../plots/{filename}'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"{rouge_title} Continuity Analysis:")
    print("=" * 50)
    print(f"Showing {len(sorted_categories)} constraint categories with both concat and iterative methods:")
    
    for category in sorted_categories:
        concat_data_point = concat_dict[category]
        iterative_data_point = iterative_dict[category]
        
        print(f"\n{category} constraint:")
        print(f"  Concat:    {int(concat_data_point[0]):3d} words ({int(concat_data_point[3]):4d} chars): {concat_data_point[1]:.1f} precision")
        print(f"  Iterative: {int(iterative_data_point[0]):3d} words ({int(iterative_data_point[3]):4d} chars): {iterative_data_point[1]:.1f} precision")
    
    # Overall averages
    if sorted_categories:
        concat_avg_precision = np.mean([concat_dict[cat][1] for cat in sorted_categories])
        concat_avg_words = np.mean([concat_dict[cat][0] for cat in sorted_categories])
        iter_avg_precision = np.mean([iterative_dict[cat][1] for cat in sorted_categories])
        iter_avg_words = np.mean([iterative_dict[cat][0] for cat in sorted_categories])
        
        print(f"\nOverall Averages:")
        print(f"  Concat:    {concat_avg_precision:.1f} precision, {concat_avg_words:.0f} words")
        print(f"  Iterative: {iter_avg_precision:.1f} precision, {iter_avg_words:.0f} words")
    
    return output_path

def create_all_rouge_continuity_charts(dataset="bmds", add_error_bars=False):
    """Create charts for all ROUGE metrics."""
    rouge_metrics = ['rougeLsum', 'rougeL', 'rouge2']
    output_paths = []
    
    for rouge_metric in rouge_metrics:
        print(f"\n{'='*60}")
        print(f"Creating {rouge_metric} continuity chart for {dataset}")
        print(f"{'='*60}")
        
        output_path = create_rouge_continuity_charts(dataset, rouge_metric, add_error_bars)
        if output_path:
            output_paths.append(output_path)
    
    return output_paths

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create ROUGE continuity comparison charts")
    parser.add_argument("dataset", nargs="?", default="bmds", 
                       choices=["bmds", "true-detective", "detectiveqa"],
                       help="Dataset to analyze (default: bmds)")
    parser.add_argument("--rouge-metric", default="all",
                       choices=["rougeLsum", "rougeL", "rouge2", "all"],
                       help="ROUGE metric to plot (default: all)")
    parser.add_argument("--add-error-bars", action="store_true",
                       help="Add error bars showing standard error of the mean")
    
    args = parser.parse_args()
    
    print(f"Creating ROUGE continuity charts for dataset: {args.dataset}")
    
    print(f"Error bars: {'enabled' if args.add_error_bars else 'disabled'}")
    
    if args.rouge_metric == "all":
        output_paths = create_all_rouge_continuity_charts(args.dataset, args.add_error_bars)
        print(f"\nAll charts saved:")
        for path in output_paths:
            print(f"  {path}")
    else:
        output_path = create_rouge_continuity_charts(args.dataset, args.rouge_metric, args.add_error_bars)
        if output_path:
            print(f"\nChart saved as: {output_path}")
        else:
            print("No chart generated - insufficient data")