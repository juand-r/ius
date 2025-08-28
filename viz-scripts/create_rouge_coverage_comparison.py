#!/usr/bin/env python3
"""Create side-by-side comparison charts for ROUGE score analysis."""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import re
import glob

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
        return "summary"
    else:
        # Try to extract any word count
        match = re.search(r'(\d+)\s*words?', constraint_lower)
        if match:
            return f"<{match.group(1)}"
        return constraint[:10] + "..." if len(constraint) > 10 else constraint

def analyze_rouge_directory(dir_path: Path, use_last_only: bool = True) -> Dict:
    """Analyze a single ROUGE evaluation directory."""
    
    # Load collection metadata
    collection_file = dir_path / "collection.json"
    if not collection_file.exists():
        return None
    
    try:
        with open(collection_file, 'r') as f:
            collection_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None
    
    # Extract method and constraint info from directory name
    dir_name = dir_path.name
    method = 'concat' if 'concat' in dir_name else 'iterative' if 'iterative' in dir_name else 'unknown'
    
    # Get constraint from source collection
    constraint = 'unknown'
    rouge_info = collection_data.get('rouge_multi_evaluation_info', {})
    collection_metadata = rouge_info.get('collection_metadata', {})
    source_collection = collection_metadata.get('source_collection', '')
    
    if source_collection:
        # Load the source collection to get the constraint (adjust path to be relative to project root)
        source_collection_file = Path("..") / source_collection / "collection.json"
        if source_collection_file.exists():
            try:
                with open(source_collection_file, 'r') as f:
                    source_data = json.load(f)
                
                # Extract constraint from source collection
                summarization_info = source_data.get('summarization_info', {})
                source_metadata = summarization_info.get('collection_metadata', {})
                constraint = source_metadata.get('optional_summary_length', 'unknown')
            except (json.JSONDecodeError, FileNotFoundError):
                pass
    
    short_constraint = shorten_length_constraint(constraint)
    
    # Find all story directories in the items subdirectory
    items_dir = dir_path / "items"
    if not items_dir.exists():
        return None
    
    story_dirs = [d for d in items_dir.iterdir() if d.is_dir()]
    
    if not story_dirs:
        return None
    
    # Initialize metrics storage
    rouge_metrics = {
        'rouge-1_precision': [],
        'rouge-1_recall': [],
        'rouge-2_precision': [],
        'rouge-2_recall': [],
        'rouge-l_precision': [],
        'rouge-l_recall': [],
        'rs-rouge2_precision': [],
        'rs-rouge2_recall': [],
        'rs-rougeLsum_precision': [],
        'rs-rougeLsum_recall': []
    }
    
    word_counts = []
    
    for story_dir in story_dirs:
        # Find the highest numbered JSON file (final chunk)
        json_files = list(story_dir.glob("*.json"))
        if not json_files:
            continue
        
        # Sort by number in filename to get the last one
        def extract_number(filepath):
            match = re.search(r'(\d+)\.json$', filepath.name)
            return int(match.group(1)) if match else 0
        
        json_files.sort(key=extract_number)
        final_file = json_files[-1]  # Get the highest numbered file
        
        try:
            with open(final_file, 'r') as f:
                story_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        
        # Extract ROUGE scores from the final chunk
        rouge_score = story_data.get('rouge_score', {})
        
        # Store each metric if available (convert structure)
        if 'rouge-1' in rouge_score:
            rouge_metrics['rouge-1_precision'].append(rouge_score['rouge-1']['precision'])
            rouge_metrics['rouge-1_recall'].append(rouge_score['rouge-1']['recall'])
        
        if 'rouge-2' in rouge_score:
            rouge_metrics['rouge-2_precision'].append(rouge_score['rouge-2']['precision'])
            rouge_metrics['rouge-2_recall'].append(rouge_score['rouge-2']['recall'])
        
        if 'rouge-l' in rouge_score:
            rouge_metrics['rouge-l_precision'].append(rouge_score['rouge-l']['precision'])
            rouge_metrics['rouge-l_recall'].append(rouge_score['rouge-l']['recall'])
        
        if 'rs-rouge2' in rouge_score:
            rouge_metrics['rs-rouge2_precision'].append(rouge_score['rs-rouge2']['precision'])
            rouge_metrics['rs-rouge2_recall'].append(rouge_score['rs-rouge2']['recall'])
        
        if 'rs-rougeLsum' in rouge_score:
            rouge_metrics['rs-rougeLsum_precision'].append(rouge_score['rs-rougeLsum']['precision'])
            rouge_metrics['rs-rougeLsum_recall'].append(rouge_score['rs-rougeLsum']['recall'])
        
        # Extract word count from summary
        summary = story_data.get('summary_text', '')
        if summary:
            word_count = len(summary.split())
            word_counts.append(word_count)
    
    # Calculate statistics for each metric
    result = {
        'method': method,
        'constraint': constraint,
        'short_constraint': short_constraint,
        'avg_word_count': np.mean(word_counts) if word_counts else 0,
        'num_stories': len(story_dirs)
    }
    
    for metric_key, values in rouge_metrics.items():
        if values:
            result[metric_key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'sem': np.std(values) / np.sqrt(len(values)),
                'count': len(values)
            }
        else:
            result[metric_key] = {
                'mean': 0,
                'std': 0,
                'sem': 0,
                'count': 0
            }
    

    
    return result

def load_rouge_results(dataset_filter="bmds", use_last_only=True):
    """Load ROUGE evaluation results from all directories."""
    
    # Find all ROUGE evaluation directories
    outputs_dir = Path("../outputs/eval/intrinsic/rouge")
    if not outputs_dir.exists():
        outputs_dir = Path("outputs/eval/intrinsic/rouge")
    
    if not outputs_dir.exists():
        print(f"ROUGE outputs directory not found: {outputs_dir}")
        return []
    
    results = []
    
    # Find directories matching the dataset filter
    for eval_dir in outputs_dir.iterdir():
        if not eval_dir.is_dir():
            continue
        
        # Check if directory name contains the dataset filter
        if dataset_filter.lower() not in eval_dir.name.lower():
            continue
        
        # Skip if not a ROUGE evaluation directory
        if 'rouge' not in eval_dir.name.lower():
            continue
        
        print(f"Analyzing ROUGE directory: {eval_dir.name}")
        result = analyze_rouge_directory(eval_dir, use_last_only)
        
        if result and result['num_stories'] > 0:
            results.append(result)
    
    return results

def create_rouge_comparison_chart(dataset_filter="bmds", use_last_only=True, add_error_bars=False):
    """Create side-by-side comparison charts for ROUGE analysis."""
    
    # Load evaluation results
    results = load_rouge_results(dataset_filter, use_last_only)
    
    if not results:
        print(f"No {dataset_filter} ROUGE results found!")
        return []
    
    print(f"Found {len(results)} {dataset_filter} ROUGE evaluation results")
    

    
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
    elif dataset_filter.lower() == 'true-detective':
        # For true-detective: summary goes after <200
        expected_constraints = ['<50', '<100', '<200', 'summary', '<500', '<1000', 'long']
    else:
        # Default order
        expected_constraints = ['<50', '<100', '<200', '<500', 'summary', '<1000', 'long']
    
    # Define ROUGE metrics to plot
    rouge_metrics = [
        ('rouge-1_precision', 'ROUGE-1 Precision'),
        ('rouge-1_recall', 'ROUGE-1 Recall'),
        ('rouge-2_precision', 'ROUGE-2 Precision'),
        ('rouge-2_recall', 'ROUGE-2 Recall'),
        ('rouge-l_precision', 'ROUGE-L Precision'),
        ('rouge-l_recall', 'ROUGE-L Recall'),
        ('rs-rouge2_precision', 'RS-ROUGE2 Precision'),
        ('rs-rouge2_recall', 'RS-ROUGE2 Recall'),
        ('rs-rougeLsum_precision', 'RS-ROUGELsum Precision'),
        ('rs-rougeLsum_recall', 'RS-ROUGELsum Recall')
    ]
    
    # Colors for different methods
    concat_color = '#FF7F50'  # Orange
    iterative_color = '#9370DB'  # Purple
    
    chart_files = []
    
    # Create separate chart for each ROUGE metric
    for metric_key, metric_title in rouge_metrics:
        # Create individual figure for this metric
        fig, ax = plt.subplots(figsize=(10, 6))
        
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
            if concat_result and metric_key in concat_result:
                concat_means.append(concat_result[metric_key]['mean'])
                concat_sems.append(concat_result[metric_key]['sem'])
                concat_word_count = concat_result['avg_word_count']
            else:
                concat_means.append(0)
                concat_sems.append(0)
                concat_word_count = None
            
            # Find iterative result for this constraint
            iterative_result = next((r for r in iterative_results if r['short_constraint'] == constraint), None)
            if iterative_result and metric_key in iterative_result:
                iterative_means.append(iterative_result[metric_key]['mean'])
                iterative_sems.append(iterative_result[metric_key]['sem'])
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
        
        # Add value labels on bars (without % sign)
        for j, (concat_mean, iter_mean) in enumerate(zip(concat_means, iterative_means)):
            if concat_mean > 0:
                ax.text(j - width/2, concat_mean + 1, f'{concat_mean:.1f}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
            if iter_mean > 0:
                ax.text(j + width/2, iter_mean + 1, f'{iter_mean:.1f}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Customize the plot
        ax.set_xlabel('Summary Length Constraint', fontsize=12)
        ax.set_ylabel(f'{metric_title} (%)', fontsize=12)
        ax.set_ylim(0, 100.0)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Only show legend if we have labeled bars
        if concat_labeled or iterative_labeled:
            ax.legend()
        
        # Save the chart
        safe_metric_name = metric_key.replace('-', '_').replace('_', '_')
        filename = f"rouge_{safe_metric_name}_{dataset_filter}.png"
        filepath = Path("../plots") / filename
        filepath.parent.mkdir(exist_ok=True)
        plt.savefig(filepath, dpi=300)
        chart_files.append(str(filepath))
        print(f"Saved: {filename}")
        
        plt.close()
    
    # Print comparative analysis
    print("\nComparative Analysis:")
    print("=" * 50)
    
    for constraint in expected_constraints:
        concat_result = next((r for r in concat_results if r['short_constraint'] == constraint), None)
        iterative_result = next((r for r in iterative_results if r['short_constraint'] == constraint), None)
        
        if concat_result and iterative_result:
            print(f"{constraint} constraint:")
            
            # Show word counts
            concat_words = int(concat_result['avg_word_count'])
            iterative_words = int(iterative_result['avg_word_count'])
            print(f"  Concat    ({concat_words}w): ", end="")
            
            # Show a few key metrics
            r1_p = concat_result.get('rouge-1_precision', {}).get('mean', 0)
            r1_r = concat_result.get('rouge-1_recall', {}).get('mean', 0)
            r2_p = concat_result.get('rouge-2_precision', {}).get('mean', 0)
            print(f"R1-P:{r1_p:.1f}%, R1-R:{r1_r:.1f}%, R2-P:{r2_p:.1f}%")
            
            print(f"  Iterative ({iterative_words}w): ", end="")
            r1_p_iter = iterative_result.get('rouge-1_precision', {}).get('mean', 0)
            r1_r_iter = iterative_result.get('rouge-1_recall', {}).get('mean', 0)
            r2_p_iter = iterative_result.get('rouge-2_precision', {}).get('mean', 0)
            print(f"R1-P:{r1_p_iter:.1f}%, R1-R:{r1_r_iter:.1f}%, R2-P:{r2_p_iter:.1f}%")
            
            # Show differences
            r1_p_diff = r1_p_iter - r1_p
            r1_r_diff = r1_r_iter - r1_r
            r2_p_diff = r2_p_iter - r2_p
            print(f"  Difference (Iterative - Concat): R1-P:{r1_p_diff:+.1f}%, R1-R:{r1_r_diff:+.1f}%, R2-P:{r2_p_diff:+.1f}%")
    
    return chart_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create ROUGE comparison charts")
    parser.add_argument("dataset", nargs="?", default="bmds", 
                       help="Dataset to analyze (bmds, true-detective, or detectiveqa)")
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
    
    print(f"Creating ROUGE comparison charts for dataset: {dataset}")
    print(f"Using {'last summary only' if use_last_only else 'all summaries'} for each item")
    print(f"Error bars: {'enabled' if args.add_error_bars else 'disabled'}")
    
    chart_files = create_rouge_comparison_chart(dataset, use_last_only, args.add_error_bars)
    if chart_files:
        print(f"\nComparison charts saved as:")
        for chart_file in chart_files:
            print(f"  - {chart_file}")
    else:
        print("No charts were created.")