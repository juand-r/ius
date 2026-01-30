#!/usr/bin/env python3
"""Create visualizations for whodunit analysis."""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def analyze_directory(dir_path):
    """Analyze a single evaluation directory."""
    
    # Load collection metadata
    collection_file = dir_path / "collection.json"
    with open(collection_file, 'r') as f:
        collection = json.load(f)
    
    meta = collection['whodunit_evaluation_info']['collection_metadata']
    
    # Count items and accuracy
    items_dir = dir_path / "items"
    total_items = 0
    correct_culprits = 0
    correct_accomplices = 0
    both_correct = 0
    total_length = 0
    
    for item_file in items_dir.glob("*.json"):
        with open(item_file, 'r') as f:
            item = json.load(f)
        
        assessment = item.get('solution_correctness_assessment', {})
        if not assessment:
            continue
            
        total_items += 1
        
        # Accuracy
        culprit_correct = assessment.get('culprit', {}).get('culprit_correct') == 'Yes'
        accomplice_correct = assessment.get('accomplice', {}).get('accomplice_correct') == 'Yes'
        
        if culprit_correct:
            correct_culprits += 1
        if accomplice_correct:
            correct_accomplices += 1
        if culprit_correct and accomplice_correct:
            both_correct += 1
            
        # Text length
        length = item.get('item_metadata', {}).get('selected_text_length', 0)
        total_length += length
    
    return {
        'directory': dir_path.name,
        'input_type': meta.get('input_type'),
        'range_spec': meta.get('range_spec'),
        'source_collection': meta.get('source_collection', ''),
        'total_items': total_items,
        'culprit_accuracy': correct_culprits / total_items if total_items > 0 else 0,
        'accomplice_accuracy': correct_accomplices / total_items if total_items > 0 else 0,
        'both_accuracy': both_correct / total_items if total_items > 0 else 0,
        'avg_length': total_length / total_items if total_items > 0 else 0,
    }

def load_ground_truth_sheets_once() -> Dict[str, Dict[str, Any]]:
    """
    Load the entire Google Sheets ground truth data once and return as dictionary.
    
    Returns:
        Dictionary mapping story_id -> ground_truth_data
    """
    try:
        # Google Sheet configuration
        SHEET_ID = "1awnPbTUjIfVOqqhd8vWXQm8iwPXRMXJ4D1-MWfwLNwM"
        GID = "0"
        
        # Construct CSV export URL
        csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
        
        print("Loading ground truth data from Google Sheets (once)...")
        
        # Read the sheet (keep "N/A" as string, don't convert to NaN)
        df = pd.read_csv(csv_url, keep_default_na=False, na_values=[''])
        
        # Convert to dictionary for fast lookups
        ground_truth_dict = {}
        
        for _, row in df.iterrows():
            story_id = row.get('story_id')
            if not story_id:
                continue
                
            # Extract ground truth data
            ground_truth = {
                "culprits": row.get('Gold label (human): main culprit(s)  (PRE-REVEAL)', None),
                "culprits_post_reveal": row.get('Gold label (human): main culprit(s), POST-REVEAL INFORMATION', None),
                "accomplices": row.get('Gold label (human): accomplice(s)', None),
                "suspects": row.get('Gold suspects pre-reveal', None)
            }
            
            # Convert pandas NaN, empty strings, and "N/A" to None
            for key, value in ground_truth.items():
                if pd.isna(value) or value == "" or value == "N/A":
                    ground_truth[key] = "None"
            
            ground_truth_dict[story_id] = ground_truth
        
        print(f"✅ Loaded ground truth for {len(ground_truth_dict)} stories from Google Sheets")
        return ground_truth_dict
        
    except Exception as e:
        print(f"Failed to load ground truth from Google Sheets: {e}")
        raise

def get_ground_truth_from_dict(story_id: str, ground_truth_dict: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Get ground truth data from pre-loaded dictionary.
    
    Args:
        story_id: The story ID to look up
        ground_truth_dict: Pre-loaded ground truth dictionary
        
    Returns:
        Dictionary with ground truth data or None if not found
    """
    return ground_truth_dict.get(story_id, None)

def calculate_true_detective_baseline():
    """Calculate baseline accuracy for true-detective dataset."""
    print("True-detective baseline set to 0.25 (4 answer options)")
    return 0.25

def calculate_true_detective_human_estimate():
    """Calculate human estimate for true-detective from solve_rate and attempts data."""
    
    # Use the specific directory that has puzzle_data
    eval_dir = Path("outputs/eval/extrinsic/true-detective_fixed_size_2000_whodunit_7fdb57")
    items_dir = eval_dir / "items"
    
    if not items_dir.exists():
        raise ValueError(f"Directory not found: {items_dir}")
    
    print(f"Calculating human estimate from: {eval_dir.name}")
    
    solve_rates = []  # p_i values
    attempts = []     # M_i values
    
    # Process all items to extract solve_rate and attempts
    for item_file in items_dir.glob("*.json"):
        try:
            with open(item_file) as f:
                item_data = json.load(f)
            
            # Navigate to puzzle_data
            puzzle_data = item_data.get("item_metadata", {}).get("puzzle_data", {})
            
            if not puzzle_data:
                print(f"Warning: No puzzle_data found for {item_file.stem}")
                continue
            
            solve_rate = puzzle_data.get("solve_rate")
            attempt_count = puzzle_data.get("attempts")
            
            if solve_rate is None or attempt_count is None:
                print(f"Warning: Missing solve_rate or attempts for {item_file.stem}")
                continue
            
            # Convert solve_rate to proportion (divide by 100 since values are percentages)
            if isinstance(solve_rate, str):
                if solve_rate.endswith('%'):
                    solve_rate = float(solve_rate[:-1]) / 100.0
                else:
                    solve_rate = float(solve_rate) / 100.0
            else:
                solve_rate = solve_rate / 100.0
            
            # Convert attempts to int if it's a string
            if isinstance(attempt_count, str):
                attempt_count = int(attempt_count)
            
            solve_rates.append(solve_rate)
            attempts.append(attempt_count)
            
            print(f"  {item_file.stem}: solve_rate={solve_rate:.3f}, attempts={attempt_count}")
            
        except Exception as e:
            print(f"Error processing {item_file}: {e}")
            continue
    
    total_items = len(solve_rates)
    
    # Check total_items (expected 191 but might be different if partial evaluation)
    print(f"Found {total_items} items with valid puzzle_data")
    if total_items != 191:
        print(f"Warning: Expected 191 items, but found {total_items}. This might be a partial evaluation.")
    
    if total_items == 0:
        raise ValueError("No items found with valid puzzle_data (solve_rate and attempts)")
    
    # Calculate human estimate (unweighted mean of task proportions)
    human_estimate = sum(solve_rates) / total_items
    
    # Calculate standard error
    # SE = sqrt(1/total_items^2 * sum(p_i * (1-p_i) / M_i))
    variance_terms = []
    for p_i, M_i in zip(solve_rates, attempts):
        if M_i > 0:  # Avoid division by zero
            # p_i is already converted to proportion when extracted
            variance_terms.append(p_i * (1 - p_i) / M_i)
        else:
            print(f"Warning: Zero attempts found, skipping variance term")
    
    if not variance_terms:
        raise ValueError("No valid variance terms calculated")
    
    variance_sum = sum(variance_terms)
    standard_error = (variance_sum / (total_items ** 2)) ** 0.5
    
    print(f"Human estimate calculation:")
    print(f"  - Total items: {total_items}")
    print(f"  - Solve rates range: {min(solve_rates):.3f} - {max(solve_rates):.3f}")
    print(f"  - Average solve rate: {human_estimate:.3f}")

    
    return human_estimate, standard_error

def calculate_baseline_accuracy(dataset_filter="bmds"):
    """Calculate baseline accuracy from random choice guessing"""
    
    # For true-detective, there are only 4 suspects so no need to read in suspects list.
    if dataset_filter == "true-detective":
        return calculate_true_detective_baseline()
    
    # Load Google Sheets data once (for BMDS)
    ground_truth_dict = load_ground_truth_sheets_once()
    if not ground_truth_dict:
        raise ValueError("Failed to load ground truth data from Google Sheets")
    
    # Get all BMDS story IDs from evaluation results
    base_path = Path("outputs/eval/extrinsic")
    story_ids = set()
    
    # Collect story IDs from all BMDS evaluation directories
    for eval_dir in base_path.iterdir():
        if not eval_dir.is_dir() or 'bmds' not in eval_dir.name:
            continue
            
        items_dir = eval_dir / "items"
        if not items_dir.exists():
            continue
            
        for item_file in items_dir.glob("*.json"):
            story_id = item_file.stem
            story_ids.add(story_id)
    
    if not story_ids:
        raise ValueError("No BMDS story IDs found in evaluation results")
    
    # Get ground truth suspects for each story using dictionary lookup
    suspect_counts = []
    
    for story_id in story_ids:
        ground_truth = get_ground_truth_from_dict(story_id, ground_truth_dict)
        if ground_truth and ground_truth.get('suspects'):
            suspects_str = ground_truth['suspects']
            if suspects_str and suspects_str != 'None':
                # Count suspects by splitting on commas and cleaning
                suspects_list = [s.strip() for s in suspects_str.split(',') if s.strip()]
                suspect_counts.append(len(suspects_list))
                print(f"  {story_id}: {len(suspects_list)} suspects - {suspects_str}")
    
    if not suspect_counts:
        raise ValueError("No valid suspect counts found from Google Sheets")
    
    # Calculate average of 1/num_suspects for each story
    baseline_accuracies = [1.0 / count for count in suspect_counts]
    average_baseline = np.mean(baseline_accuracies)
    
    print(f"Baseline calculation from Google Sheets ground truth:")
    print(f"  - Total stories analyzed: {len(suspect_counts)}")
    print(f"  - Suspect counts range: {min(suspect_counts)}-{max(suspect_counts)}")
    print(f"  - Average suspects per story: {np.mean(suspect_counts):.1f}")
    print(f"  - Baseline accuracy (avg of 1/num_suspects): {average_baseline:.3f}")
    
    return average_baseline

def load_evaluation_results():
    """Load and analyze all evaluation results."""
    base_path = Path("outputs/eval/extrinsic")
    results = []
    
    for eval_dir in base_path.iterdir():
        if eval_dir.is_dir():
            try:
                result = analyze_directory(eval_dir)
                results.append(result)
            except Exception as e:
                print(f"Error processing {eval_dir.name}: {e}")
    
    return results

def create_charts(dataset_filter="bmds"):
    """Create charts for the whodunit analysis for specified dataset."""
    
    # Load evaluation results
    all_results = load_evaluation_results()
    
    # Filter results by dataset
    if dataset_filter == "bmds":
        results = [r for r in all_results if 'bmds' in r['directory'].lower()]
    elif dataset_filter == "true-detective":
        results = [r for r in all_results if 'true-detective' in r['directory'].lower()]
    else:
        results = all_results  # No filtering for "all"
    
    print(f"Filtered to {len(results)} {dataset_filter} evaluation results")
    
    # Calculate baseline accuracy for specified dataset
    baseline_accuracy = calculate_baseline_accuracy(dataset_filter)
    
    # Calculate human estimate for true-detective
    human_estimate = None
    human_se = None
    if dataset_filter == "true-detective":
        human_estimate, human_se = calculate_true_detective_human_estimate()
    
    # Calculate dynamic values from actual data
    # Chunks (all)
    chunks_all = [r for r in results if r['input_type'] == 'chunks' and r['range_spec'] == 'all']
    chunks_all_acc = sum(r['culprit_accuracy'] * r['total_items'] for r in chunks_all) / sum(r['total_items'] for r in chunks_all) if chunks_all else 0
    chunks_all_len = sum(r['avg_length'] * r['total_items'] for r in chunks_all) / sum(r['total_items'] for r in chunks_all) if chunks_all else 0
    
    # Chunks (pre-reveal)
    chunks_pre = [r for r in results if r['input_type'] == 'chunks' and r['range_spec'] in ['penultimate', 'all-but-last']]
    chunks_pre_acc = sum(r['culprit_accuracy'] * r['total_items'] for r in chunks_pre) / sum(r['total_items'] for r in chunks_pre) if chunks_pre else 0
    chunks_pre_len = sum(r['avg_length'] * r['total_items'] for r in chunks_pre) / sum(r['total_items'] for r in chunks_pre) if chunks_pre else 0
    
    # Chunks (first chunk only)
    chunks_first = [r for r in results if r['input_type'] == 'chunks' and r['range_spec'] == '1']
    chunks_first_acc = sum(r['culprit_accuracy'] * r['total_items'] for r in chunks_first) / sum(r['total_items'] for r in chunks_first) if chunks_first else 0
    chunks_first_len = sum(r['avg_length'] * r['total_items'] for r in chunks_first) / sum(r['total_items'] for r in chunks_first) if chunks_first else 0
    
    # Summaries (pre-reveal) - concat
    summaries_pre = [r for r in results if r['input_type'] == 'summaries' and r['range_spec'] in ['penultimate', 'all-but-last']]
    concat_pre = [r for r in summaries_pre if 'concat' in r['directory']]
    concat_pre_acc = sum(r['culprit_accuracy'] * r['total_items'] for r in concat_pre) / sum(r['total_items'] for r in concat_pre) if concat_pre else 0
    concat_pre_len = sum(r['avg_length'] * r['total_items'] for r in concat_pre) / sum(r['total_items'] for r in concat_pre) if concat_pre else 0
    
    # Summaries (pre-reveal) - iterative
    iterative_pre = [r for r in summaries_pre if 'iterative' in r['directory']]
    iterative_pre_acc = sum(r['culprit_accuracy'] * r['total_items'] for r in iterative_pre) / sum(r['total_items'] for r in iterative_pre) if iterative_pre else 0
    iterative_pre_len = sum(r['avg_length'] * r['total_items'] for r in iterative_pre) / sum(r['total_items'] for r in iterative_pre) if iterative_pre else 0
    
    # Print calculated values for debugging
    print(f"Calculated values:")
    print(f"  Chunks (all): {chunks_all_acc:.3f} accuracy, {chunks_all_len:.0f} avg chars")
    print(f"  Chunks (pre-reveal): {chunks_pre_acc:.3f} accuracy, {chunks_pre_len:.0f} avg chars")
    print(f"  Chunks (first only): {chunks_first_acc:.3f} accuracy, {chunks_first_len:.0f} avg chars")
    print(f"  Concat (pre-reveal): {concat_pre_acc:.3f} accuracy, {concat_pre_len:.0f} avg chars")
    print(f"  Iterative (pre-reveal): {iterative_pre_acc:.3f} accuracy, {iterative_pre_len:.0f} avg chars")
    
    # Data from the analysis
    conditions = ['Chunks\n(all)', 'Chunks\n(pre-reveal)', 'Concat\n(pre-reveal)', 'Iterative\n(pre-reveal)']
    accuracies = [chunks_all_acc, chunks_pre_acc, concat_pre_acc, iterative_pre_acc]
    lengths = [chunks_all_len, chunks_pre_len, concat_pre_len, iterative_pre_len]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chart 1: Accuracy comparison
    colors = ['#2E8B57', '#4682B4', '#CD853F', '#9370DB']
    bars1 = ax1.bar(conditions, accuracies, color=colors, alpha=0.8)
    ax1.set_ylabel('Culprit Accuracy')
    ax1.set_title(f'Whodunit Evaluation Accuracy by Condition ({dataset_filter.upper()})')
    ax1.set_ylim(0, 1.0)
    
    # Add baseline lines to bar chart
    ax1.axhline(y=baseline_accuracy, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Random Baseline: {baseline_accuracy:.3f}')
    
    # Add human estimate line
    if dataset_filter == "bmds":
        ax1.axhline(y=0.853, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Human Estimate: 85.3%')
    elif dataset_filter == "true-detective" and human_estimate is not None:
                ax1.axhline(y=human_estimate, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                    label=f'Human Estimate: {human_estimate:.1%}')
    
    ax1.axhline(y=chunks_first_acc, color='#FF6347', linestyle=':', alpha=0.7, linewidth=2, label='Only read first chunk')
    ax1.legend()
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Chart 2: Length vs Accuracy scatter
    ax2.scatter([chunks_all_len, chunks_pre_len, chunks_first_len], [chunks_all_acc, chunks_pre_acc, chunks_first_acc], color='#2E8B57', s=100, label='Chunks', alpha=0.8)
    ax2.scatter([concat_pre_len, iterative_pre_len], [concat_pre_acc, iterative_pre_acc], color='#CD853F', s=100, label='Summaries', alpha=0.8)
    
    # Add baseline lines to scatter plot
    ax2.axhline(y=baseline_accuracy, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Random Baseline: {baseline_accuracy:.3f}')
    
    # Add human estimate line
    if dataset_filter == "bmds":
        ax2.axhline(y=0.853, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Human Estimate: 85.3%')
    elif dataset_filter == "true-detective" and human_estimate is not None:
                ax2.axhline(y=human_estimate, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                    label=f'Human Estimate: {human_estimate:.1%}')
    
    ax2.axhline(y=chunks_first_acc, color='#FF6347', linestyle=':', alpha=0.7, linewidth=2, label='Only read first chunk')
    
    ax2.set_xlabel('Average Text Length (characters)')
    ax2.set_ylabel('Culprit Accuracy')
    ax2.set_title(f'Accuracy vs Text Length ({dataset_filter.upper()})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    ax2.annotate(f'Full chunks\n({chunks_all_acc:.1%})', (chunks_all_len, chunks_all_acc), xytext=(chunks_all_len-4000, chunks_all_acc+0.05),
                arrowprops=dict(arrowstyle='->', color='gray'), ha='center')
    ax2.annotate(f'Pre-reveal chunks\n({chunks_pre_acc:.1%})', (chunks_pre_len, chunks_pre_acc), xytext=(chunks_pre_len-4000, chunks_pre_acc-0.05),
                arrowprops=dict(arrowstyle='->', color='gray'), ha='center')
    ax2.annotate(f'First chunk only\n({chunks_first_acc:.1%})', (chunks_first_len, chunks_first_acc), xytext=(chunks_first_len+2000, chunks_first_acc+0.05),
                arrowprops=dict(arrowstyle='->', color='gray'), ha='center')
    
    plt.tight_layout()
    chart_filename = f'whodunit_evaluation_charts_{dataset_filter}.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a second figure for error analysis
    fig2, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Error types for "all" chunks (from the analysis)
    error_types = ['Different\nSuspect', 'Confused w/\nAccomplice', 'Included\nOthers', 'Only\nAlias']
    error_rates = [0.56, 0.38, 0.31, 0.06]  # Rates among incorrect cases
    
    bars3 = ax3.bar(error_types, error_rates, color='#DC143C', alpha=0.7)
    ax3.set_ylabel('Rate Among Incorrect Cases')
    ax3.set_title(f'Common Error Types in Full Chunk Evaluations ({dataset_filter.upper()})')
    ax3.set_ylim(0, 0.6)
    
    # Add value labels
    for bar, rate in zip(bars3, error_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.0%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    error_chart_filename = f'whodunit_error_analysis_{dataset_filter}.png'
    plt.savefig(error_chart_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return chart_filename, error_chart_filename

if __name__ == "__main__":
    import sys
    
    # Parse command line argument for dataset
    dataset = "bmds"  # Default
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
        if dataset not in ["bmds", "true-detective"]:
            print(f"Error: Unknown dataset '{dataset}'. Use 'bmds' or 'true-detective'")
            sys.exit(1)
    
    print(f"Creating charts for dataset: {dataset}")
    chart_file, error_file = create_charts(dataset)
    print("Charts saved as:")
    if dataset == "bmds":
        print(f"  - {chart_file} (includes random baseline + human estimate at 85.3%)")
    elif dataset == "true-detective":
        print(f"  - {chart_file} (includes random baseline + human estimate from solve rates)")
    else:
        print(f"  - {chart_file} (includes random baseline)")
    print(f"  - {error_file}")