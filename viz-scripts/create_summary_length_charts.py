#!/usr/bin/env python3
"""Create charts showing accuracy vs summary length for concat and iterative methods."""

import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import pandas as pd
from pathlib import Path

def calculate_true_detective_human_estimate():
    """Calculate human estimate for true-detective from solve_rate and attempts data."""
    
    # Use the specific directory that has puzzle_data
    eval_dir = Path("../outputs/eval/extrinsic/true-detective_fixed_size_2000_whodunit_7fdb57")
    items_dir = eval_dir / "items"
    
    if not items_dir.exists():
        print(f"Warning: Directory not found: {items_dir}")
        return None, None
    
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
                continue
            
            solve_rate = puzzle_data.get("solve_rate")
            attempt_count = puzzle_data.get("attempts")
            
            if solve_rate is None or attempt_count is None:
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
            
        except Exception as e:
            continue
    
    total_items = len(solve_rates)
    
    if total_items == 0:
        print("Warning: No items found with valid puzzle_data (solve_rate and attempts)")
        return None, None
    
    # Calculate human estimate (unweighted mean of task proportions)
    human_estimate = sum(solve_rates) / total_items
    
    # Calculate standard error
    # SE = sqrt(1/total_items^2 * sum(p_i * (1-p_i) / M_i))
    variance_terms = []
    for p_i, M_i in zip(solve_rates, attempts):
        if M_i > 0:  # Avoid division by zero
            variance_terms.append(p_i * (1 - p_i) / M_i)
    
    if not variance_terms:
        return human_estimate, None
    
    variance_sum = sum(variance_terms)
    standard_error = (variance_sum / (total_items ** 2)) ** 0.5
    
    print(f"Human estimate: {human_estimate:.1%} from {total_items} items")
    
    return human_estimate, standard_error

def calculate_true_detective_baseline():
    """Calculate baseline accuracy for true-detective dataset."""
    return 0.25  # 25% - 4 answer options

def load_ground_truth_sheets_once():
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
        
        print("Loading ground truth data from Google Sheets...")
        
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
        return None

def calculate_bmds_baseline():
    """Calculate baseline accuracy for bmds dataset from random choice guessing"""
    
    # Load Google Sheets data once
    ground_truth_dict = load_ground_truth_sheets_once()
    if not ground_truth_dict:
        print("Warning: Failed to load ground truth data, using fallback baseline")
        return 0.167  # Fallback estimate
    
    # Get all BMDS story IDs from evaluation results
    base_path = Path("../outputs/eval/extrinsic")
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
        print("Warning: No BMDS story IDs found, using fallback baseline")
        return 0.167
    
    # Get ground truth suspects for each story using dictionary lookup
    suspect_counts = []
    
    for story_id in story_ids:
        ground_truth = ground_truth_dict.get(story_id)
        if ground_truth and ground_truth.get('suspects'):
            suspects_str = ground_truth['suspects']
            if suspects_str and suspects_str != 'None':
                # Count suspects by splitting on commas and cleaning
                suspects_list = [s.strip() for s in suspects_str.split(',') if s.strip()]
                suspect_counts.append(len(suspects_list))
    
    if not suspect_counts:
        print("Warning: No valid suspect counts found, using fallback baseline")
        return 0.167
    
    # Calculate average of 1/num_suspects for each story
    baseline_accuracies = [1.0 / count for count in suspect_counts]
    average_baseline = np.mean(baseline_accuracies)
    
    print(f"BMDS random baseline: {average_baseline:.3f} from {len(suspect_counts)} stories")
    
    return average_baseline

def get_summary_data(dataset_filter="bmds", include_error_bars=False):
    """Extract summary data from the evaluation results and their source collections."""
    
    base_path = Path("../outputs/eval/extrinsic")
    concat_data = []
    iterative_data = []
    first_chunk_data = []
    chunks_all_data = []
    chunks_pre_data = []  # For range_spec "1"
    
    # Scan for directories matching the dataset filter
    all_eval_dirs = []
    
    for eval_dir in base_path.iterdir():
        if not eval_dir.is_dir():
            continue
        if not eval_dir.name.startswith(dataset_filter):
            continue
        if "whodunit" not in eval_dir.name:
            continue
        all_eval_dirs.append(eval_dir)
    
    if not all_eval_dirs:
        print(f"No {dataset_filter} whodunit evaluation directories found!")
        return [], [], []
    
    print(f"Found {len(all_eval_dirs)} {dataset_filter} whodunit evaluation directories")
    
    for eval_dir in all_eval_dirs:
            
        # Load evaluation collection metadata
        collection_file = eval_dir / "collection.json"
        if not collection_file.exists():
            continue
            
        with open(collection_file, 'r') as f:
            eval_collection = json.load(f)
        
        eval_meta = eval_collection.get('whodunit_evaluation_info', {}).get('collection_metadata', {})
        source_collection = eval_meta.get('source_collection', '')
        range_spec = eval_meta.get('range_spec', '')
        input_type = eval_meta.get('input_type', '')
        
        if not source_collection or not range_spec or not input_type:
            continue
            
        # Skip if not one of the range specs we're interested in
        if range_spec not in ['penultimate', 'last', 'all', 'all-but-last', '1']:
            continue
            
        # Skip if not summaries or chunks
        if input_type not in ['summaries', 'chunks']:
            continue
            
        # For summaries, load source collection metadata to get length constraint
        # For chunks, we don't have length constraints
        length_constraint = ''
        if input_type == 'summaries':
            source_collection_file = Path("..") / source_collection / "collection.json"
            if not source_collection_file.exists():
            continue
            
        with open(source_collection_file, 'r') as f:
            source_data = json.load(f)
        
        source_meta = source_data.get('summarization_info', {}).get('collection_metadata', {})
        length_constraint = source_meta.get('optional_summary_length', '')
        
        # Calculate accuracy from evaluation results
        items_dir = eval_dir / "items"
        if not items_dir.exists():
            continue
            
        total_items = 0
        correct_items = 0
        total_words = 0
        total_chars = 0
        
        for item_file in items_dir.glob("*.json"):
            with open(item_file, 'r') as f:
                item_data = json.load(f)
            
            assessment = item_data.get('solution_correctness_assessment', {})
            if not assessment:
                continue
                
            total_items += 1
            if assessment.get('culprit', {}).get('culprit_correct') == 'Yes':
                correct_items += 1
                
            # Handle text length differently for summaries vs chunks
            if input_type == 'summaries':
                # Get summary text from source collection
                item_id = item_file.stem
                source_items_dir = Path("..") / source_collection / "items"
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
                            
                            if range_spec == 'penultimate':
                                if len(summaries) >= 2:
                                    summary_texts = [summaries[-2]]
                            elif range_spec == 'last':
                                summary_texts = [summaries[-1]]
                            elif range_spec == 'all':
                                summary_texts = summaries
                            elif range_spec == 'all-but-last':
                                if len(summaries) >= 2:
                                    summary_texts = summaries[:-1]
                            elif range_spec == '1':
                                if len(summaries) >= 1:
                                    summary_texts = [summaries[0]]
                            
                            # Count words and chars for the selected summaries
                            for summary_text in summary_texts:
                                if summary_text:
                                    word_count = len(summary_text.split())
                                    char_count = len(summary_text)
                                    total_words += word_count
                                    total_chars += char_count
            
            elif input_type == 'chunks':
                # For chunks, get text length from evaluation metadata
                selected_text_length = item_data.get('item_metadata', {}).get('selected_text_length', 0)
                if selected_text_length > 0:
                    # Estimate word count from character count (rough estimate)
                    estimated_words = selected_text_length / 5  # Rough estimate: 5 chars per word
                    total_words += estimated_words
                    total_chars += selected_text_length
        
        if total_items == 0:
            continue
            
        accuracy = correct_items / total_items
        
        # Calculate error bars if requested
        if include_error_bars:
            # Bootstrap confidence interval for proportion
            # Create binary array: 1 for correct, 0 for incorrect
            successes = np.array([1] * correct_items + [0] * (total_items - correct_items))
            
            # Bootstrap resampling
            n_bootstrap = 1000
            bootstrap_means = []
            np.random.seed(42)  # For reproducibility
            
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(successes, size=total_items, replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            # Calculate 95% confidence interval
            ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
            bootstrap_error = ci_upper - accuracy  # Use upper bound distance as error bar
        else:
            bootstrap_error = 0.0
        
        words = total_words / total_items  # Actual average word count
        avg_chars = total_chars / total_items  # Actual average character count
        
        # Handle categorization differently for summaries vs chunks
        if input_type == 'summaries':
        # Shorten the length constraint as requested
        short_constraint = shorten_length_constraint(length_constraint)
        
            # Categorize by range_spec and method
            if range_spec == '1':
                first_chunk_data.append((words, accuracy, short_constraint, avg_chars, bootstrap_error))
                print(f"    -> Added to first_chunk_data: {words} words, {accuracy:.1%} accuracy, constraint: {short_constraint}")
            elif 'concat' in eval_dir.name:
                concat_data.append((words, accuracy, short_constraint, avg_chars, bootstrap_error))
                print(f"    -> Added to concat_data: {words} words, {accuracy:.1%} accuracy, constraint: {short_constraint}")
        elif 'iterative' in eval_dir.name:
                iterative_data.append((words, accuracy, short_constraint, avg_chars, bootstrap_error))
                print(f"    -> Added to iterative_data: {words} words, {accuracy:.1%} accuracy, constraint: {short_constraint}")
            else:
                print(f"    -> NOT CATEGORIZED: range_spec={range_spec}, dir={eval_dir.name}")
        
        elif input_type == 'chunks':
            # Categorize chunks by their range_spec - no constraint categorization needed
            if range_spec == 'all':
                chunks_all_data.append((words, accuracy, 'chunks-all', avg_chars, bootstrap_error))
                print(f"    -> Added to chunks_all_data: {words:.1f} words, {accuracy:.1%} accuracy")
            elif range_spec in ['penultimate', 'all-but-last']:
                chunks_pre_data.append((words, accuracy, 'chunks-pre', avg_chars, bootstrap_error))
                print(f"    -> Added to chunks_pre_data: {words:.1f} words, {accuracy:.1%} accuracy")
            elif range_spec == '1':
                first_chunk_data.append((words, accuracy, 'chunks-first', avg_chars, bootstrap_error))
                print(f"    -> Added to first_chunk_data (chunks): {words:.1f} words, {accuracy:.1%} accuracy")
    
    # Sort by word count
    concat_data.sort(key=lambda x: x[0])
    iterative_data.sort(key=lambda x: x[0])
    first_chunk_data.sort(key=lambda x: x[0])
    chunks_all_data.sort(key=lambda x: x[0])
    chunks_pre_data.sort(key=lambda x: x[0])
    
    return concat_data, iterative_data, first_chunk_data, chunks_all_data, chunks_pre_data

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

def create_summary_length_charts(dataset="bmds", include_error_bars=False):
    """Create charts showing accuracy vs summary length."""
    
    concat_data, iterative_data, first_chunk_data, chunks_all_data, chunks_pre_data = get_summary_data(dataset, include_error_bars)
    
    # Calculate baselines
    human_estimate = None
    random_baseline = None
    
    if dataset == "true-detective":
        human_estimate, _ = calculate_true_detective_human_estimate()
        random_baseline = calculate_true_detective_baseline()
    elif dataset == "bmds":
        human_estimate = 0.853  # 85.3% human estimate for BMDS
        random_baseline = calculate_bmds_baseline()
    elif dataset == "detectiveqa":
        # No random baseline or human estimate available for detectiveqa yet
        random_baseline = None
        human_estimate = None
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    if dataset == "bmds":
        dataset_title = "BMDS"
    elif dataset == "true-detective":
        dataset_title = "True-Detective"
    elif dataset == "detectiveqa":
        dataset_title = "DetectiveQA"
    else:
        dataset_title = dataset.upper()
    
    # Set title based on whether error bars are included
    if include_error_bars:
        fig.suptitle(f'{dataset_title}: Summary Method Accuracy vs Length (Bootstrap 95% CI)', fontsize=16, fontweight='bold')
    else:
        fig.suptitle(f'{dataset_title}: Summary Method Accuracy vs Length', fontsize=16, fontweight='bold')
    
    # Match concat and iterative data by constraint category
    concat_dict = {d[2]: d for d in concat_data}  # category -> data tuple
    iterative_dict = {d[2]: d for d in iterative_data}  # category -> data tuple
    
    # Find common categories and sort by word length
    common_categories = set(concat_dict.keys()) & set(iterative_dict.keys())
    if not common_categories:
        print("No common constraint categories found between concat and iterative methods!")
        return None
    
    # Sort categories by the concat word length (could use iterative too, should be similar)
    sorted_categories = sorted(common_categories, key=lambda cat: concat_dict[cat][0])
    
    # Prepare data for plotting
    n_pairs = len(sorted_categories)
    x_positions = []
    concat_bars = []
    iterative_bars = []
    
    bar_width = 0.6  # Make individual bars wider
    x_base = np.arange(n_pairs) * 2.0  # Back to original spacing
    
    # Colors
    concat_color = '#FF6B6B'  # Orange-red
    iterative_color = '#9370DB'  # Purple
    
    for i, category in enumerate(sorted_categories):
        concat_data_point = concat_dict[category]
        iterative_data_point = iterative_dict[category]
        
        # Position bars side by side
        concat_x = x_base[i] - bar_width/2
        iterative_x = x_base[i] + bar_width/2
        
        # Plot concat bar
        if include_error_bars:
            concat_bar = ax.bar(concat_x, concat_data_point[1], yerr=concat_data_point[4], 
                              color=concat_color, alpha=0.8, width=bar_width, capsize=5, label='Concat' if i == 0 else "")
        else:
            concat_bar = ax.bar(concat_x, concat_data_point[1], 
                              color=concat_color, alpha=0.8, width=bar_width, label='Concat' if i == 0 else "")
        concat_bars.append((concat_bar, concat_data_point))
        
        # Plot iterative bar
        if include_error_bars:
            iterative_bar = ax.bar(iterative_x, iterative_data_point[1], yerr=iterative_data_point[4], 
                                 color=iterative_color, alpha=0.8, width=bar_width, capsize=5, label='Iterative' if i == 0 else "")
        else:
            iterative_bar = ax.bar(iterative_x, iterative_data_point[1], 
                                 color=iterative_color, alpha=0.8, width=bar_width, label='Iterative' if i == 0 else "")
        iterative_bars.append((iterative_bar, iterative_data_point))
        
        # Add accuracy labels above bars
        ax.text(concat_x, concat_data_point[1] + 0.01, f'{concat_data_point[1]*100:.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.text(iterative_x, iterative_data_point[1] + 0.01, f'{iterative_data_point[1]*100:.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add word count labels below each bar
        ax.text(concat_x, -0.05, f'~{int(concat_data_point[0])}', 
                ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=9)
        ax.text(iterative_x, -0.05, f'~{int(iterative_data_point[0])}', 
                ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=9)
        
        # Add constraint label below the pair of bars
        pair_center = x_base[i]
        ax.text(pair_center, -0.12, category, ha='center', va='top', 
                transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold')
    
    # Add chunk bars if we have chunk data
    chunk_x_positions = []
    if chunks_pre_data:  # Only show pre-reveal chunks, not 'all'
        # Position chunk bars to the right of summary bars with less spacing
        chunk_start_x = x_base[-1] + 1.5  # Reduced gap from 3.0 to 1.5
        chunk_colors = ['#228B22', '#32CD32']  # Dark green, lime green
        
        chunk_x = chunk_start_x
        # Comment out the 'all' chunk bar as requested
        # if chunks_all_data:
        #     chunk_all_avg_accuracy = np.mean([d[1] for d in chunks_all_data])
        #     chunk_all_avg_words = np.mean([d[0] for d in chunks_all_data])
        #     
        #     chunk_bar = ax.bar(chunk_x, chunk_all_avg_accuracy, 
        #                      color=chunk_colors[0], alpha=0.8, width=bar_width, 
        #                      label='Chunks (all)')
        #     
        #     # Add accuracy and word count labels
        #     ax.text(chunk_x, chunk_all_avg_accuracy + 0.01, f'{chunk_all_avg_accuracy*100:.1f}', 
        #             ha='center', va='bottom', fontweight='bold', fontsize=10)
        #     ax.text(chunk_x, -0.05, f'~{int(chunk_all_avg_words)}', 
        #             ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=9)
        #     
        #     chunk_x_positions.append(chunk_x)
        #     chunk_x += bar_width + 0.2  # Small spacing between chunk bars
        
        if chunks_pre_data:
            chunk_pre_avg_accuracy = np.mean([d[1] for d in chunks_pre_data])
            chunk_pre_avg_words = np.mean([d[0] for d in chunks_pre_data])
            
            chunk_bar = ax.bar(chunk_x, chunk_pre_avg_accuracy, 
                             color=chunk_colors[1], alpha=0.8, width=bar_width,
                             label='Chunks (pre-reveal)')
            
            # Add accuracy and word count labels
            ax.text(chunk_x, chunk_pre_avg_accuracy + 0.01, f'{chunk_pre_avg_accuracy*100:.1f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
            ax.text(chunk_x, -0.05, f'~{int(chunk_pre_avg_words)}', 
                    ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=9)
            
            chunk_x_positions.append(chunk_x)
        
        # Add chunk section label
        if chunk_x_positions:
            chunk_center = np.mean(chunk_x_positions)
            ax.text(chunk_center, -0.12, 'Source', ha='center', va='top', 
                    transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold')
    
    # Customize the plot
    ax.set_ylabel('Culprit Accuracy', fontsize=12)
    # Set y-axis limit based on dataset - true-detective has lower accuracy scores
    if dataset == "true-detective":
        ax.set_ylim(0, 0.6)
    else:
        ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis - extend to include chunk bars if present
    all_x_positions = list(x_base)
    all_x_positions.extend(chunk_x_positions)
    ax.set_xticks(all_x_positions)
    ax.set_xticklabels([''] * len(all_x_positions))  # Hide default x-axis labels since we have custom ones
    
    # Add random baseline line
    if random_baseline is not None:
        ax.axhline(y=random_baseline, color='red', linestyle='--', alpha=0.7, linewidth=2,
                  label=f'Random Baseline: {random_baseline:.1%}')
    
    # Add dotted line for first chunk only evaluations
    if first_chunk_data:
        # Calculate average accuracy for first chunk evaluations
        first_chunk_avg_accuracy = np.mean([d[1] for d in first_chunk_data])
        ax.axhline(y=first_chunk_avg_accuracy, color='gray', linestyle='--', linewidth=2, 
                  label='Only read first chunk', alpha=0.8)
    
    # Add human baseline line
    if human_estimate is not None:
        ax.axhline(y=human_estimate, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                  label=f'Human Estimate: {human_estimate:.1%}')
    
    # Add legend - position based on dataset
    if dataset == "detectiveqa":
        ax.legend(loc='lower left')
    else:
        ax.legend(loc='upper left')
    
    plt.tight_layout(pad=4.0)  # Add padding for constraint labels
    
    # Add x-axis label at the very bottom of the figure
    plt.figtext(0.5, 0.02, 'Summary Length Constraint', ha='center', fontsize=12)
    
    # Generate filename based on error bar setting
    error_suffix = "_with_ci" if include_error_bars else ""
    filename = f'summary_length_comparison_{dataset}{error_suffix}.png'
    output_path = f'../plots/{filename}'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("Summary Length Analysis:")
    print("=" * 50)
    print(f"Showing {len(sorted_categories)} constraint categories with both concat and iterative methods:")
    
    for category in sorted_categories:
        concat_data_point = concat_dict[category]
        iterative_data_point = iterative_dict[category]
        
        print(f"\n{category} constraint:")
            if include_error_bars:
            print(f"  Concat:    {int(concat_data_point[0]):3d} words ({int(concat_data_point[3]):4d} chars): {concat_data_point[1]:.1%} ± {concat_data_point[4]:.1%} (95% CI)")
            print(f"  Iterative: {int(iterative_data_point[0]):3d} words ({int(iterative_data_point[3]):4d} chars): {iterative_data_point[1]:.1%} ± {iterative_data_point[4]:.1%} (95% CI)")
            else:
            print(f"  Concat:    {int(concat_data_point[0]):3d} words ({int(concat_data_point[3]):4d} chars): {concat_data_point[1]:.1%} accuracy")
            print(f"  Iterative: {int(iterative_data_point[0]):3d} words ({int(iterative_data_point[3]):4d} chars): {iterative_data_point[1]:.1%} accuracy")
    
    # Overall averages
    if sorted_categories:
        concat_avg_acc = np.mean([concat_dict[cat][1] for cat in sorted_categories])
        concat_avg_words = np.mean([concat_dict[cat][0] for cat in sorted_categories])
        iter_avg_acc = np.mean([iterative_dict[cat][1] for cat in sorted_categories])
        iter_avg_words = np.mean([iterative_dict[cat][0] for cat in sorted_categories])
        
        print(f"\nOverall Averages:")
        print(f"  Concat:    {concat_avg_acc:.1%} accuracy, {concat_avg_words:.0f} words")
        print(f"  Iterative: {iter_avg_acc:.1%} accuracy, {iter_avg_words:.0f} words")
    
    # First chunk only results
    if first_chunk_data:
        first_chunk_avg_acc = np.mean([d[1] for d in first_chunk_data])
        first_chunk_avg_words = np.mean([d[0] for d in first_chunk_data])
        print(f"\nFirst Chunk Only:")
        print(f"  Found {len(first_chunk_data)} evaluation(s)")
        for words, acc, cat, chars, bootstrap_err in first_chunk_data:
            if include_error_bars:
                print(f"    {int(words):3d} words ({int(chars):4d} chars): {acc:.1%} ± {bootstrap_err:.1%} (95% CI) [{cat}]")
            else:
                print(f"    {int(words):3d} words ({int(chars):4d} chars): {acc:.1%} accuracy [{cat}]")
        print(f"  Average: {first_chunk_avg_acc:.1%} accuracy, {first_chunk_avg_words:.0f} words")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create summary length comparison charts")
    parser.add_argument("dataset", nargs="?", default="bmds", 
                       choices=["bmds", "true-detective", "detectiveqa"],
                       help="Dataset to analyze (default: bmds)")
    parser.add_argument("--no-error-bars", action="store_true",
                       help="Disable bootstrap confidence interval error bars")
    
    args = parser.parse_args()
    
    include_error_bars = not args.no_error_bars
    
    print(f"Creating charts for dataset: {args.dataset}")
    if include_error_bars:
        print("Including bootstrap 95% confidence intervals")
    else:
        print("Error bars disabled")
    
    filename = create_summary_length_charts(args.dataset, include_error_bars)
    print(f"\nChart saved as: {filename}")