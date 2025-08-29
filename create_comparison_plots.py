#!/usr/bin/env python3
"""Create comparison plots showing concat vs iterative progressions for different evaluation types."""

import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.colors as mcolors

def darken_color(color, amount=0.3):
    """Darken a hex color by reducing its brightness."""
    try:
        c = mcolors.hex2color(color)
        c = [max(0, channel - amount) for channel in c]
        return mcolors.rgb2hex(c)
    except:
        return color

def find_evaluation_directories(optional_summary_length, eval_type, base_dir_map, dataset=None):
    """Find concat and iterative directories matching the summary length constraint."""
    
    base_dir = base_dir_map[eval_type]
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Base directory not found: {base_path}")
        return None, None
    
    # Search for directories and check their source collection metadata
    concat_dir = None
    iterative_dir = None
    
    # Create mapping for common length constraint patterns based on actual data
    length_mappings = {
        "very long": "and very long summary (as long as you can make it, try to reach 5000 words if possible)",
        "<50": "summary in less than 50 words",
        "<100": "summary in less than 100 words", 
        "<200": "summary in less than 200 words",
        "<500": "summary in less than 500 words",
        "<1000": "summary in less than 1000 words",
        "summary": "summary"
    }
    
    # Get the pattern to search for
    search_pattern = length_mappings.get(optional_summary_length, optional_summary_length)
    
    for dir_path in base_path.iterdir():
        if not dir_path.is_dir():
            continue
        
        # Filter by dataset if specified
        if dataset and not dir_path.name.startswith(dataset):
            continue
            
        # Check if this is the correct evaluation type directory
        eval_markers = {
            'rouge': 'rouge_multi',
            'supert': 'supert_multi',
            'entity-coverage': 'entity_coverage_multi'
        }
        
        if eval_markers[eval_type] not in dir_path.name:
            continue
        
        # Read the collection.json to find the source collection
        collection_json = dir_path / "collection.json"
        if not collection_json.exists():
            continue
            
        try:
            with open(collection_json) as f:
                collection_data = json.load(f)
                
            # Handle different evaluation structures
            if eval_type == 'entity-coverage':
                # Entity-coverage has different structure
                eval_info = collection_data.get('entity_coverage_multi_evaluation_info', {})
            elif eval_type == 'rouge':
                eval_info = collection_data.get('rouge_multi_evaluation_info', {})
            elif eval_type == 'supert':
                eval_info = collection_data.get('supert_multi_evaluation_info', {})
            else:
                eval_info = {}
                
            source_collection = eval_info.get("collection_metadata", {}).get("source_collection")
            if not source_collection:
                continue
                
            # Read the source collection metadata
            source_collection_json = Path(source_collection) / "collection.json"
            if not source_collection_json.exists():
                continue
                
            with open(source_collection_json) as f:
                source_data = json.load(f)
                
            # Get the optional_summary_length from source (try different structures)
            source_length = (
                source_data.get("summarization_info", {}).get("collection_metadata", {}).get("optional_summary_length", "") or
                source_data.get("collection_metadata", {}).get("optional_summary_length", "")
            )
            
            # Check if this matches our search pattern
            # For exact mapping, do exact match; otherwise do substring match
            if optional_summary_length in length_mappings:
                # Exact match for mapped patterns
                matches = source_length == search_pattern
            else:
                # Substring match for custom patterns
                matches = search_pattern.lower() in source_length.lower()
                
            if matches:
                dir_name = dir_path.name
                
                if "concat" in dir_name:
                    if concat_dir is None:  # Take the first match
                        concat_dir = dir_path
                        print(f"Found concat match: {dir_path.name} (length: '{source_length}')")
                elif "iterative" in dir_name:
                    if iterative_dir is None:  # Take the first match
                        iterative_dir = dir_path
                        print(f"Found iterative match: {dir_path.name} (length: '{source_length}')")
                        
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error reading metadata for {dir_path.name}: {e}")
            continue
    
    return concat_dir, iterative_dir

def load_evaluation_data_from_directory(eval_dir, eval_type):
    """Load evaluation data from a specific directory."""
    
    eval_dir = Path(eval_dir)
    items_dir = eval_dir / "items"
    
    if not items_dir.exists():
        print(f"Items directory not found: {items_dir}")
        return {}
    
    story_data = {}  # {story_id: {metric: [(position, score), ...]}}
    
    # Process each story directory
    for story_dir in items_dir.iterdir():
        if not story_dir.is_dir():
            continue
            
        story_id = story_dir.name
        story_data[story_id] = defaultdict(list)
        
        # Process each chunk file in the story (sort numerically by filename)
        chunk_files = sorted([f for f in story_dir.iterdir() if f.name.endswith('.json')],
                            key=lambda f: int(f.stem) if f.stem.isdigit() else float('inf'))
        
        for chunk_file in chunk_files:
            try:
                chunk_num = int(chunk_file.stem)  # e.g., "1.json" -> 1
            except ValueError:
                continue
                
            try:
                with open(chunk_file) as f:
                    data = json.load(f)
                    
                if eval_type == 'rouge':
                    # Extract ROUGE scores
                    if 'rouge_score' in data:
                        rouge_scores = data['rouge_score']
                        
                        # Process each ROUGE variant
                        for rouge_key, metrics in rouge_scores.items():
                            if isinstance(metrics, dict):
                                for metric_name, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        # Store as percentage (already in percentage format)
                                        metric_key = f"{rouge_key}_{metric_name}"
                                        story_data[story_id][metric_key].append((chunk_num, value))
                
                elif eval_type == 'supert':
                    # Extract SUPERT scores
                    if 'supert_score' in data:
                        supert_score = data['supert_score']
                        if isinstance(supert_score, (int, float)):
                            # Store SUPERT score scaled by 100 for visualization (0-1 -> 0-100)
                            metric_key = "supert_score"
                            story_data[story_id][metric_key].append((chunk_num, supert_score * 100))
                
                elif eval_type == 'entity-coverage':
                    # Extract entity coverage metrics
                    if 'entity_analysis' in data and 'metrics' in data['entity_analysis']:
                        metrics = data['entity_analysis']['metrics']
                        
                        for metric_name in ['jaccard_similarity', 'recall', 'precision']:
                            if metric_name in metrics and isinstance(metrics[metric_name], (int, float)):
                                # Store as percentage (convert to percentage if needed)
                                metric_key = f"entity_coverage_{metric_name}"
                                value = metrics[metric_name] * 100  # Convert to percentage
                                story_data[story_id][metric_key].append((chunk_num, value))
                                    
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"Error processing {chunk_file}: {e}")
                continue
        
        # Sort all metrics by chunk number
        for metric_key in story_data[story_id]:
            story_data[story_id][metric_key].sort(key=lambda x: x[0])
    
    print(f"Loaded data for {len(story_data)} stories from {eval_dir}")
    return story_data

def create_comparison_plots(concat_data, iterative_data, optional_summary_length, eval_type, output_dir=".", dataset=""):
    """Create comparison plots showing concat vs iterative progressions."""
    
    if not concat_data or not iterative_data:
        print("Missing data for comparison!")
        return []
    
    # Find common stories between both datasets
    common_stories = set(concat_data.keys()) & set(iterative_data.keys())
    if not common_stories:
        print("No common stories found between concat and iterative data!")
        return []
    
    print(f"Comparing {len(common_stories)} common stories")
    
    # Define variants and metrics based on evaluation type
    if eval_type == 'rouge':
        eval_variants = ['rs-rouge2', 'rs-rougeLsum', 'rouge-l', 'rouge-1']
        metric_types = ['recall', 'precision']
    elif eval_type == 'supert':
        eval_variants = ['supert']
        metric_types = ['score']
    elif eval_type == 'entity-coverage':
        eval_variants = ['entity_coverage']
        metric_types = ['jaccard_similarity', 'recall', 'precision']
    
    # Base colors for different stories
    base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#9B59B6', 
                   '#E67E22', '#1ABC9C', '#F39C12', '#34495E', '#E74C3C',
                   '#8E44AD', '#16A085', '#F39C12', '#27AE60', '#E67E22',
                   '#3498DB', '#E74C3C', '#9B59B6']
    
    saved_files = []
    
    # Create a separate plot for each evaluation variant and metric type combination
    for eval_variant in eval_variants:
        for metric_type in metric_types:
            # Create figure with 2 rows (top: first 17 stories, bottom: last 17 stories)
            fig, axes = plt.subplots(2, 1, figsize=(24, 12))
            fig.suptitle(f'{eval_variant.upper()} {metric_type.replace("_", " ").title()}: Concat vs Iterative\n'
                         f'({optional_summary_length})', fontsize=16, fontweight='bold')
            
            # Sort common stories for consistent ordering
            sorted_stories = sorted(common_stories)
            
            # Split stories into two groups (max 17 each)
            story_groups = [
                ("First 17 Stories", sorted_stories[:17]),
                ("Next 17 Stories", sorted_stories[17:34])
            ]
            
            # Construct metric key based on evaluation type
            if eval_type == 'rouge':
                metric = f"{eval_variant}_{metric_type}"
            elif eval_type == 'supert':
                metric = "supert_score"
            elif eval_type == 'entity-coverage':
                metric = f"entity_coverage_{metric_type}"
            
            # Check if we have data for this metric in both datasets
            has_concat_data = any(metric in concat_data[story_id] and concat_data[story_id][metric]
                                 for story_id in sorted_stories)
            has_iterative_data = any(metric in iterative_data[story_id] and iterative_data[story_id][metric]
                                    for story_id in sorted_stories)
            
            if not has_concat_data or not has_iterative_data:
                for ax in axes:
                    ax.text(0.5, 0.5, f'No data for {metric_type}', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=10, color='gray')
                continue
            
            # Calculate shared y-limit for both subplots
            shared_y_max = 100  # Default
            if eval_type == 'rouge' and dataset == 'bmds':
                # Hardcode for bmds dataset
                shared_y_max = 20
            elif eval_type == 'rouge':
                all_scores_both_subplots = []
                for _, story_subset in story_groups:
                    for story_id in story_subset:
                        if (story_id in concat_data and metric in concat_data[story_id] and 
                            concat_data[story_id][metric] and
                            story_id in iterative_data and metric in iterative_data[story_id] and 
                            iterative_data[story_id][metric]):
                            concat_positions, concat_story_scores = zip(*concat_data[story_id][metric])
                            iterative_positions, iterative_story_scores = zip(*iterative_data[story_id][metric])
                            min_chunks = min(len(concat_story_scores), len(iterative_story_scores))
                            all_scores_both_subplots.extend(concat_story_scores[:min_chunks])
                            all_scores_both_subplots.extend(iterative_story_scores[:min_chunks])
                
                if all_scores_both_subplots:
                    max_val = max(all_scores_both_subplots)
                    y_max_raw = min(100, max_val + 5)
                    # Round to nearest 10
                    shared_y_max = round(y_max_raw / 10) * 10
                    if shared_y_max < y_max_raw:
                        shared_y_max += 10
            
            # Process each subplot (first 17, last 17)
            for subplot_idx, (group_name, story_subset) in enumerate(story_groups):
                ax = axes[subplot_idx]
                
                # Prepare data for this subplot
                x_positions_concat = []
                x_positions_iterative = []
                concat_scores = []
                iterative_scores = []
                story_centers = []
                story_ids_filtered = []
                
                current_pos = 0
                bar_width = 0.8
                story_color_mapping = {}  # Keep track of colors for each story
                
                for story_idx, story_id in enumerate(story_subset):
                    # Check if both datasets have data for this story
                    if (metric not in concat_data[story_id] or not concat_data[story_id][metric] or
                        metric not in iterative_data[story_id] or not iterative_data[story_id][metric]):
                        continue
                        
                    # Get data for both methods
                    concat_positions, concat_story_scores = zip(*concat_data[story_id][metric])
                    iterative_positions, iterative_story_scores = zip(*iterative_data[story_id][metric])
                    
                    # Align by chunk position (in case they have different chunks)
                    min_chunks = min(len(concat_story_scores), len(iterative_story_scores))
                    concat_story_scores = concat_story_scores[:min_chunks]
                    iterative_story_scores = iterative_story_scores[:min_chunks]
                    
                    # Get colors for this story
                    base_color = base_colors[story_idx % len(base_colors)]
                    concat_color = base_color
                    iterative_color = darken_color(base_color, 0.3)
                    story_color_mapping[story_id] = (concat_color, iterative_color)
                    
                    # Create concat bars first, then iterative bars for this story
                    concat_positions = []
                    iterative_positions = []
                    
                    # Concat bars (first K chunks)
                    for chunk_idx in range(min_chunks):
                        concat_pos = current_pos + chunk_idx
                        concat_positions.append(concat_pos)
                        x_positions_concat.append(concat_pos)
                        concat_scores.append(concat_story_scores[chunk_idx])
                    
                    # No gap between concat and iterative within same story
                    iterative_start = current_pos + min_chunks
                    
                    # Iterative bars (next K chunks)
                    for chunk_idx in range(min_chunks):
                        iterative_pos = iterative_start + chunk_idx
                        iterative_positions.append(iterative_pos)
                        x_positions_iterative.append(iterative_pos)
                        iterative_scores.append(iterative_story_scores[chunk_idx])
                    
                    # Calculate center position for story label (center of all bars for this story)
                    story_start = current_pos
                    story_end = iterative_start + min_chunks - 1
                    story_center = (story_start + story_end) / 2
                    story_centers.append(story_center)
                    story_ids_filtered.append(story_id)
                    
                    # Move to next story with gap between stories
                    current_pos = iterative_start + min_chunks + 1
                
                if not story_ids_filtered:
                    ax.text(0.5, 0.5, f'No data for {group_name}', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=10, color='gray')
                    continue
                
                # Prepare colors for each bar
                concat_colors = []
                iterative_colors = []
                
                current_pos = 0
                for story_idx, story_id in enumerate(story_ids_filtered):
                    # Get colors for this story
                    base_color = base_colors[story_idx % len(base_colors)]
                    concat_color = base_color
                    iterative_color = darken_color(base_color, 0.3)
                    
                    # Count chunks for this story
                    if story_id in concat_data and metric in concat_data[story_id]:
                        num_chunks = len(concat_data[story_id][metric])
                        concat_colors.extend([concat_color] * num_chunks)
                        iterative_colors.extend([iterative_color] * num_chunks)
                
                # Plot bars with individual colors
                bars_concat = ax.bar(x_positions_concat, concat_scores, width=bar_width, 
                                    color=concat_colors, alpha=0.8, label='Concat')
                bars_iterative = ax.bar(x_positions_iterative, iterative_scores, width=bar_width,
                                       color=iterative_colors, alpha=0.8, label='Iterative')
                
                # Set x-axis limits to remove empty space on left and right
                all_positions = x_positions_concat + x_positions_iterative
                if all_positions:
                    padding = 1.6  # Two bar widths of padding around the bars
                    ax.set_xlim(min(all_positions) - padding, max(all_positions) + padding)
                
                # Customize subplot
                
                # Show x-axis labels on both subplots
                ax.set_xticks(story_centers)
                ax.set_xticklabels(story_ids_filtered, fontsize=8, rotation=45, ha='right')
                
                # Set y-axis limits - use shared y-limit for consistency between subplots
                ax.set_ylim(0, shared_y_max)
                
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Add legend only to top subplot
                if subplot_idx == 0:
                    ax.legend(fontsize=12, loc='upper right')
            
            plt.tight_layout()
            
            # Save plot for this evaluation variant and metric combination
            filename = f'{eval_type}_comparison_{eval_variant}_{metric_type}_{dataset}_{optional_summary_length.replace("<", "lt").replace(">", "gt")}.png'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(str(filepath))
            print(f"Saved: {filename}")
            
            #plt.show()
    
    return saved_files

def create_dumbbell_plot(concat_data, iterative_data, optional_summary_length, eval_type, output_dir=".", dataset=""):
    """Create horizontal dumbbell plot showing first vs last chunk performance for each story."""
    
    if not concat_data or not iterative_data:
        print("Missing data for dumbbell plot!")
        return []
    
    # Find common stories between both datasets
    common_stories = set(concat_data.keys()) & set(iterative_data.keys())
    if not common_stories:
        print("No common stories found for dumbbell plot!")
        return []
    
    # Define variants and metrics based on evaluation type
    if eval_type == 'rouge':
        eval_variants = ['rs-rouge2', 'rs-rougeLsum', 'rouge-l', 'rouge-1']
        metric_types = ['recall', 'precision']
    elif eval_type == 'supert':
        eval_variants = ['supert']
        metric_types = ['score']
    elif eval_type == 'entity-coverage':
        eval_variants = ['entity_coverage']
        metric_types = ['jaccard_similarity', 'recall', 'precision']
    
    saved_files = []
    
    # Create a separate dumbbell plot for each evaluation variant and metric type
    for eval_variant in eval_variants:
        for metric_type in metric_types:
            # Construct metric key
            if eval_type == 'rouge':
                metric = f"{eval_variant}_{metric_type}"
            elif eval_type == 'supert':
                metric = "supert_score"
            elif eval_type == 'entity-coverage':
                metric = f"entity_coverage_{metric_type}"
            
            # Collect first vs last data for each story
            story_data = []
            
            for story_id in sorted(common_stories):
                # Get concat data
                if (story_id in concat_data and metric in concat_data[story_id] and 
                    len(concat_data[story_id][metric]) >= 2):
                    concat_scores = [score for _, score in concat_data[story_id][metric]]
                    concat_first = concat_scores[0]
                    concat_last = concat_scores[-1]
                else:
                    continue
                
                # Get iterative data
                if (story_id in iterative_data and metric in iterative_data[story_id] and 
                    len(iterative_data[story_id][metric]) >= 2):
                    iterative_scores = [score for _, score in iterative_data[story_id][metric]]
                    iterative_first = iterative_scores[0]
                    iterative_last = iterative_scores[-1]
                else:
                    continue
                
                story_data.append({
                    'story_id': story_id,
                    'concat_first': concat_first,
                    'concat_last': concat_last,
                    'iterative_first': iterative_first,
                    'iterative_last': iterative_last
                })
            
            if not story_data:
                print(f"No data for dumbbell plot: {metric}")
                continue
            
            # Split stories into two groups for two panes
            mid_point = len(story_data) // 2
            left_stories = story_data[:mid_point]
            right_stories = story_data[mid_point:]
            
            # Create figure with two horizontal panes (landscape orientation)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f'{eval_variant.upper()} {metric_type.replace("_", " ").title()}: First vs Last Chunk\n'
                         f'Concat vs Iterative ({optional_summary_length})', 
                         fontsize=16, fontweight='bold')
            
            # Plot left pane
            for i, data in enumerate(left_stories):
                y = len(left_stories) - 1 - i  # Reverse order so first story is at top
                
                # Concat arrow (orange)
                ax1.annotate('', xy=(data['concat_last'], y + 0.1), xytext=(data['concat_first'], y + 0.1),
                           arrowprops=dict(arrowstyle='->', lw=1.5, color='#FF7F50'))
                
                # Iterative arrow (purple)
                ax1.annotate('', xy=(data['iterative_last'], y - 0.1), xytext=(data['iterative_first'], y - 0.1),
                           arrowprops=dict(arrowstyle='->', lw=1.5, color='#9370DB'))
            
            # Plot right pane
            for i, data in enumerate(right_stories):
                y = len(right_stories) - 1 - i  # Reverse order so first story is at top
                
                # Concat arrow (orange)
                ax2.annotate('', xy=(data['concat_last'], y + 0.1), xytext=(data['concat_first'], y + 0.1),
                           arrowprops=dict(arrowstyle='->', lw=1.5, color='#FF7F50'))
                
                # Iterative arrow (purple)
                ax2.annotate('', xy=(data['iterative_last'], y - 0.1), xytext=(data['iterative_first'], y - 0.1),
                           arrowprops=dict(arrowstyle='->', lw=1.5, color='#9370DB'))
            
            # Add legend manually to the first pane
            ax1.plot([], [], color='#FF7F50', linewidth=2, label='Concat')
            ax1.plot([], [], color='#9370DB', linewidth=2, label='Iterative')
            
            # Customize left pane
            ax1.set_yticks(range(len(left_stories)))
            ax1.set_yticklabels([data['story_id'] for data in reversed(left_stories)], fontsize=9)
            ax1.set_xlabel(f'{metric_type.replace("_", " ").title()} Score', fontsize=12)
            ax1.set_xlim(0, 100)
            ax1.grid(True, alpha=0.3, axis='x')
            ax1.legend(loc='lower right')
            ax1.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
            
            # Customize right pane
            ax2.set_yticks(range(len(right_stories)))
            ax2.set_yticklabels([data['story_id'] for data in reversed(right_stories)], fontsize=9)
            ax2.set_xlabel(f'{metric_type.replace("_", " ").title()} Score', fontsize=12)
            ax2.set_xlim(0, 100)
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save plot
            filename = f'{eval_type}_dumbbell_{eval_variant}_{metric_type}_{dataset}_{optional_summary_length.replace("<", "lt").replace(">", "gt")}.png'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(str(filepath))
            print(f"Saved dumbbell plot: {filename}")
            
            plt.close()
    
    return saved_files

def create_chunk_average_plot(concat_data, iterative_data, optional_summary_length, eval_type, output_dir=".", dataset=""):
    """Create bar plot showing average performance by chunk position (concat vs iterative)."""
    
    if not concat_data or not iterative_data:
        print("Missing data for chunk average plot!")
        return []
    
    # Find common stories between both datasets
    common_stories = set(concat_data.keys()) & set(iterative_data.keys())
    if not common_stories:
        print("No common stories found for chunk average plot!")
        return []
    
    # Define variants and metrics based on evaluation type
    if eval_type == 'rouge':
        eval_variants = ['rs-rouge2', 'rs-rougeLsum', 'rouge-l', 'rouge-1']
        metric_types = ['recall', 'precision']
    elif eval_type == 'supert':
        eval_variants = ['supert']
        metric_types = ['score']
    elif eval_type == 'entity-coverage':
        eval_variants = ['entity_coverage']
        metric_types = ['jaccard_similarity', 'recall', 'precision']
    
    saved_files = []
    
    # Create a separate chunk average plot for each evaluation variant and metric type
    for eval_variant in eval_variants:
        for metric_type in metric_types:
            # Construct metric key
            if eval_type == 'rouge':
                metric = f"{eval_variant}_{metric_type}"
            elif eval_type == 'supert':
                metric = "supert_score"
            elif eval_type == 'entity-coverage':
                metric = f"entity_coverage_{metric_type}"
            
            # Collect scores by chunk position
            concat_by_chunk = defaultdict(list)  # {chunk_num: [scores]}
            iterative_by_chunk = defaultdict(list)
            
            for story_id in common_stories:
                # Process concat data
                if story_id in concat_data and metric in concat_data[story_id]:
                    for chunk_num, score in concat_data[story_id][metric]:
                        concat_by_chunk[chunk_num].append(score)
                
                # Process iterative data
                if story_id in iterative_data and metric in iterative_data[story_id]:
                    for chunk_num, score in iterative_data[story_id][metric]:
                        iterative_by_chunk[chunk_num].append(score)
            
            # Find common chunk positions
            concat_chunks = set(concat_by_chunk.keys())
            iterative_chunks = set(iterative_by_chunk.keys())
            common_chunks = sorted(concat_chunks & iterative_chunks)
            
            if not common_chunks:
                print(f"No common chunks for chunk average plot: {metric}")
                continue
            
            # Calculate averages and standard errors
            chunk_positions = []
            concat_means = []
            concat_sems = []
            iterative_means = []
            iterative_sems = []
            
            for chunk_num in common_chunks:
                concat_scores = concat_by_chunk[chunk_num]
                iterative_scores = iterative_by_chunk[chunk_num]
                
                if concat_scores and iterative_scores:
                    chunk_positions.append(chunk_num)
                    
                    # Concat stats
                    concat_mean = np.mean(concat_scores)
                    concat_sem = np.std(concat_scores) / np.sqrt(len(concat_scores))
                    concat_means.append(concat_mean)
                    concat_sems.append(concat_sem)
                    
                    # Iterative stats
                    iterative_mean = np.mean(iterative_scores)
                    iterative_sem = np.std(iterative_scores) / np.sqrt(len(iterative_scores))
                    iterative_means.append(iterative_mean)
                    iterative_sems.append(iterative_sem)
            
            if not chunk_positions:
                print(f"No valid chunk data for: {metric}")
                continue
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(max(8, len(chunk_positions) * 1.2), 6))
            fig.suptitle(f'{eval_variant.upper()} {metric_type.replace("_", " ").title()}: Average by Chunk Position\n'
                         f'Concat vs Iterative ({optional_summary_length})', 
                         fontsize=14, fontweight='bold')
            
            # Create bars
            x_positions = np.arange(len(chunk_positions))
            width = 0.35
            
            bars_concat = ax.bar(x_positions - width/2, concat_means, width, 
                               yerr=concat_sems, capsize=5, 
                               color='#FF7F50', alpha=0.8, label='Concat')
            bars_iterative = ax.bar(x_positions + width/2, iterative_means, width,
                                  yerr=iterative_sems, capsize=5,
                                  color='#9370DB', alpha=0.8, label='Iterative')
            
            # Add value labels on bars
            for i, (concat_mean, iterative_mean) in enumerate(zip(concat_means, iterative_means)):
                ax.text(i - width/2, concat_mean + concat_sems[i] + 1, f'{concat_mean:.1f}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
                ax.text(i + width/2, iterative_mean + iterative_sems[i] + 1, f'{iterative_mean:.1f}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Customize plot
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f'Chunk {pos}' for pos in chunk_positions], fontsize=10)
            ax.set_xlabel('Chunk Position', fontsize=12)
            ax.set_ylabel(f'{metric_type.replace("_", " ").title()} Score', fontsize=12)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper left')
            
            plt.tight_layout()
            
            # Save plot
            filename = f'{eval_type}_chunk_avg_{eval_variant}_{metric_type}_{dataset}_{optional_summary_length.replace("<", "lt").replace(">", "gt")}.png'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(str(filepath))
            print(f"Saved chunk average plot: {filename}")
            
            plt.close()
    
    return saved_files

def create_chunk_average_diffs_plot(concat_data, iterative_data, optional_summary_length, eval_type, output_dir=".", dataset="", add_error_bars=False):
    """Create bar plot showing average differences between consecutive chunks (concat vs iterative)."""
    
    if not concat_data or not iterative_data:
        print("Missing data for chunk average diffs plot!")
        return []
    
    # Find common stories between both datasets
    common_stories = set(concat_data.keys()) & set(iterative_data.keys())
    if not common_stories:
        print("No common stories found for chunk average diffs plot!")
        return []
    
    # Define variants and metrics based on evaluation type
    if eval_type == 'rouge':
        eval_variants = ['rs-rouge2', 'rs-rougeLsum', 'rouge-l', 'rouge-1']
        metric_types = ['recall', 'precision']
    elif eval_type == 'supert':
        eval_variants = ['supert']
        metric_types = ['score']
    elif eval_type == 'entity-coverage':
        eval_variants = ['entity_coverage']
        metric_types = ['jaccard_similarity', 'recall', 'precision']
    
    saved_files = []
    
    # Create a separate chunk average diffs plot for each evaluation variant and metric type
    for eval_variant in eval_variants:
        for metric_type in metric_types:
            # Construct metric key
            if eval_type == 'rouge':
                metric = f"{eval_variant}_{metric_type}"
            elif eval_type == 'supert':
                metric = "supert_score"
            elif eval_type == 'entity-coverage':
                metric = f"entity_coverage_{metric_type}"
            
            # Collect differences by diff position (diff1, diff2, etc.)
            concat_diffs_by_position = defaultdict(list)  # {diff_pos: [diff_values]}
            iterative_diffs_by_position = defaultdict(list)
            
            # Store all diffs for overall mean calculation
            all_concat_diffs = []
            all_iterative_diffs = []
            
            for story_id in common_stories:
                # Process concat data
                if story_id in concat_data and metric in concat_data[story_id]:
                    scores = [score for _, score in sorted(concat_data[story_id][metric])]
                    if len(scores) >= 2:
                        for i in range(len(scores) - 1):
                            diff = scores[i + 1] - scores[i]  # chunk_{i+2} - chunk_{i+1}
                            diff_pos = i + 1  # diff1, diff2, etc.
                            concat_diffs_by_position[diff_pos].append(diff)
                            all_concat_diffs.append(diff)
                
                # Process iterative data
                if story_id in iterative_data and metric in iterative_data[story_id]:
                    scores = [score for _, score in sorted(iterative_data[story_id][metric])]
                    if len(scores) >= 2:
                        for i in range(len(scores) - 1):
                            diff = scores[i + 1] - scores[i]  # chunk_{i+2} - chunk_{i+1}
                            diff_pos = i + 1  # diff1, diff2, etc.
                            iterative_diffs_by_position[diff_pos].append(diff)
                            all_iterative_diffs.append(diff)
            
            # Find common diff positions (limit to Diff 4)
            concat_diff_positions = set(concat_diffs_by_position.keys())
            iterative_diff_positions = set(iterative_diffs_by_position.keys())
            common_diff_positions = sorted(concat_diff_positions & iterative_diff_positions)
            # Limit to only Diff 1-4
            common_diff_positions = [pos for pos in common_diff_positions if pos <= 4]
            
            if not common_diff_positions:
                print(f"No common diff positions for chunk average diffs plot: {metric}")
                continue
            
            # Calculate averages and standard errors for each diff position
            diff_positions = []
            concat_means = []
            concat_sems = []
            iterative_means = []
            iterative_sems = []
            
            for diff_pos in common_diff_positions:
                concat_diffs = concat_diffs_by_position[diff_pos]
                iterative_diffs = iterative_diffs_by_position[diff_pos]
                
                if concat_diffs and iterative_diffs:
                    diff_positions.append(diff_pos)
                    
                    # Concat stats
                    concat_mean = np.mean(concat_diffs)
                    concat_sem = np.std(concat_diffs) / np.sqrt(len(concat_diffs))
                    concat_means.append(concat_mean)
                    concat_sems.append(concat_sem)
                    
                    # Iterative stats
                    iterative_mean = np.mean(iterative_diffs)
                    iterative_sem = np.std(iterative_diffs) / np.sqrt(len(iterative_diffs))
                    iterative_means.append(iterative_mean)
                    iterative_sems.append(iterative_sem)
            
            if not diff_positions:
                print(f"No valid diff data for: {metric}")
                continue
            
            # Calculate overall means
            overall_concat_mean = np.mean(all_concat_diffs) if all_concat_diffs else 0
            overall_iterative_mean = np.mean(all_iterative_diffs) if all_iterative_diffs else 0
            
            print(f"Overall mean diff for {metric}:")
            print(f"  Concat: {overall_concat_mean:.3f}")
            print(f"  Iterative: {overall_iterative_mean:.3f}")
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(max(8, len(diff_positions) * 1.2), 6))
            
            # Create bars
            x_positions = np.arange(len(diff_positions))
            width = 0.35
            
            bars_concat = ax.bar(x_positions - width/2, concat_means, width, 
                               color='#FF7F50', alpha=0.8, label='Concat',
                               yerr=concat_sems if add_error_bars else None, 
                               capsize=5 if add_error_bars else 0,
                               error_kw={'ecolor': 'lightgray', 'alpha': 0.7, 'elinewidth': 1.5} if add_error_bars else {})
            bars_iterative = ax.bar(x_positions + width/2, iterative_means, width,
                                  color='#9370DB', alpha=0.8, label='Iterative',
                                  yerr=iterative_sems if add_error_bars else None,
                                  capsize=5 if add_error_bars else 0,
                                  error_kw={'ecolor': 'lightgray', 'alpha': 0.7, 'elinewidth': 1.5} if add_error_bars else {})
            
            # Add value labels on bars
            for i, (concat_mean, iterative_mean) in enumerate(zip(concat_means, iterative_means)):
                y_concat = concat_mean + 0.05 if concat_mean >= 0 else concat_mean - 0.05
                y_iterative = iterative_mean + 0.05 if iterative_mean >= 0 else iterative_mean - 0.05
                
                ax.text(i - width/2, y_concat, f'{concat_mean:.2f}', 
                       ha='center', va='bottom' if concat_mean >= 0 else 'top', fontweight='bold', fontsize=9)
                ax.text(i + width/2, y_iterative, f'{iterative_mean:.2f}', 
                       ha='center', va='bottom' if iterative_mean >= 0 else 'top', fontweight='bold', fontsize=9)
            
            # Customize plot
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f'Diff {pos}' for pos in diff_positions], fontsize=10)
            ax.set_xlabel('Consecutive Chunk Difference Position', fontsize=12)
            ax.set_ylabel(f'{metric_type.replace("_", " ").title()} Score Difference', fontsize=12)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)  # Reference line at 0
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper left')
            
            # Set y-axis ticks to 0, 0.5, 1.0, etc.
            y_min, y_max = ax.get_ylim()
            y_ticks = np.arange(np.floor(y_min * 2) / 2, np.ceil(y_max * 2) / 2 + 0.5, 0.5)
            ax.set_yticks(y_ticks)
            
            # Add more space between x-tick labels and x-axis label
            ax.tick_params(axis='x', pad=10)
            
            plt.tight_layout()
            
            # Save plot
            filename = f'{eval_type}_chunk_avg_diffs_{eval_variant}_{metric_type}_{dataset}_{optional_summary_length.replace("<", "lt").replace(">", "gt")}.png'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(str(filepath))
            print(f"Saved chunk average diffs plot: {filename}")
            
            plt.close()
    
    return saved_files

def create_line_plots(concat_data, iterative_data, optional_summary_length, eval_type, output_dir=".", dataset=""):
    """Create line plots showing individual story progressions (5 stories max)."""
    
    if not concat_data or not iterative_data:
        print("Missing data for line plots!")
        return []
    
    # Find common stories between both datasets
    common_stories = set(concat_data.keys()) & set(iterative_data.keys())
    if not common_stories:
        print("No common stories found for line plots!")
        return []
    
    # Select 10 random stories for line plots (2x5 grid) with fixed seed for reproducibility
    # Only consider stories with at least 4 chunks for meaningful progressions
    import random
    
    # Filter stories that have at least 4 chunks in both concat and iterative data
    # Use a representative metric to check chunk count (first available metric)
    if eval_type == 'rouge':
        check_metric = 'rs-rouge2_recall'
    elif eval_type == 'supert':
        check_metric = 'supert_score'
    elif eval_type == 'entity-coverage':
        check_metric = 'entity_coverage_jaccard_similarity'
    
    stories_with_enough_chunks = []
    for story_id in common_stories:
        concat_chunks = len(concat_data.get(story_id, {}).get(check_metric, []))
        iterative_chunks = len(iterative_data.get(story_id, {}).get(check_metric, []))
        if concat_chunks >= 4 and iterative_chunks >= 4:
            stories_with_enough_chunks.append(story_id)
    
    if not stories_with_enough_chunks:
        print("No stories found with at least 4 chunks!")
        return []
    
    # Set random seed based on dataset and eval_type to ensure consistent selection across different summary lengths
    all_stories = sorted(stories_with_enough_chunks)
    seed_string = f"{dataset}_{eval_type}"
    random.seed(hash(seed_string) % (2**32))  # Convert string to consistent integer seed
    selected_stories = random.sample(all_stories, min(10, len(all_stories)))
    
    # Define variants and metrics based on evaluation type
    if eval_type == 'rouge':
        eval_variants = ['rs-rouge2', 'rs-rougeLsum', 'rouge-l', 'rouge-1']
        metric_types = ['recall', 'precision']
    elif eval_type == 'supert':
        eval_variants = ['supert']
        metric_types = ['score']
    elif eval_type == 'entity-coverage':
        eval_variants = ['entity_coverage']
        metric_types = ['jaccard_similarity', 'recall', 'precision']
    
    saved_files = []
    
    # Create a separate line plot for each evaluation variant and metric type
    for eval_variant in eval_variants:
        for metric_type in metric_types:
            # Construct metric key
            if eval_type == 'rouge':
                metric = f"{eval_variant}_{metric_type}"
            elif eval_type == 'supert':
                metric = "supert_score"
            elif eval_type == 'entity-coverage':
                metric = f"entity_coverage_{metric_type}"
            
            # Check if we have data for this metric
            valid_stories = []
            for story_id in selected_stories:
                has_concat = (story_id in concat_data and metric in concat_data[story_id] and 
                             len(concat_data[story_id][metric]) >= 2)
                has_iterative = (story_id in iterative_data and metric in iterative_data[story_id] and 
                               len(iterative_data[story_id][metric]) >= 2)
                if has_concat and has_iterative:
                    valid_stories.append(story_id)
            
            if not valid_stories:
                print(f"No valid stories for line plot: {metric}")
                continue
            
            # Limit to 10 stories for 2x5 grid
            plot_stories = valid_stories[:10]
            
            # Create figure with 2x5 subplots
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            
            # Flatten axes for easier indexing
            axes_flat = axes.flatten()
            
            # Plot each story in its own subplot
            for i, story_id in enumerate(plot_stories):
                ax = axes_flat[i]
                
                # Get concat data
                concat_chunks = sorted(concat_data[story_id][metric])
                concat_x = [chunk_num for chunk_num, _ in concat_chunks]
                concat_y = [score for _, score in concat_chunks]
                
                # Get iterative data
                iterative_chunks = sorted(iterative_data[story_id][metric])
                iterative_x = [chunk_num for chunk_num, _ in iterative_chunks]
                iterative_y = [score for _, score in iterative_chunks]
                
                # Plot concat line (dashed, orange)
                ax.plot(concat_x, concat_y, color='#FF7F50', linestyle='--', linewidth=2.5, 
                       marker='o', markersize=5)
                
                # Plot iterative line (solid, purple)
                ax.plot(iterative_x, iterative_y, color='#9370DB', linestyle='-', linewidth=2.5,
                       marker='s', markersize=5)
                
                # Customize subplot
                ax.set_title(story_id, fontsize=10, fontweight='bold')
                
                # Set y-limit based on evaluation type
                if eval_type == 'rouge' and dataset == 'bmds':
                    # Hardcode for bmds dataset
                    ax.set_ylim(0, 20)
                elif eval_type == 'rouge':
                    # For ROUGE, use dynamic y-limit: min(100, max(all_values) + 5), rounded to nearest 10
                    all_values = concat_y + iterative_y
                    if all_values:
                        max_val = max(all_values)
                        y_max_raw = min(100, max_val + 5)
                        # Round to nearest 10
                        y_max = round(y_max_raw / 10) * 10
                        if y_max < y_max_raw:
                            y_max += 10
                        ax.set_ylim(0, y_max)
                    else:
                        ax.set_ylim(0, 100)
                else:
                    # For other evaluation types, keep 0-100 range
                    ax.set_ylim(0, 100)
                
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Chunk', fontsize=9)
                ax.set_ylabel('Score', fontsize=9)
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                # Set x-axis to show only integer chunk numbers
                if concat_x or iterative_x:
                    all_chunks = set(concat_x + iterative_x)
                    integer_chunks = sorted([int(chunk) for chunk in all_chunks if chunk == int(chunk)])
                    if integer_chunks:
                        ax.set_xticks(integer_chunks)
                        ax.set_xticklabels([str(chunk) for chunk in integer_chunks])
            
            # Hide unused subplots
            for i in range(len(plot_stories), len(axes_flat)):
                axes_flat[i].set_visible(False)
            
            # Create legend in the last visible subplot or as a separate legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='#FF7F50', linestyle='--', linewidth=2.5, 
                       marker='o', markersize=5, label='Concat'),
                Line2D([0], [0], color='#9370DB', linestyle='-', linewidth=2.5,
                       marker='s', markersize=5, label='Iterative')
            ]
            
            # Add legend to the figure
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
            
            plt.tight_layout()
            
            # Save plot
            filename = f'{eval_type}_line_plots_{eval_variant}_{metric_type}_{dataset}_{optional_summary_length.replace("<", "lt").replace(">", "gt")}.png'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(str(filepath))
            print(f"Saved line plot: {filename}")
            
            plt.close()
    
    return saved_files

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create comparison plots for different evaluation types")
    parser.add_argument("optional_summary_length", 
                       help="Summary length constraint (e.g., '<200', '<500', 'summary', 'very long')")
    parser.add_argument("--base-dir", default="outputs/eval/intrinsic",
                       help="Base directory to search for evaluation results")
    parser.add_argument("--dataset", required=True,
                       help="Dataset to analyze (e.g., 'bmds', 'true-detective')")
    
    # Evaluation type arguments (mutually exclusive)
    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("--rouge", action="store_true", 
                           help="Generate ROUGE comparison plots")
    eval_group.add_argument("--supert", action="store_true",
                           help="Generate SUPERT comparison plots") 
    eval_group.add_argument("--entity-coverage", action="store_true",
                           help="Generate entity coverage comparison plots")
    
    # Plot type arguments
    parser.add_argument("--detailed", action="store_true", default=False,
                       help="Generate detailed comparison plots (original)")
    parser.add_argument("--dumbbell", action="store_true", default=False,
                       help="Generate dumbbell plots (first vs last chunk)")
    parser.add_argument("--chunk-averages", action="store_true", default=False,
                       help="Generate chunk average plots")
    parser.add_argument("--chunk-average-diffs", action="store_true", default=False,
                       help="Generate chunk average difference plots (consecutive chunk differences)")
    parser.add_argument("--line-plots", action="store_true", default=False,
                       help="Generate line plots for individual stories (5 stories max)")
    parser.add_argument("--all-plots", action="store_true", default=False,
                       help="Generate all plot types")
    parser.add_argument("--add-error-bars", action="store_true", default=False,
                       help="Add error bars showing standard error of the mean (for --chunk-average-diffs)")
    
    args = parser.parse_args()
    
    # If no plot type specified, default to detailed
    if not any([args.detailed, args.dumbbell, args.chunk_averages, args.chunk_average_diffs, args.line_plots, args.all_plots]):
        args.detailed = True
    
    # Determine evaluation type
    if args.rouge:
        eval_type = 'rouge'
    elif args.supert:
        eval_type = 'supert'
    elif args.entity_coverage:
        eval_type = 'entity-coverage'
    
    # Set up base directory mapping
    base_dir_map = {
        'rouge': f"{args.base_dir}/rouge",
        'supert': f"{args.base_dir}/supert", 
        'entity-coverage': f"{args.base_dir}/entity-coverage"
    }
    
    print(f"Looking for {eval_type} directories with summary length: {args.optional_summary_length} in dataset: {args.dataset}")
    
    # Find matching concat and iterative directories
    concat_dir, iterative_dir = find_evaluation_directories(args.optional_summary_length, eval_type, base_dir_map, args.dataset)
    
    if not concat_dir:
        print(f"Could not find concat directory with summary length: {args.optional_summary_length}")
        return
    
    if not iterative_dir:
        print(f"Could not find iterative directory with summary length: {args.optional_summary_length}")
        return
        
    print(f"Found concat directory: {concat_dir}")
    print(f"Found iterative directory: {iterative_dir}")
    
    # Load data from both directories
    print("Loading concat data...")
    concat_data = load_evaluation_data_from_directory(concat_dir, eval_type)
    
    print("Loading iterative data...")
    iterative_data = load_evaluation_data_from_directory(iterative_dir, eval_type)
    
    if not concat_data or not iterative_data:
        print("Failed to load data from one or both directories!")
        return
    
    # Create plots based on requested types
    all_saved_files = []
    
    if args.all_plots or args.detailed:
        print("Creating detailed comparison plots...")
        saved_files = create_comparison_plots(concat_data, iterative_data, args.optional_summary_length, eval_type, dataset=args.dataset)
        all_saved_files.extend(saved_files)
    
    if args.all_plots or args.dumbbell:
        print("Creating dumbbell plots...")
        saved_files = create_dumbbell_plot(concat_data, iterative_data, args.optional_summary_length, eval_type, dataset=args.dataset)
        all_saved_files.extend(saved_files)
    
    if args.all_plots or args.chunk_averages:
        print("Creating chunk average plots...")
        saved_files = create_chunk_average_plot(concat_data, iterative_data, args.optional_summary_length, eval_type, dataset=args.dataset)
        all_saved_files.extend(saved_files)
    
    if args.all_plots or args.chunk_average_diffs:
        print("Creating chunk average diffs plots...")
        saved_files = create_chunk_average_diffs_plot(concat_data, iterative_data, args.optional_summary_length, eval_type, dataset=args.dataset, add_error_bars=args.add_error_bars)
        all_saved_files.extend(saved_files)
    
    if args.all_plots or args.line_plots:
        print("Creating line plots...")
        saved_files = create_line_plots(concat_data, iterative_data, args.optional_summary_length, eval_type, dataset=args.dataset)
        all_saved_files.extend(saved_files)
    
    print(f"\nGenerated {len(all_saved_files)} plots:")
    for filename in all_saved_files:
        print(f"  {filename}")
    
    print(f"\nUsage examples:")
    print(f"  python {sys.argv[0]} '<200' --dataset bmds --rouge --detailed")
    print(f"  python {sys.argv[0]} 'summary' --dataset true-detective --supert --dumbbell") 
    print(f"  python {sys.argv[0]} 'very long' --dataset bmds --entity-coverage --chunk-averages")
    print(f"  python {sys.argv[0]} '<200' --dataset bmds --entity-coverage --chunk-average-diffs")
    print(f"  python {sys.argv[0]} '<200' --dataset bmds --entity-coverage --line-plots")
    print(f"  python {sys.argv[0]} '<500' --dataset bmds --entity-coverage --all-plots")

if __name__ == "__main__":
    main()