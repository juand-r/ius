#!/usr/bin/env python3
"""Create enhanced visualizations for whodunit analysis with baseline accuracy."""

import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

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

def calculate_baseline_accuracy():
    """Calculate baseline accuracy from number of suspects across all BMDS items."""
    
    base_path = Path("outputs/eval/extrinsic")
    suspect_counts = []
    
    # Collect suspect counts from all evaluation directories
    for eval_dir in base_path.iterdir():
        if not eval_dir.is_dir() or 'bmds' not in eval_dir.name:
            continue
            
        items_dir = eval_dir / "items"
        if not items_dir.exists():
            continue
            
        for item_file in items_dir.glob("*.json"):
            try:
                with open(item_file, 'r') as f:
                    item_data = json.load(f)
                
                suspects_str = item_data.get('parsed_response', {}).get('suspects', '')
                if suspects_str and suspects_str != 'None':
                    # Count suspects by splitting on commas and cleaning
                    suspects_list = [s.strip() for s in suspects_str.split(',') if s.strip()]
                    suspect_counts.append(len(suspects_list))
                    
            except Exception as e:
                continue
    
    if not suspect_counts:
        return 0.2  # Default fallback
    
    # Calculate average of 1/num_suspects for each item
    baseline_accuracies = [1.0 / count for count in suspect_counts]
    average_baseline = np.mean(baseline_accuracies)
    
    print(f"Baseline calculation:")
    print(f"  - Total items analyzed: {len(suspect_counts)}")
    print(f"  - Suspect counts range: {min(suspect_counts)}-{max(suspect_counts)}")
    print(f"  - Average suspects per item: {np.mean(suspect_counts):.1f}")
    print(f"  - Baseline accuracy (avg of 1/num_suspects): {average_baseline:.3f}")
    
    return average_baseline

def get_summary_annotations():
    """Get annotations for summary data points."""
    
    # Based on the analysis results, map lengths to categories and methods
    annotations = {
        3036: "Concat\n(~600w)",  # Average concat length
        4113: "Iterative\n(~800w)",  # Average iterative length
        1262: "Concat\n(<300w)",  # Short concat
        2810: "Concat\n(~560w)",  # Medium concat
        3188: "Concat\n(~640w)",  # Medium concat
        4939: "Concat\n(~990w)",  # Long concat
        3972: "Iterative\n(~790w)",  # Iterative
        4285: "Iterative\n(~860w)",  # Iterative
    }
    
    return annotations

def create_enhanced_charts():
    """Create enhanced charts with annotations and baseline."""
    
    # Load evaluation results
    results = load_evaluation_results()
    
    # Calculate baseline accuracy
    baseline_accuracy = calculate_baseline_accuracy()
    
    # Calculate dynamic values from actual data
    # Chunks (all)
    chunks_all = [r for r in results if r['input_type'] == 'chunks' and r['range_spec'] == 'all']
    chunks_all_acc = sum(r['culprit_accuracy'] * r['total_items'] for r in chunks_all) / sum(r['total_items'] for r in chunks_all) if chunks_all else 0
    chunks_all_len = sum(r['avg_length'] * r['total_items'] for r in chunks_all) / sum(r['total_items'] for r in chunks_all) if chunks_all else 0
    
    # Chunks (pre-reveal)
    chunks_pre = [r for r in results if r['input_type'] == 'chunks' and r['range_spec'] in ['penultimate', 'all-but-last']]
    chunks_pre_acc = sum(r['culprit_accuracy'] * r['total_items'] for r in chunks_pre) / sum(r['total_items'] for r in chunks_pre) if chunks_pre else 0
    chunks_pre_len = sum(r['avg_length'] * r['total_items'] for r in chunks_pre) / sum(r['total_items'] for r in chunks_pre) if chunks_pre else 0
    
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
    print(f"  Concat (pre-reveal): {concat_pre_acc:.3f} accuracy, {concat_pre_len:.0f} avg chars")
    print(f"  Iterative (pre-reveal): {iterative_pre_acc:.3f} accuracy, {iterative_pre_len:.0f} avg chars")
    
    # Data from the analysis
    conditions = ['Chunks\n(all)', 'Chunks\n(pre-reveal)', 'Concat\n(pre-reveal)', 'Iterative\n(pre-reveal)']
    accuracies = [chunks_all_acc, chunks_pre_acc, concat_pre_acc, iterative_pre_acc]
    lengths = [chunks_all_len, chunks_pre_len, concat_pre_len, iterative_pre_len]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Chart 1: Accuracy comparison
    colors = ['#2E8B57', '#4682B4', '#CD853F', '#9370DB']
    bars1 = ax1.bar(conditions, accuracies, color=colors, alpha=0.8)
    ax1.set_ylabel('Culprit Accuracy', fontsize=12)
    ax1.set_title('Whodunit Evaluation Accuracy by Condition', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    
    # Add baseline line to bar chart
    ax1.axhline(y=baseline_accuracy, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(0.02, baseline_accuracy + 0.02, f'Baseline: {baseline_accuracy:.3f}', 
             transform=ax1.get_yaxis_transform(), color='red', fontweight='bold')
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Chart 2: Enhanced Length vs Accuracy scatter
    # Separate chunk and summary data
    chunk_lengths = [chunks_all_len, chunks_pre_len]
    chunk_accuracies = [chunks_all_acc, chunks_pre_acc]
    summary_lengths = [concat_pre_len, iterative_pre_len]
    summary_accuracies = [concat_pre_acc, iterative_pre_acc]
    
    # Plot chunks
    ax2.scatter(chunk_lengths, chunk_accuracies, color='#2E8B57', s=120, 
               label='Chunks', alpha=0.8, marker='o')
    
    # Plot summaries with annotations
    summary_points = ax2.scatter(summary_lengths, summary_accuracies, color='#9370DB', s=120, 
                                label='Summaries', alpha=0.8, marker='s')
    
    # Add baseline line
    ax2.axhline(y=baseline_accuracy, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Baseline ({baseline_accuracy:.3f})')
    
    ax2.set_xlabel('Average Text Length (characters)', fontsize=12)
    ax2.set_ylabel('Culprit Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Text Length with Baseline', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for chunks
    ax2.annotate('Full chunks\n(76.1%)', (36408, 0.761), xytext=(32000, 0.72),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5), 
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    ax2.annotate('Pre-reveal chunks\n(61.8%)', (30228, 0.618), xytext=(26000, 0.55),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5), 
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    # Add annotations for summaries
    ax2.annotate('Concat summaries\n(~600w, 47.4%)', (3036, 0.474), xytext=(5000, 0.42),
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5), 
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='plum', alpha=0.7))
    
    ax2.annotate('Iterative summaries\n(~800w, 50.0%)', (4113, 0.500), xytext=(6500, 0.52),
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5), 
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='plum', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('whodunit_evaluation_charts_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a detailed summary chart showing all individual data points
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
    
    # Individual data points from the analysis
    individual_data = [
        # Chunks
        (chunks_all_len, chunks_all_acc, 'Chunks (all)', '#2E8B57', 'o'),
        (chunks_pre_len, chunks_pre_acc, 'Chunks (pre-reveal)', '#4682B4', 'o'),
        
        # Concat summaries
        (1262, 0.441, 'Concat (<300w)', '#CD853F', 's'),
        (2810, 0.471, 'Concat (~560w)', '#CD853F', 's'),
        (3188, 0.559, 'Concat (~640w)', '#CD853F', 's'),
        (4939, 0.424, 'Concat (~990w)', '#CD853F', 's'),
        
        # Iterative summaries  
        (3972, 0.529, 'Iterative (~790w)', '#9370DB', '^'),
        (4285, 0.464, 'Iterative (~860w)', '#9370DB', '^'),
    ]
    
    # Plot all points
    for length, accuracy, label, color, marker in individual_data:
        ax3.scatter(length, accuracy, color=color, s=100, alpha=0.8, marker=marker)
        
        # Add text annotation for each point
        ax3.annotate(f'{label}\n({accuracy:.1%})', (length, accuracy), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color))
    
    ax3.set_xlabel('Text Length (characters)', fontsize=12)
    ax3.set_ylabel('Culprit Accuracy', fontsize=12)
    ax3.set_title('Detailed Accuracy vs Text Length Analysis', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('whodunit_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_enhanced_charts()
    print("Enhanced charts saved as:")
    print("  - whodunit_evaluation_charts_enhanced.png")
    print("  - whodunit_detailed_analysis.png")
