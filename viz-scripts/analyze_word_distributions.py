#!/usr/bin/env python3
"""
Quick throwaway script to analyze word count distributions across chunk directories.
Creates histograms showing word counts per story and calculates averages.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_word_data(collection_path):
    """Load collection.json and extract word counts per item."""
    try:
        with open(collection_path, 'r') as f:
            data = json.load(f)
        
        word_counts = []
        items_dir = collection_path.parent / "items"
        
        # Load each item file to count words
        for item_id in data.get('items', []):
            item_file = items_dir / f"{item_id}.json"
            if item_file.exists():
                try:
                    with open(item_file, 'r') as f:
                        item_data = json.load(f)
                    
                    # Count words in first document
                    documents = item_data.get('documents', [])
                    if documents and 'chunks' in documents[0]:
                        chunks = documents[0]['chunks']
                        total_words = 0
                        
                        # Count words in all chunks (including reveal if present)
                        chunks_to_count = chunks
                        
                        for chunk in chunks_to_count:
                            total_words += len(chunk.split())
                        
                        word_counts.append(total_words)
                except Exception as e:
                    print(f"  Error loading {item_file}: {e}")
        
        # Get strategy and parameters from chunking_info
        chunking_info = data.get('chunking_info', {})
        strategy = chunking_info.get('strategy', 'unknown')
        
        # Extract parameters from command_run if available
        command_run = chunking_info.get('command_run', '')
        parameters = {}
        if '--size' in command_run:
            try:
                size_idx = command_run.split().index('--size')
                parameters['chunk_size'] = int(command_run.split()[size_idx + 1])
            except (ValueError, IndexError):
                pass
        
        return {
            'word_counts': word_counts,
            'total_items': data.get('num_items', len(word_counts)),
            'strategy': strategy,
            'parameters': parameters
        }
    except Exception as e:
        print(f"Error loading {collection_path}: {e}")
        return None

def analyze_word_directories():
    """Analyze all directories in outputs/chunks/ for word distributions."""
    chunks_dir = Path("../outputs/chunks")
    
    if not chunks_dir.exists():
        print("../outputs/chunks directory not found!")
        return
    
    results = {}
    
    do_not_process = ["detectiveqa_fixed_size_8000", "true-detective_fixed_size_1500", "squality_fixed_size_8000"]
    
    # Find all subdirectories with collection.json
    for subdir in chunks_dir.iterdir():
        if subdir.is_dir():
            collection_path = subdir / "collection.json"
            if collection_path.exists() and subdir.name not in do_not_process:
                print(f"Analyzing {subdir.name}...")
                data = load_word_data(collection_path)
                if data and data['word_counts']:
                    results[subdir.name] = data
                else:
                    print(f"  ⚠️  No valid word data found")
            else:
                print(f"  ⚠️  Skipping {subdir.name}")
    
    return results

def create_word_histograms(results):
    """Create word count histograms for each directory."""
    if not results:
        print("No data to plot!")
        return
    
    # Calculate grid dimensions
    n_dirs = len(results)
    cols = min(3, n_dirs)  # Max 3 columns
    rows = (n_dirs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_dirs == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Hide extra subplots
    for i in range(n_dirs, len(axes)):
        axes[i].set_visible(False)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_dirs))
    
    # Order results: BMDS, True Detective, DetectiveQA
    def get_sort_key(item):
        key, _ = item
        if 'bmds' in key.lower():
            return (0, key)  # BMDS first
        elif 'true-detective' in key.lower():
            return (1, key)  # True Detective second  
        elif 'detectiveqa' in key.lower():
            return (2, key)  # DetectiveQA last
        else:
            return (3, key)  # Others at the end
    
    ordered_results = sorted(results.items(), key=get_sort_key)
    
    for i, (dir_name, data) in enumerate(ordered_results):
        ax = axes[i]
        word_counts = data['word_counts']
        
        # Create histogram with reasonable bins
        ax.hist(word_counts, bins=20, alpha=0.7, color=colors[i], edgecolor='black')
        
        # Determine dataset name for title
        if 'bmds' in dir_name.lower():
            dataset_name = "BMDS"
        elif 'true-detective' in dir_name.lower():
            dataset_name = "True Detective"
        elif 'detectiveqa' in dir_name.lower():
            dataset_name = "DetectiveQA"
        else:
            dataset_name = dir_name.replace('_', ' ').title()
        
        # Customize plot
        ax.set_title(f"{dataset_name}\nn={len(word_counts)}", 
                    fontsize=10, pad=10)
        ax.set_xlabel("Words per story")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        # Add statistics text (top right)
        mean_words = np.mean(word_counts)
        std_words = np.std(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)
        
        stats_text = f"μ={mean_words:.0f}, σ={std_words:.0f}\nRange: {min_words:,}-{max_words:,}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = "../plots/word_distributions_histograms.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Word distribution histograms saved to: {output_file}")
    
    return fig

def print_word_summary(results):
    """Print summary statistics for word counts."""
    print("\n" + "="*80)
    print("WORD COUNT DISTRIBUTION SUMMARY")
    print("="*80)
    
    for dir_name, data in results.items():
        word_counts = data['word_counts']
        print(f"\n📁 {dir_name}")
        print(f"   Strategy: {data['strategy']}")
        print(f"   Parameters: {data['parameters']}")
        print(f"   Items: {len(word_counts)}")
        print(f"   Words per story: {min(word_counts):,}-{max(word_counts):,} "
              f"(μ={np.mean(word_counts):.0f}, σ={np.std(word_counts):.0f})")
        
        # Show some percentiles
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            val = np.percentile(word_counts, p)
            print(f"     {p}th percentile: {val:,.0f} words")

def main():
    print("🔍 Analyzing word count distributions...")
    
    results = analyze_word_directories()
    
    if not results:
        print("No chunk directories found with valid data!")
        return
    
    print(f"\n✅ Found {len(results)} directories with word data")
    
    # Create word histograms
    create_word_histograms(results)
    
    # Print summary
    print_word_summary(results)

if __name__ == "__main__":
    main()