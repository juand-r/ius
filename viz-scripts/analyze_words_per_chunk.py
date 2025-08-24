#!/usr/bin/env python3
"""
Script to analyze words per chunk distributions across chunk directories.
Creates histograms showing the distribution of words per individual chunk.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_chunk_word_data(collection_path):
    """Load collection.json and extract words per chunk data."""
    try:
        with open(collection_path, 'r') as f:
            data = json.load(f)
        
        words_per_chunk = []
        items_dir = collection_path.parent / "items"
        
        # Load each item file to count words per chunk
        for item_id in data.get('items', []):
            item_file = items_dir / f"{item_id}.json"
            if item_file.exists():
                try:
                    with open(item_file, 'r') as f:
                        item_data = json.load(f)
                    
                    # Count words in each chunk
                    documents = item_data.get('documents', [])
                    if documents and 'chunks' in documents[0]:
                        chunks = documents[0]['chunks']
                        
                        # Check if reveal segment was added (last chunk might be reveal)
                        chunking_params = item_data.get('item_metadata', {}).get('chunking_params', {})
                        has_reveal = chunking_params.get('reveal_add_on', False)
                        
                        # Count words in regular chunks (exclude reveal if present)
                        chunks_to_count = chunks[:-1] if has_reveal and len(chunks) > 1 else chunks
                        
                        for chunk in chunks_to_count:
                            chunk_word_count = len(chunk.split())
                            words_per_chunk.append(chunk_word_count)
                            
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
        if '--count' in command_run:
            try:
                count_idx = command_run.split().index('--count')
                parameters['num_chunks'] = int(command_run.split()[count_idx + 1])
            except (ValueError, IndexError):
                pass
        
        return {
            'words_per_chunk': words_per_chunk,
            'total_chunks': len(words_per_chunk),
            'total_items': data.get('num_items', 0),
            'strategy': strategy,
            'parameters': parameters
        }
    except Exception as e:
        print(f"Error loading {collection_path}: {e}")
        return None

def analyze_chunk_word_directories():
    """Analyze all directories in outputs/chunks/ for words per chunk."""
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
                data = load_chunk_word_data(collection_path)
                if data and data['words_per_chunk']:
                    results[subdir.name] = data
                else:
                    print(f"  ⚠️  No valid chunk word data found")
            else:
                print(f"  ⚠️  Skipping {subdir.name}")
    
    return results

def create_words_per_chunk_histograms(results):
    """Create histograms for words per chunk distributions."""
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
        words_per_chunk = data['words_per_chunk']
        
        # Create histogram with reasonable bins
        bins = min(30, len(set(words_per_chunk)))  # Adaptive bin count
        ax.hist(words_per_chunk, bins=bins, alpha=0.7, color=colors[i], edgecolor='black')
        
        # Determine dataset name for title
        if 'bmds' in dir_name.lower():
            dataset_name = "BMDS"
        elif 'true-detective' in dir_name.lower():
            dataset_name = "True Detective"
        elif 'detectiveqa' in dir_name.lower():
            dataset_name = "DetectiveQA"
        else:
            dataset_name = dir_name.replace('_', ' ').title()
        
        # Extract chunk size from parameters
        chunk_size = data['parameters'].get('chunk_size', 'unknown')
        
        # Customize plot
        ax.set_title(f"{dataset_name}\ntarget chunk size: {chunk_size}, n={data['total_chunks']} chunks", 
                    fontsize=10, pad=10)
        ax.set_xlabel("Words per chunk")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        # Add statistics text (top right)
        mean_words = np.mean(words_per_chunk)
        std_words = np.std(words_per_chunk)
        min_words = min(words_per_chunk)
        max_words = max(words_per_chunk)
        
        stats_text = f"μ={mean_words:.0f}, σ={std_words:.0f}\nRange: {min_words:,}-{max_words:,}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = "../plots/words_per_chunk_histograms.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Words per chunk histograms saved to: {output_file}")
    
    return fig

def print_chunk_word_summary(results):
    """Print summary statistics for words per chunk."""
    print("\n" + "="*80)
    print("WORDS PER CHUNK DISTRIBUTION SUMMARY")
    print("="*80)
    
    for dir_name, data in results.items():
        words_per_chunk = data['words_per_chunk']
        print(f"\n📁 {dir_name}")
        print(f"   Strategy: {data['strategy']}")
        print(f"   Parameters: {data['parameters']}")
        print(f"   Total chunks: {len(words_per_chunk)} (from {data['total_items']} items)")
        print(f"   Words per chunk: {min(words_per_chunk):,}-{max(words_per_chunk):,} "
              f"(μ={np.mean(words_per_chunk):.0f}, σ={np.std(words_per_chunk):.0f})")
        
        # Show some percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            val = np.percentile(words_per_chunk, p)
            print(f"     {p}th percentile: {val:,.0f} words")
        
        # Show target vs actual
        target_size = data['parameters'].get('chunk_size', 'unknown')
        if target_size != 'unknown':
            actual_mean = np.mean(words_per_chunk)
            print(f"   Target vs actual: {target_size:,} → {actual_mean:.0f} ({actual_mean/target_size*100:.1f}% of target)")

def main():
    print("🔍 Analyzing words per chunk distributions...")
    
    results = analyze_chunk_word_directories()
    
    if not results:
        print("No chunk directories found with valid data!")
        return
    
    print(f"\n✅ Found {len(results)} directories with chunk word data")
    
    # Create histograms
    create_words_per_chunk_histograms(results)
    
    # Print summary
    print_chunk_word_summary(results)

if __name__ == "__main__":
    main()