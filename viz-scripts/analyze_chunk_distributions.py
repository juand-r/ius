#!/usr/bin/env python3
"""
Analyze chunk distributions across all chunk output directories.
Creates histograms showing the number of chunks per item for each directory.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_collection_data(collection_path):
    """Load collection.json and extract chunk counts per item."""
    try:
        with open(collection_path, 'r') as f:
            data = json.load(f)
        
        chunk_counts = []
        word_counts = []
        items_dir = collection_path.parent / "items"
        
        # Load each item file to count chunks and words
        for item_id in data.get('items', []):
            item_file = items_dir / f"{item_id}.json"
            if item_file.exists():
                try:
                    with open(item_file, 'r') as f:
                        item_data = json.load(f)
                    
                    # Count chunks in first document
                    documents = item_data.get('documents', [])
                    if documents and 'chunks' in documents[0]:
                        chunk_count = len(documents[0]['chunks'])
                        chunk_counts.append(chunk_count)
                        
                        # Count total words in all chunks (excluding reveal segment if present)
                        chunks = documents[0]['chunks']
                        total_words = 0
                        
                        # Check if reveal segment was added (last chunk might be reveal)
                        chunking_params = item_data.get('item_metadata', {}).get('chunking_params', {})
                        has_reveal = chunking_params.get('reveal_add_on', False)
                        
                        # Count words in regular chunks (exclude reveal if present)
                        chunks_to_count = chunks[:-1] if has_reveal and len(chunks) > 1 else chunks
                        
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
        if '--count' in command_run:
            try:
                count_idx = command_run.split().index('--count')
                parameters['num_chunks'] = int(command_run.split()[count_idx + 1])
            except (ValueError, IndexError):
                pass
        
        return {
            'chunk_counts': chunk_counts,
            'word_counts': word_counts,
            'total_items': data.get('num_items', len(chunk_counts)),
            'strategy': strategy,
            'parameters': parameters
        }
    except Exception as e:
        print(f"Error loading {collection_path}: {e}")
        return None

def analyze_chunks_directories():
    """Analyze all directories in outputs/chunks/"""
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
                data = load_collection_data(collection_path)
                if data and data['chunk_counts']:
                    results[subdir.name] = data
                else:
                    print(f"  ⚠️  No valid chunk data found")
            else:
                print(f"  ⚠️  No collection.json found in {subdir.name}")
    
    return results

def create_histograms(results):
    """Create histograms for each directory."""
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
        chunk_counts = data['chunk_counts']
        
        # Create histogram
        bins = range(min(chunk_counts), max(chunk_counts) + 2)
        ax.hist(chunk_counts, bins=bins, alpha=0.7, color=colors[i], edgecolor='black')
        
        # Determine dataset name and chunk size for title
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
        ax.set_title(f"{dataset_name}\ntarget chunk size: {chunk_size}, n={len(chunk_counts)}", 
                    fontsize=10, pad=10)
        ax.set_xlabel("Number of chunks per item")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        # Add statistics text (top right)
        mean_chunks = np.mean(chunk_counts)
        std_chunks = np.std(chunk_counts)
        min_chunks = min(chunk_counts)
        max_chunks = max(chunk_counts)
        
        stats_text = f"μ={mean_chunks:.1f}, σ={std_chunks:.1f}\nRange: {min_chunks}-{max_chunks}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = "../plots/chunk_distributions_histograms.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Histograms saved to: {output_file}")
    
    return fig

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("CHUNK DISTRIBUTION SUMMARY")
    print("="*80)
    
    for dir_name, data in results.items():
        chunk_counts = data['chunk_counts']
        word_counts = data['word_counts']
        print(f"\n📁 {dir_name}")
        print(f"   Strategy: {data['strategy']}")
        print(f"   Parameters: {data['parameters']}")
        print(f"   Items: {len(chunk_counts)}")
        print(f"   Chunks per item: {min(chunk_counts)}-{max(chunk_counts)} "
              f"(μ={np.mean(chunk_counts):.1f}, σ={np.std(chunk_counts):.1f})")
        
        # Word count statistics
        if word_counts:
            print(f"   Words per item: {min(word_counts)}-{max(word_counts)} "
                  f"(μ={np.mean(word_counts):.0f}, σ={np.std(word_counts):.0f})")
        
        # Show chunk distribution
        unique_counts = sorted(set(chunk_counts))
        for count in unique_counts:
            freq = chunk_counts.count(count)
            pct = (freq / len(chunk_counts)) * 100
            print(f"     {count} chunks: {freq} items ({pct:.1f}%)")

def main():
    print("🔍 Analyzing chunk distributions...")
    
    results = analyze_chunks_directories()
    
    if not results:
        print("No chunk directories found with valid data!")
        return
    
    print(f"\n✅ Found {len(results)} directories with chunk data")
    
    # Create histograms
    create_histograms(results)
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main()
