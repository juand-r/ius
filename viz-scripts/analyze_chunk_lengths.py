#!/usr/bin/env python3
"""
Script to analyze chunk lengths from a specified chunks directory
and create a scatter plot showing chunk lengths (in words) per story.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import Counter


def count_words(text):
    """Count words in a text string."""
    return len(text.split())
    #return len(text)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze chunk lengths from a chunks directory')
    parser.add_argument('--directory', '-d', required=True, 
                       help='Name of the directory in ../outputs/chunks to analyze (e.g., bmds_fixed_size2_8000)')
    
    args = parser.parse_args()
    
    # Path to the chunks directory
    chunks_dir = Path("../outputs/chunks") / args.directory
    items_dir = chunks_dir / "items"
    
    if not chunks_dir.exists():
        print(f"❌ Directory not found: {chunks_dir}")
        return
    
    print(f"📂 Analyzing chunks directory: {args.directory}")
    
    # Load collection to get story list
    collection_file = chunks_dir / "collection.json"
    if not collection_file.exists():
        print(f"❌ Collection file not found: {collection_file}")
        return
        
    with open(collection_file, 'r') as f:
        collection = json.load(f)
    
    story_ids = collection['items']
    print(f"📚 Found {len(story_ids)} stories")
    
    # Collect data for plotting
    x_positions_regular = []  # Story positions for regular chunks
    y_values_regular = []     # Chunk lengths for regular chunks
    x_positions_last = []     # Story positions for last chunks
    y_values_last = []        # Chunk lengths for last chunks
    story_labels = []         # Story IDs for x-axis labels
    chunks_per_story = []     # Number of chunks per story for histogram
    
    for i, story_id in enumerate(story_ids):
        story_file = items_dir / f"{story_id}.json"
        
        if not story_file.exists():
            print(f"⚠️  Missing file: {story_file}")
            continue
            
        # Load story data
        with open(story_file, 'r') as f:
            story_data = json.load(f)
        
        # Extract chunks from the first (and only) document
        chunks = story_data['documents'][0]['chunks']
        
        # Count words in each chunk
        chunk_lengths = [count_words(chunk) for chunk in chunks]
        num_chunks = len(chunks)
        
        print(f"📖 {story_id}: {num_chunks} chunks, lengths: {min(chunk_lengths)}-{max(chunk_lengths)} words")
        
        # Add data points for this story - separate last chunk from regular chunks
        for j, chunk_length in enumerate(chunk_lengths):
            if j == len(chunk_lengths) - 1:  # Last chunk
                x_positions_last.append(i)
                y_values_last.append(chunk_length)
            else:  # Regular chunk
                x_positions_regular.append(i)
                y_values_regular.append(chunk_length)
        
        story_labels.append(story_id)
        chunks_per_story.append(num_chunks)
    
    all_y_values = y_values_regular + y_values_last
    max_chunk_length = max(all_y_values)
    y_limit_upper = max_chunk_length + 200
    
    print(f"\n📊 Total data points: {len(all_y_values)}")
    print(f"📊 Regular chunks: {len(y_values_regular)}, Last chunks: {len(y_values_last)}")
    print(f"📊 Word count range: {min(all_y_values)}-{max_chunk_length} words")
    print(f"📊 Y-axis limit: 0-{y_limit_upper} words")
    
    # Create scatter plot
    plt.figure(figsize=(15, 8))
    
    # Plot regular chunks in blue
    if x_positions_regular:
        plt.scatter(x_positions_regular, y_values_regular, alpha=0.6, s=30, color='blue', label='Regular chunks')
    
    # Plot last chunks in red
    if x_positions_last:
        plt.scatter(x_positions_last, y_values_last, alpha=0.8, s=40, color='red', label='Last chunks')
    
    # Customize the plot
    plt.xlabel('Stories', fontsize=12)
    plt.ylabel('Chunk Length (words)', fontsize=12)
    plt.title(f'Chunk Length Distribution by Story - {args.directory}\n(Last Chunks in Red)', fontsize=14)
    plt.ylim(0, y_limit_upper)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set x-axis labels
    plt.xticks(range(len(story_labels)), story_labels, rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    #plt.tight_layout()
    
    # Save the plot
    output_file = f"../plots/chunk_lengths_scatter_{args.directory}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 Scatter plot saved as: {output_file}")
    # Close the plot to free memory
    plt.close()
    
    # Create histogram of number of chunks per story
    plt.figure(figsize=(12, 6))
    
    # Create histogram
    unique_chunk_counts = sorted(set(chunks_per_story))
    bins = [x - 0.5 for x in unique_chunk_counts] + [max(unique_chunk_counts) + 0.5]
    
    plt.hist(chunks_per_story, bins=bins, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.8)
    
    # Customize the histogram
    plt.xlabel('Number of Chunks per Story', fontsize=12)
    plt.ylabel('Number of Stories', fontsize=12)
    plt.title(f'Distribution of Chunks per Story - {args.directory}', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis ticks to show only integer values
    plt.xticks(unique_chunk_counts)
    
    # Add value labels on top of bars
    hist_counts, _, _ = plt.hist(chunks_per_story, bins=bins, alpha=0)  # Invisible hist to get counts
    for i, count in enumerate(hist_counts):
        if count > 0:
            plt.text(unique_chunk_counts[i], count + 0.1, str(int(count)), 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save the histogram
    #histogram_output_file = f"../plots/chunks_per_story_histogram_{args.directory}.png"
    #plt.savefig(histogram_output_file, dpi=300, bbox_inches='tight')
    #print(f"💾 Histogram saved as: {histogram_output_file}")
    # Close the plot to free memory
    #plt.close()
    
    # Print some statistics
    print(f"\n📈 Chunk Length Statistics:")
    print(f"   Average chunk length: {sum(all_y_values)/len(all_y_values):.1f} words")
    print(f"   Median chunk length: {sorted(all_y_values)[len(all_y_values)//2]} words")
    print(f"   Total chunks analyzed: {len(all_y_values)}")
    print(f"   Stories analyzed: {len(story_labels)}")
    if y_values_last:
        print(f"   Average last chunk length: {sum(y_values_last)/len(y_values_last):.1f} words")
        print(f"   Average regular chunk length: {sum(y_values_regular)/len(y_values_regular):.1f} words")
    
    print(f"\n📈 Chunk Count Statistics:")
    print(f"   Average chunks per story: {sum(chunks_per_story)/len(chunks_per_story):.1f}")
    print(f"   Median chunks per story: {sorted(chunks_per_story)[len(chunks_per_story)//2]}")
    print(f"   Range of chunks per story: {min(chunks_per_story)}-{max(chunks_per_story)}")
    
    # Show distribution details
    chunk_distribution = Counter(chunks_per_story)
    print(f"   Chunk count distribution:")
    for num_chunks in sorted(chunk_distribution.keys()):
        count = chunk_distribution[num_chunks]
        print(f"     {num_chunks} chunks: {count} stories")

if __name__ == "__main__":
    main()
