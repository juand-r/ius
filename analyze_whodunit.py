#!/usr/bin/env python3
"""Analyze whodunit evaluation results."""

import json
import pandas as pd
from pathlib import Path

def load_data():
    """Load evaluation data."""
    data = []
    base_path = Path("outputs/eval/extrinsic")
    
    for eval_dir in base_path.iterdir():
        if not eval_dir.is_dir():
            continue
            
        print(f"Processing {eval_dir.name}...")
        
        # Load collection metadata
        collection_file = eval_dir / "collection.json"
        if not collection_file.exists():
            continue
            
        with open(collection_file, 'r') as f:
            collection_data = json.load(f)
        
        eval_info = collection_data.get('whodunit_evaluation_info', {})
        collection_meta = eval_info.get('collection_metadata', {})
        
        # Parse summary method from directory name
        dir_name = eval_dir.name
        summary_method = None
        if 'concat' in dir_name:
            summary_method = 'concat'
        elif 'iterative' in dir_name:
            summary_method = 'iterative'
        
        # Load item results
        items_dir = eval_dir / "items"
        if not items_dir.exists():
            continue
            
        for item_file in items_dir.glob("*.json"):
            try:
                with open(item_file, 'r') as f:
                    item_data = json.load(f)
                
                assessment = item_data.get('solution_correctness_assessment', {})
                if not assessment:
                    continue
                
                row = {
                    'directory': eval_dir.name,
                    'item_id': item_data.get('item_metadata', {}).get('item_id'),
                    'input_type': collection_meta.get('input_type'),
                    'range_spec': collection_meta.get('range_spec'),
                    'summary_method': summary_method,
                    'text_length': item_data.get('item_metadata', {}).get('selected_text_length'),
                    'culprit_correct': assessment.get('culprit', {}).get('culprit_correct') == 'Yes',
                    'accomplice_correct': assessment.get('accomplice', {}).get('accomplice_correct') == 'Yes',
                }
                
                # Add error types
                culprit_errors = assessment.get('culprit', {}).get('major_errors', {})
                row.update({
                    'different_suspect': culprit_errors.get('different_suspect_not_accomplice') == 'Yes',
                    'confused_accomplice': culprit_errors.get('confused_swapped_culprit_and_accomplice') == 'Yes',
                    'only_alias': culprit_errors.get('missing_real_name_only_has_alias') == 'Yes',
                    'included_others': culprit_errors.get('included_other_non_accomplice_suspects') == 'Yes',
                })
                
                data.append(row)
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    return pd.DataFrame(data)

def analyze_data(df):
    """Analyze the data and answer questions."""
    
    print("="*60)
    print("WHODUNIT EVALUATION ANALYSIS")
    print("="*60)
    
    print(f"\nOVERALL STATS:")
    print(f"Total evaluations: {len(df)}")
    print(f"Culprit accuracy: {df['culprit_correct'].mean():.3f}")
    print(f"Accomplice accuracy: {df['accomplice_correct'].mean():.3f}")
    
    # Question 1: Chunk accuracy
    print(f"\n1. CHUNK ACCURACY (especially 'all' range):")
    print("-" * 40)
    
    chunks = df[df['input_type'] == 'chunks']
    if len(chunks) > 0:
        chunk_stats = chunks.groupby('range_spec').agg({
            'culprit_correct': ['count', 'mean'],
            'text_length': 'mean'
        }).round(3)
        print(chunk_stats)
        
        # Focus on "all" range
        all_chunks = chunks[chunks['range_spec'] == 'all']
        if len(all_chunks) > 0:
            print(f"\n'All' range chunks:")
            print(f"  Accuracy: {all_chunks['culprit_correct'].mean():.3f}")
            print(f"  Avg length: {all_chunks['text_length'].mean():.0f} chars")
            
            # Mistakes
            wrong = all_chunks[~all_chunks['culprit_correct']]
            if len(wrong) > 0:
                print(f"  Mistakes ({len(wrong)} cases):")
                print(f"    Different suspect: {wrong['different_suspect'].mean():.3f}")
                print(f"    Confused w/ accomplice: {wrong['confused_accomplice'].mean():.3f}")
    
    # Question 2: Pre-reveal comparison
    print(f"\n2. PRE-REVEAL (chunks vs summaries):")
    print("-" * 40)
    
    pre_reveal = df[df['range_spec'].isin(['all-but-last', 'penultimate'])]
    if len(pre_reveal) > 0:
        comparison = pre_reveal.groupby(['input_type', 'summary_method']).agg({
            'culprit_correct': ['count', 'mean'],
            'text_length': 'mean'
        }).round(3)
        print(comparison)
    
    # Question 3: Length effect on concat
    print(f"\n3. CONCAT LENGTH EFFECT:")
    print("-" * 40)
    
    concat = df[df['summary_method'] == 'concat']
    if len(concat) > 0:
        concat['length_cat'] = pd.cut(concat['text_length'], 
                                    bins=[0, 2500, 5000, 10000, float('inf')],
                                    labels=['<500w', '500-1000w', '1000-2000w', '>2000w'])
        
        length_stats = concat.groupby('length_cat').agg({
            'culprit_correct': ['count', 'mean']
        }).round(3)
        print(length_stats)
    
    # Question 4: Concat vs iterative
    print(f"\n4. CONCAT vs ITERATIVE:")
    print("-" * 40)
    
    summaries = df[df['summary_method'].isin(['concat', 'iterative'])]
    if len(summaries) > 0:
        method_stats = summaries.groupby(['summary_method', 'range_spec']).agg({
            'culprit_correct': ['count', 'mean'],
            'text_length': 'mean'
        }).round(3)
        print(method_stats)
    
    return df

if __name__ == "__main__":
    df = load_data()
    if len(df) > 0:
        df.to_csv('whodunit_data.csv', index=False)
        analyze_data(df)
        print(f"\nData saved to whodunit_data.csv")
    else:
        print("No data found!")