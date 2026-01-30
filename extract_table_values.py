#!/usr/bin/env python3
"""Extract values for the comparison table."""

import json
from pathlib import Path
from collections import defaultdict

def find_summary_hash(dataset, method):
    """Find the hash for summary constraint collections."""
    summaries_path = Path("outputs/summaries")
    
    for dir_path in summaries_path.iterdir():
        if not dir_path.is_dir():
            continue
        
        # Check if this matches dataset and method
        if dataset.lower() not in dir_path.name.lower():
            continue
        
        if method == "concat" and "concat" not in dir_path.name:
            continue
        if method == "iterative" and "iterative" not in dir_path.name:
            continue
        
        # Check if it has "summary" constraint
        collection_file = dir_path / "collection.json"
        if not collection_file.exists():
            continue
        
        with open(collection_file) as f:
            data = json.load(f)
        
        length = data.get('summarization_info', {}).get('collection_metadata', {}).get('optional_summary_length', '')
        
        if length == "summary":
            return dir_path.name
    
    return None

def get_whodunit_accuracy(dataset, summary_hash):
    """Get whodunit accuracy for a summary collection."""
    whodunit_path = Path("outputs/eval/extrinsic")
    
    for eval_dir in whodunit_path.iterdir():
        if not eval_dir.is_dir():
            continue
        
        if dataset.lower() not in eval_dir.name.lower():
            continue
        
        if summary_hash not in eval_dir.name:
            continue
        
        if "whodunit" not in eval_dir.name:
            continue
        
        # Calculate accuracy from individual item files
        items_dir = eval_dir / "items"
        if not items_dir.exists():
            continue
        
        total = 0
        correct = 0
        
        for item_file in items_dir.glob("*.json"):
            # Skip backup files
            if ".bak" in item_file.name:
                continue
            
            try:
                with open(item_file) as f:
                    item_data = json.load(f)
                
                total += 1
                
                # Check if culprit was correct
                culprit_correct = item_data.get('solution_correctness_assessment', {}).get('culprit', {}).get('culprit_correct')
                if culprit_correct == "Yes":
                    correct += 1
            except Exception as e:
                print(f"    Warning: Could not process {item_file}: {e}")
                continue
        
        if total > 0:
            return correct / total
    
    return None

def get_faithfulness_score(dataset, summary_hash):
    """Get mean faithfulness score."""
    faith_path = Path("outputs/eval/intrinsic/faithfulness")
    
    for eval_dir in faith_path.iterdir():
        if not eval_dir.is_dir():
            continue
        
        if dataset.lower() not in eval_dir.name.lower():
            continue
        
        if summary_hash not in eval_dir.name:
            continue
        
        collection_file = eval_dir / "collection.json"
        if not collection_file.exists():
            continue
        
        with open(collection_file) as f:
            data = json.load(f)
        
        # Get mean faithfulness score
        score = data.get('faithfulness_evaluation_info', {}).get('faithfulness_statistics', {}).get('mean_faithfulness_score')
        
        if score is not None:
            return score
    
    return None

def get_entity_coverage_recall(dataset, summary_hash):
    """Get entity coverage recall for last summaries."""
    ec_path = Path("outputs/eval/intrinsic/entity-coverage")
    
    for eval_dir in ec_path.iterdir():
        if not eval_dir.is_dir():
            continue
        
        if dataset.lower() not in eval_dir.name.lower():
            continue
        
        if summary_hash not in eval_dir.name:
            continue
        
        if "entity_coverage_multi" not in eval_dir.name:
            continue
        
        # Load all item files and get recall from last summaries
        items_dir = eval_dir / "items"
        if not items_dir.exists():
            continue
        
        recalls = []
        for item_dir in items_dir.iterdir():
            if not item_dir.is_dir():
                continue
            
            # Get the last JSON file (highest number)
            json_files = list(item_dir.glob("*.json"))
            if not json_files:
                continue
            
            json_files.sort(key=lambda x: int(x.stem))
            last_file = json_files[-1]
            
            with open(last_file) as f:
                item_data = json.load(f)
            
            recall = item_data.get('entity_analysis', {}).get('metrics', {}).get('recall')
            if recall is not None:
                recalls.append(recall)
        
        if recalls:
            return sum(recalls) / len(recalls)
    
    return None

def get_rouge_scores(dataset, summary_hash):
    """Get ROUGE scores (rs-rouge1, rs-rouge2, rs-rougeL) with P/R/F1."""
    rouge_path = Path("outputs/eval/intrinsic/rouge")
    
    for eval_dir in rouge_path.iterdir():
        if not eval_dir.is_dir():
            continue
        
        if dataset.lower() not in eval_dir.name.lower():
            continue
        
        if summary_hash not in eval_dir.name:
            continue
        
        if "rouge_multi" not in eval_dir.name:
            continue
        
        # Load all item files and get ROUGE scores from last summaries
        items_dir = eval_dir / "items"
        if not items_dir.exists():
            continue
        
        rouge1_p, rouge1_r, rouge1_f = [], [], []
        rouge2_p, rouge2_r, rouge2_f = [], [], []
        rougeL_p, rougeL_r, rougeL_f = [], [], []
        
        for item_dir in items_dir.iterdir():
            if not item_dir.is_dir():
                continue
            
            # Get the last JSON file
            json_files = list(item_dir.glob("*.json"))
            if not json_files:
                continue
            
            json_files.sort(key=lambda x: int(x.stem))
            last_file = json_files[-1]
            
            with open(last_file) as f:
                item_data = json.load(f)
            
            rouge_score = item_data.get('rouge_score', {})
            
            # rs-rouge1
            if 'rs-rouge1' in rouge_score:
                rouge1_p.append(rouge_score['rs-rouge1']['precision'])
                rouge1_r.append(rouge_score['rs-rouge1']['recall'])
                rouge1_f.append(rouge_score['rs-rouge1']['f1'])
            
            # rs-rouge2
            if 'rs-rouge2' in rouge_score:
                rouge2_p.append(rouge_score['rs-rouge2']['precision'])
                rouge2_r.append(rouge_score['rs-rouge2']['recall'])
                rouge2_f.append(rouge_score['rs-rouge2']['f1'])
            
            # rs-rougeL
            if 'rs-rougeL' in rouge_score:
                rougeL_p.append(rouge_score['rs-rougeL']['precision'])
                rougeL_r.append(rouge_score['rs-rougeL']['recall'])
                rougeL_f.append(rouge_score['rs-rougeL']['f1'])
        
        if rouge1_p:
            return {
                'rouge1': {
                    'precision': sum(rouge1_p) / len(rouge1_p),
                    'recall': sum(rouge1_r) / len(rouge1_r),
                    'f1': sum(rouge1_f) / len(rouge1_f)
                },
                'rouge2': {
                    'precision': sum(rouge2_p) / len(rouge2_p),
                    'recall': sum(rouge2_r) / len(rouge2_r),
                    'f1': sum(rouge2_f) / len(rouge2_f)
                },
                'rougeL': {
                    'precision': sum(rougeL_p) / len(rougeL_p),
                    'recall': sum(rougeL_r) / len(rougeL_r),
                    'f1': sum(rougeL_f) / len(rougeL_f)
                }
            }
    
    return None

def main():
    datasets = ["bmds", "true-detective"]
    methods = ["concat", "iterative"]
    
    results = {}
    
    for dataset in datasets:
        results[dataset] = {}
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")
        
        for method in methods:
            print(f"\n--- {method.capitalize()} ---")
            
            # Find summary hash
            summary_hash = find_summary_hash(dataset, method)
            if not summary_hash:
                print(f"  ERROR: Could not find summary collection for {dataset} {method}")
                continue
            
            print(f"  Summary collection: {summary_hash}")
            
            results[dataset][method] = {}
            
            # Get whodunit accuracy
            whodunit = get_whodunit_accuracy(dataset, summary_hash)
            results[dataset][method]['whodunit'] = whodunit
            print(f"  Whodunit: {whodunit:.4f}" if whodunit else "  Whodunit: NOT FOUND")
            
            # Get faithfulness
            faithfulness = get_faithfulness_score(dataset, summary_hash)
            results[dataset][method]['faithfulness'] = faithfulness
            print(f"  Faithfulness: {faithfulness:.4f}" if faithfulness else "  Faithfulness: NOT FOUND")
            
            # Get entity coverage recall
            ec_recall = get_entity_coverage_recall(dataset, summary_hash)
            results[dataset][method]['entity_coverage'] = ec_recall
            print(f"  Entity Coverage (Recall): {ec_recall:.4f}" if ec_recall else "  Entity Coverage: NOT FOUND")
            
            # Get ROUGE scores
            rouge = get_rouge_scores(dataset, summary_hash)
            results[dataset][method]['rouge'] = rouge
            if rouge:
                print(f"  ROUGE-1: P={rouge['rouge1']['precision']:.2f} / R={rouge['rouge1']['recall']:.2f} / F1={rouge['rouge1']['f1']:.2f}")
                print(f"  ROUGE-2: P={rouge['rouge2']['precision']:.2f} / R={rouge['rouge2']['recall']:.2f} / F1={rouge['rouge2']['f1']:.2f}")
                print(f"  ROUGE-L: P={rouge['rougeL']['precision']:.2f} / R={rouge['rougeL']['recall']:.2f} / F1={rouge['rougeL']['f1']:.2f}")
            else:
                print(f"  ROUGE: NOT FOUND")
    
    # Print LaTeX table
    print(f"\n\n{'='*60}")
    print("LATEX TABLE")
    print(f"{'='*60}\n")
    
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{Method} & \textbf{Extrinsic} & \textbf{Faithfulness} & \textbf{Entity Coverage} & \textbf{ROUGE-1} & \textbf{ROUGE-2} & \textbf{ROUGE-L} \\")
    print(r"\midrule")
    
    for dataset in datasets:
        display_name = "BMDS" if dataset == "bmds" else "True Detective"
        print(r"\multirow{2}{*}{" + display_name + r"}")
        
        for method in methods:
            data = results[dataset][method]
            
            # Format values
            whodunit = f"{data['whodunit']:.2f}" if data['whodunit'] else "--"
            faithfulness = f"{data['faithfulness']:.2f}" if data['faithfulness'] else "--"
            ec_recall = f"{data['entity_coverage']:.2f}" if data['entity_coverage'] else "--"
            
            if data['rouge']:
                rouge1 = f"{data['rouge']['rouge1']['precision']:.2f}/{data['rouge']['rouge1']['recall']:.2f}/{data['rouge']['rouge1']['f1']:.2f}"
                rouge2 = f"{data['rouge']['rouge2']['precision']:.2f}/{data['rouge']['rouge2']['recall']:.2f}/{data['rouge']['rouge2']['f1']:.2f}"
                rougeL = f"{data['rouge']['rougeL']['precision']:.2f}/{data['rouge']['rougeL']['recall']:.2f}/{data['rouge']['rougeL']['f1']:.2f}"
            else:
                rouge1 = rouge2 = rougeL = "--"
            
            method_name = "Concat" if method == "concat" else "Iterative"
            
            print(f"& {method_name} & {whodunit} & {faithfulness} & {ec_recall} & {rouge1} & {rouge2} & {rougeL} \\\\")
        
        print(r"\midrule")
    
    # Add Medical Transcripts with --
    print(r"\multirow{2}{*}{Medical Transcripts}")
    print(r"& Concat & -- & -- & -- & -- & -- & -- \\")
    print(r"& Iterative & -- & -- & -- & -- & -- & -- \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\caption{Comparison of Concatenate vs. Iterative summarization methods across datasets and metrics (without length constraints). Results show mean scores across all stories in each dataset.}")
    print(r"\label{tab:main_results}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()

