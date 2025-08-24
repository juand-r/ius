#!/usr/bin/env python3
"""
Script to verify detectiveqa data using GPT-4o with web search.
Extracts ground truth and compares with GPT-4o web search predictions.
"""

import json
import csv
import glob
from pathlib import Path
from typing import Dict, Any
import re

from ius.utils import call_llm

def extract_detectiveqa_info(file_path: str) -> Dict[str, Any]:
    """Extract title, author, question, suspects, and correct answer from detectiveqa item."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    item_id = Path(file_path).stem
    metadata = data['documents'][0]['metadata']
    original_metadata = metadata['original_metadata']
    questions = original_metadata['questions']
    question_data = questions[0]
    
    return {
        'item_id': item_id,
        'title': original_metadata.get('title', 'Unknown'),
        'author': original_metadata.get('author', 'Unknown'),
        'question': question_data['question'],
        'suspects': question_data['options'],
        'answer_letter': question_data['answer'],
        'correct_culprit': question_data['options'][question_data['answer']]
    }

def call_gpt5_victims(title: str, author: str) -> Dict[str, Any]:
    """Ask GPT-4o with web search to find victims in the novel."""
    result = call_llm(
        text="",
        system_and_user_prompt={
            "system": "You are a helpful assistant that can search the web for information about novels and provide accurate answers.",
            "user": "In {title} by {author}, who are all the victims? Search the web and return a comprehensive list of all the victims in the novel. Just respond with a list of names, nothing else"
        },
        template_vars={
            "title": title,
            "author": author
        },
        model="gpt-4o-search-preview",
        max_completion_tokens=5000
    )
    
    return result

def call_gpt5_culprit(title: str, author: str, question: str) -> Dict[str, Any]:
    """Ask GPT-4o with web search to answer the whodunit question."""
    result = call_llm(
        text="",
        system_and_user_prompt={
            "system": "You are a helpful assistant that can search the web for information about novels and provide accurate answers about mystery plots.",
            "user": "In {title} by {author}, {question} Search the web, explain what you found, and then tell me the final answer name like ANSWER: \"culprit_name\". If you cannot find information about this specific character or scenario, respond with ANSWER: \"NOT_FOUND\""
        },
        template_vars={
            "title": title,
            "author": author,
            "question": question
        },
        model="gpt-4o-search-preview", 
        max_completion_tokens=5000
    )
    
    return result

def extract_answer_from_response(response: str) -> str:
    """Extract the culprit name after 'ANSWER:' from GPT-5 response."""
    # Look for ANSWER: "name" or ANSWER: name
    match = re.search(r'ANSWER:\s*["\']?([^"\'\\n]+?)["\']?(?:\\n|$)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for just ANSWER: followed by text
    match = re.search(r'ANSWER:\s*(.+)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return "PARSE_ERROR"

def save_to_csv(data: Dict[str, Any], csv_file: str = "detectiveqa_verification.csv"):
    """Save data to CSV file (append mode for incremental progress)."""
    file_exists = Path(csv_file).exists()
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['item_id', 'title', 'author', 'question', 'gpt4o_web_search_victims', 'ground_truth_culprit', 'gpt4o_web_search_predicted_culprit']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'item_id': data['item_id'],
            'title': data['title'],
            'author': data['author'],
            'question': data['question'],
            'gpt4o_web_search_victims': data['gpt4o_web_search_victims_response'],
            'ground_truth_culprit': data['ground_truth_culprit'],
            'gpt4o_web_search_predicted_culprit': data['gpt4o_web_search_predicted_culprit']
        })

def save_to_json(data: Dict[str, Any], json_file: str = "detectiveqa_verification_detailed.json"):
    """Save detailed data to JSON file (append mode for incremental progress)."""
    # Load existing data if file exists
    if Path(json_file).exists():
        with open(json_file, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    
    # Append new data
    existing_data.append(data)
    
    # Save back to file
    with open(json_file, 'w') as f:
        json.dump(existing_data, f, indent=2)

def main():
    """Main function to verify detectiveqa data."""
    print("🔍 Starting DetectiveQA data verification with GPT-4o Web Search...")
    
    # Find all detectiveqa items (use just one directory to avoid duplicates)
    files = glob.glob('outputs/chunks/detectiveqa_fixed_size_8000/items/*.json')
    print(f"Found {len(files)} detectiveqa items")
    
    if not files:
        print("❌ No detectiveqa files found!")
        return
    
    # Process all items
    for i, file_path in enumerate(sorted(files)):
        print(f"\\n📖 Processing item {i+1}: {Path(file_path).stem}")
        
        try:
            # Extract basic info
            info = extract_detectiveqa_info(file_path)
            print(f"Title: {info['title']}")
            print(f"Author: {info['author']}")
            print(f"Question: {info['question']}")
            print(f"Suspects: {list(info['suspects'].values())}")
            print(f"Correct Answer: {info['answer_letter']} -> {info['correct_culprit']}")
            
            # Call GPT-4o with web search for victims
            print("\\n🤖 Asking GPT-4o (with web search) about victims...")
            victims_result = call_gpt5_victims(info['title'], info['author'])
            victims_response = victims_result.get('response', 'ERROR')
            print(f"GPT-4o Web Search Victims Response: {victims_response[:200]}...")
            
            # Call GPT-4o with web search for culprit
            print("\\n🕵️ Asking GPT-4o (with web search) about culprit...")
            culprit_result = call_gpt5_culprit(info['title'], info['author'], info['question'])
            culprit_response = culprit_result.get('response', 'ERROR')
            print(f"GPT-4o Web Search Culprit Response: {culprit_response[:200]}...")
            
            # Extract predicted culprit
            predicted_culprit = extract_answer_from_response(culprit_response)
            print(f"\\n📊 Results:")
            print(f"Ground Truth: {info['correct_culprit']}")
            print(f"GPT-4o Web Search Predicted: {predicted_culprit}")
            
            # Prepare data for saving
            verification_data = {
                'item_id': info['item_id'],
                'title': info['title'],
                'author': info['author'],
                'question': info['question'],
                'suspects': info['suspects'],
                'ground_truth_culprit': info['correct_culprit'],
                'gpt4o_web_search_victims_response': victims_response,
                'gpt4o_web_search_culprit_response': culprit_response,
                'gpt4o_web_search_predicted_culprit': predicted_culprit,
                'victims_llm_result': victims_result,
                'culprit_llm_result': culprit_result
            }
            
            # Save to files
            save_to_csv(verification_data)
            save_to_json(verification_data)
            
            print(f"\\n💾 Saved results to CSV and JSON files")
            #print("🛑 Stopping after first item for debugging")
            #break
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()