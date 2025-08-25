#!/usr/bin/env python3
"""
Script to display questions and answers for novels with more than 1 question,
ordered by number of questions (most questions first).
"""

import json
from pathlib import Path

def show_novel_questions():
    """Display questions and answers for novels with >1 question."""
    
    items_dir = Path('datasets/detectiveqa/items')
    novels_data = []
    
    print("📚 DetectiveQA Novels - Questions & Answers")
    print("=" * 80)
    
    # Collect all novels with their question counts
    for json_file in items_dir.glob('*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        novel_id = json_file.stem
        metadata = data['documents'][0]['metadata']
        questions = metadata['questions']
        
        novels_data.append({
            'novel_id': novel_id,
            'title': metadata['title'],
            'author': metadata['author'],
            'question_count': len(questions),
            'questions': questions
        })
    
    # Sort by question count (descending) then by novel_id
    novels_data.sort(key=lambda x: (-x['question_count'], int(x['novel_id'])))
    
    print(f"Found {len(novels_data)} novels total")
    print()
    
    # Display each novel's questions and answers
    for i, novel in enumerate(novels_data, 1):
        print(f"📖 {i}. Novel {novel['novel_id']}: {novel['title']}")
        print(f"   Author: {novel['author']}")
        print(f"   Questions: {novel['question_count']}")
        print()
        
        for j, question_data in enumerate(novel['questions'], 1):
            print(f"   ❓ Question {j}:")
            print(f"      {question_data['question']}")
            print()
            
            # Show options
            if 'options' in question_data and question_data['options']:
                print(f"   📝 Options:")
                for option_key, option_text in question_data['options'].items():
                    marker = "✅" if option_key == question_data['answer'] else "  "
                    print(f"      {marker} {option_key}: {option_text}")
                print()
            
            print(f"   ✅ Answer: {question_data['answer']}")
            
            # Show reasoning if available
            if 'reasoning' in question_data and question_data['reasoning']:
                print(f"   💭 Reasoning:")
                for k, reason in enumerate(question_data['reasoning'], 1):
                    print(f"      {k}. {reason}")
                print()
            
            # Show answer position
            if 'answer_position' in question_data:
                print(f"   📍 Answer found at paragraph: {question_data['answer_position']}")
            
            if j < len(novel['questions']):  # Add separator between questions
                print("   " + "-" * 60)
            print()
        
        if i < len(novels_data):  # Add separator between novels
            print("=" * 80)
            print()
    
    # Summary statistics
    print("📊 SUMMARY:")
    total_questions = sum(novel['question_count'] for novel in novels_data)
    novels_with_multiple = [n for n in novels_data if n['question_count'] > 1]
    novels_with_one = [n for n in novels_data if n['question_count'] == 1]
    
    print(f"   Total novels: {len(novels_data)}")
    print(f"   Novels with >1 question: {len(novels_with_multiple)}")
    print(f"   Novels with 1 question: {len(novels_with_one)}")
    print(f"   Total questions: {total_questions}")
    print(f"   Average questions per novel: {total_questions/len(novels_data):.1f}")
    
    question_counts = {}
    for novel in novels_data:
        count = novel['question_count']
        question_counts[count] = question_counts.get(count, 0) + 1
    
    print(f"   Distribution:")
    for count in sorted(question_counts.keys(), reverse=True):
        novels_with_count = question_counts[count]
        print(f"     {count} questions: {novels_with_count} novels")

if __name__ == "__main__":
    # Redirect output to file
    import sys
    
    output_file = 'novel_questions_report.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        # Redirect stdout to file
        original_stdout = sys.stdout
        sys.stdout = f
        
        show_novel_questions()
        
        # Restore stdout
        sys.stdout = original_stdout
    
    print(f"📝 Report saved to: {output_file}")
    print(f"📊 Use 'cat {output_file}' or open in your editor to view the full report")