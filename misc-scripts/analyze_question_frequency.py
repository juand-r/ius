#!/usr/bin/env python3
"""
Analyze question frequency in the detectiveqa dataset.
Collects all questions, counts frequency, and outputs sorted results.
"""

import json
import os
from collections import Counter
from pathlib import Path

def collect_all_questions():
    """Collect all questions from the detectiveqa dataset."""
    questions = []
    items_dir = Path("datasets/detectiveqa/items")
    
    if not items_dir.exists():
        print(f"❌ Directory not found: {items_dir}")
        return questions
    
    json_files = list(items_dir.glob("*.json"))
    print(f"📁 Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract questions from the structure
            novel_questions = data['documents'][0]['metadata']['questions']
            novel_id = data['item_metadata']['item_id']
            
            # Extract title and author from metadata
            metadata = data['documents'][0]['metadata']
            title = metadata.get('title', 'Unknown Title')
            author = metadata.get('author', 'Unknown Author')
            
            for q in novel_questions:
                question_text = q['question'].strip()
                
                # Get the answer text by matching the letter to the option
                answer_letter = q.get('answer', 'N/A')
                answer_text = 'N/A'
                if answer_letter and answer_letter in q.get('options', {}):
                    answer_text = q['options'][answer_letter].strip()
                
                questions.append({
                    'text': question_text,
                    'novel_id': novel_id,
                    'novel_title': title,
                    'novel_author': author,
                    'answer_position': q.get('answer_position', 0),
                    'answer': f"{answer_letter}: {answer_text}"
                })
                
        except Exception as e:
            print(f"❌ Error processing {json_file}: {e}")
    
    return questions

def analyze_and_write_results(questions):
    """Analyze question frequency and write results to file."""
    
    if not questions:
        print("❌ No questions found!")
        return
    
    print(f"📊 Collected {len(questions)} total questions")
    
    # Count frequency of each question text
    question_texts = [q['text'] for q in questions]
    frequency_counter = Counter(question_texts)
    
    # Sort by frequency (descending)
    sorted_questions = frequency_counter.most_common()
    
    print(f"📈 Found {len(sorted_questions)} unique questions")
    
    # Write results to file
    output_file = "detectiveqa_question_frequency_analysis.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DETECTIVEQA QUESTION FREQUENCY ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total questions: {len(questions)}\n")
        f.write(f"Unique questions: {len(sorted_questions)}\n")
        f.write(f"Duplicates found: {len(questions) - len(sorted_questions)}\n\n")
        
        # Summary of duplicates
        duplicates = [item for item in sorted_questions if item[1] > 1]
        if duplicates:
            f.write(f"DUPLICATE QUESTIONS ({len(duplicates)} found):\n")
            f.write("-" * 40 + "\n")
            for question, count in duplicates:
                f.write(f"[{count}x] {question}\n")
                
                # Show which novels contain this duplicate question
                novels_with_question = []
                for q in questions:
                    if q['text'] == question:
                        novels_with_question.append(f"Novel {q['novel_id']} (pos: {q['answer_position']})")
                
                f.write(f"       Found in: {', '.join(novels_with_question)}\n\n")
        else:
            f.write("✅ No duplicate questions found!\n\n")
        
        # All questions sorted by frequency
        f.write("ALL QUESTIONS (sorted by frequency):\n")
        f.write("-" * 40 + "\n")
        
        for i, (question, count) in enumerate(sorted_questions, 1):
            frequency_marker = f"[{count}x]" if count > 1 else "[1x] "
            
            # Find which novels contain this question
            novels_with_question = []
            for q in questions:
                if q['text'] == question:
                    novels_with_question.append(q['novel_id'])
            
            # Remove duplicates and sort
            novel_ids = sorted(list(set(novels_with_question)))
            novel_str = ", ".join(novel_ids)
            
            f.write(f"{i:3d}. {frequency_marker} (Novel {novel_str}) {question}\n")
        
        # Group questions by novel
        f.write("\n" + "=" * 50 + "\n")
        f.write("QUESTIONS GROUPED BY NOVEL (sorted by question count):\n")
        f.write("-" * 50 + "\n")
        
        # Create a dictionary to group questions by novel
        questions_by_novel = {}
        for q in questions:
            novel_id = q['novel_id']
            if novel_id not in questions_by_novel:
                questions_by_novel[novel_id] = {
                    'title': q['novel_title'],
                    'author': q['novel_author'],
                    'questions': []
                }
            questions_by_novel[novel_id]['questions'].append({
                'text': q['text'],
                'position': q['answer_position'],
                'answer': q['answer']  # This already contains "B: Mary Jones" format from above
            })
        
        # Sort novels by number of questions (descending)
        sorted_novels = sorted(questions_by_novel.items(), 
                             key=lambda x: len(x[1]['questions']), 
                             reverse=True)
        
        for novel_id, novel_data in sorted_novels:
            novel_questions = novel_data['questions']
            novel_title = novel_data['title']
            novel_author = novel_data['author']
            f.write(f"\nNovel {novel_id} - {novel_title} by {novel_author} ({len(novel_questions)} questions):\n")
            
            # Sort questions within novel by answer position
            sorted_novel_questions = sorted(novel_questions, key=lambda x: x['position'])
            
            for i, q in enumerate(sorted_novel_questions, 1):
                f.write(f"  {i}. (pos: {q['position']:4d}, ans: {q['answer']}) \"{q['text']}\"\n")
    
    print(f"✅ Results written to: {output_file}")
    
    # Print summary to console
    print(f"\n📈 SUMMARY:")
    print(f"   Total questions: {len(questions)}")
    print(f"   Unique questions: {len(sorted_questions)}")
    print(f"   Duplicates: {len(duplicates)} questions appear multiple times")
    
    if duplicates:
        print(f"\n🔍 TOP DUPLICATES:")
        for question, count in duplicates[:5]:  # Show top 5 duplicates
            print(f"   [{count}x] {question[:80]}{'...' if len(question) > 80 else ''}")

def main():
    print("🔍 Analyzing DetectiveQA question frequency...")
    
    questions = collect_all_questions()
    analyze_and_write_results(questions)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()