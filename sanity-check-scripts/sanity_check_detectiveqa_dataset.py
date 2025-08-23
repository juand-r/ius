#!/usr/bin/env python3
"""
Sanity check script for DetectiveQA dataset.
Extracts names from questions and answer choices and verifies they appear in the story content.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import spacy

def load_spacy_model():
    """Load spaCy model for name extraction."""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        print("⚠️  spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
        print("    Falling back to regex-based name extraction.")
        return None

def extract_names_regex(text):
    """Extract potential names using regex (fallback method)."""
    # Find sequences of capitalized words that could be names
    # Pattern: One or more capitalized words, possibly with apostrophes, hyphens
    name_pattern = r'\b[A-Z][a-zA-Z]*(?:\'s?|\-[A-Z][a-zA-Z]*)*(?:\s+[A-Z][a-zA-Z]*(?:\'s?|\-[A-Z][a-zA-Z]*)*)*\b'
    names = re.findall(name_pattern, text)
    
    # Filter out common false positives
    false_positives = {
        'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'Who', 'What', 'Where', 
        'When', 'Why', 'How', 'Which', 'Mr', 'Mrs', 'Ms', 'Dr', 'Professor', 'Sir',
        'Lady', 'Lord', 'Captain', 'Colonel', 'Detective', 'Inspector', 'Officer',
        'According', 'Based', 'Following', 'Question', 'Answer', 'Text', 'Story',
        'Novel', 'Book', 'Chapter', 'Scene', 'Case', 'Murder', 'Killer', 'Murderer',
        'Victim', 'Suspect', 'Culprit', 'Criminal', 'Person', 'People', 'Someone',
        'Everyone', 'Anyone', 'Nobody', 'Somebody', 'Something', 'Nothing', 'Everything'
    }
    
    # Filter and clean names
    filtered_names = []
    for name in names:
        # Remove possessive 's
        clean_name = re.sub(r"'s$", "", name)
        if clean_name not in false_positives and len(clean_name) > 1:
            filtered_names.append(clean_name)
    
    return list(set(filtered_names))

def extract_names_spacy(text, nlp):
    """Extract person names using spaCy NER."""
    if nlp is None:
        return extract_names_regex(text)
    
    doc = nlp(text)
    names = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Clean the name (remove possessive, etc.)
            clean_name = re.sub(r"'s$", "", ent.text.strip())
            names.append(clean_name)
    
    return list(set(names))

def check_name_in_content(name, content):
    """Check if a name appears in the content with various forms."""
    # Convert content to lowercase for case-insensitive search
    content_lower = content.lower()
    name_lower = name.lower()
    
    # Check exact match
    if name_lower in content_lower:
        return True
    
    # Check with common variations
    variations = [
        name_lower,
        name_lower + "'s",  # possessive
        name_lower + "s",   # plural or possessive without apostrophe
        "mr. " + name_lower,
        "mrs. " + name_lower,
        "miss " + name_lower,
        "dr. " + name_lower,
    ]
    
    for variation in variations:
        if variation in content_lower:
            return True
    
    # For compound names, check if ALL parts are present individually
    name_parts = name_lower.split()
    if len(name_parts) > 1:
        # All parts must be found (but not necessarily together)
        all_parts_found = True
        for part in name_parts:
            if part not in content_lower:
                all_parts_found = False
                break
        if all_parts_found:
            return True
    
    return False

def analyze_item(item_path, nlp):
    """Analyze a single detectiveqa item for name consistency."""
    try:
        with open(item_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {
            'error': f"Failed to load {item_path}: {e}",
            'novel_id': str(item_path.stem),
            'missing_from_questions': [],
            'missing_from_correct_answers': [],
            'question_names': [],
            'correct_answer_names': []
        }
    
    novel_id = data['item_metadata']['item_id']
    content = data['documents'][0]['content']
    metadata = data['documents'][0]['metadata']
    title = metadata.get('title', 'Unknown')
    author = metadata.get('author', 'Unknown')
    questions_data = metadata.get('questions', [])
    
    missing_from_questions = []
    missing_from_correct_answers = []
    all_question_names = []
    all_correct_answer_names = []
    
    for i, q_data in enumerate(questions_data):
        question = q_data['question']
        options = q_data.get('options', {})
        correct_answer_letter = q_data.get('answer', '')
        
        # Extract names from question
        question_names = extract_names_spacy(question, nlp)
        all_question_names.extend(question_names)
        
        # Extract names from CORRECT answer only
        correct_answer_names = []
        if correct_answer_letter and correct_answer_letter in options:
            correct_answer_text = options[correct_answer_letter]
            correct_answer_names = extract_names_spacy(correct_answer_text, nlp)
        all_correct_answer_names.extend(correct_answer_names)
        
        # Check question names against content
        for name in question_names:
            if not check_name_in_content(name, content):
                missing_from_questions.append({
                    'name': name,
                    'question_num': i + 1,
                    'question_text': question,
                    'correct_answer_letter': correct_answer_letter,
                    'correct_answer_text': options.get(correct_answer_letter, 'N/A')
                })
        
        # Check correct answer names against content
        for name in correct_answer_names:
            if not check_name_in_content(name, content):
                missing_from_correct_answers.append({
                    'name': name,
                    'question_num': i + 1,
                    'question_text': question,
                    'correct_answer_letter': correct_answer_letter,
                    'correct_answer_text': options.get(correct_answer_letter, 'N/A')
                })
    
    return {
        'novel_id': novel_id,
        'title': title,
        'author': author,
        'missing_from_questions': missing_from_questions,
        'missing_from_correct_answers': missing_from_correct_answers,
        'question_names': list(set(all_question_names)),
        'correct_answer_names': list(set(all_correct_answer_names)),
        'total_questions': len(questions_data),
        'error': None
    }

def main():
    """Main function to analyze all detectiveqa items."""
    print("🔍 Starting DetectiveQA Dataset Sanity Check...")
    print("   Checking if names in questions and CORRECT answers appear in story content...")
    
    # Load spaCy model
    nlp = load_spacy_model()
    
    # Find all JSON files
    items_dir = Path("datasets/detectiveqa/items")
    if not items_dir.exists():
        print(f"❌ Error: Directory {items_dir} not found!")
        return
    
    json_files = list(items_dir.glob("*.json"))
    print(f"📚 Found {len(json_files)} novels to analyze...")
    
    results = []
    novels_with_question_issues = []
    novels_with_answer_issues = []
    
    # Analyze each file
    for json_file in sorted(json_files):
        print(f"   Analyzing {json_file.stem}...", end="")
        
        result = analyze_item(json_file, nlp)
        results.append(result)
        
        if result['error']:
            print(f" ❌ {result['error']}")
        else:
            issues = []
            if result['missing_from_questions']:
                issues.append(f"{len(result['missing_from_questions'])} question")
                novels_with_question_issues.append(result)
            if result['missing_from_correct_answers']:
                issues.append(f"{len(result['missing_from_correct_answers'])} answer")
                novels_with_answer_issues.append(result)
            
            if issues:
                print(f" ⚠️  {', '.join(issues)} missing names")
            else:
                print(" ✅")
    
    # Generate report
    output_file = "detectiveqa_sanity_check_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DETECTIVEQA DATASET SANITY CHECK REPORT\n")
        f.write("(Checking questions and CORRECT answers separately)\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary statistics
        total_novels = len(results)
        novels_with_errors = len([r for r in results if r['error']])
        novels_with_question_issues_count = len(novels_with_question_issues)
        novels_with_answer_issues_count = len(novels_with_answer_issues)
        novels_with_any_issues = len(set([r['novel_id'] for r in novels_with_question_issues + novels_with_answer_issues]))
        novels_clean = total_novels - novels_with_errors - novels_with_any_issues
        
        f.write(f"SUMMARY:\n")
        f.write(f"  Total novels analyzed: {total_novels}\n")
        f.write(f"  Novels with loading errors: {novels_with_errors}\n")
        f.write(f"  Novels with missing names in QUESTIONS: {novels_with_question_issues_count}\n")
        f.write(f"  Novels with missing names in CORRECT ANSWERS: {novels_with_answer_issues_count}\n")
        f.write(f"  Novels with ANY issues: {novels_with_any_issues}\n")
        f.write(f"  Clean novels: {novels_clean}\n\n")
        
        # Total counts
        total_question_issues = sum(len(r['missing_from_questions']) for r in results)
        total_answer_issues = sum(len(r['missing_from_correct_answers']) for r in results)
        f.write(f"TOTAL ISSUE COUNTS:\n")
        f.write(f"  Missing names from questions: {total_question_issues}\n")
        f.write(f"  Missing names from correct answers: {total_answer_issues}\n\n")
        
        # Issues in QUESTIONS
        if novels_with_question_issues:
            f.write("1. MISSING NAMES IN QUESTIONS:\n")
            f.write("-" * 40 + "\n\n")
            
            for result in novels_with_question_issues:
                if result['missing_from_questions']:
                    f.write(f"Novel {result['novel_id']} - {result['title']} by {result['author']}\n")
                    f.write(f"  Missing from questions: {len(result['missing_from_questions'])}\n")
                    
                    for missing in result['missing_from_questions']:
                        f.write(f"    ❌ '{missing['name']}' - Q{missing['question_num']}\n")
                        f.write(f"       Question: \"{missing['question_text']}\"\n")
                        f.write(f"       Correct Answer ({missing['correct_answer_letter']}): \"{missing['correct_answer_text']}\"\n")
                    f.write("\n")
        
        # Issues in CORRECT ANSWERS
        if novels_with_answer_issues:
            f.write("2. MISSING NAMES IN CORRECT ANSWERS:\n")
            f.write("-" * 40 + "\n\n")
            
            for result in novels_with_answer_issues:
                if result['missing_from_correct_answers']:
                    f.write(f"Novel {result['novel_id']} - {result['title']} by {result['author']}\n")
                    f.write(f"  Missing from correct answers: {len(result['missing_from_correct_answers'])}\n")
                    
                    for missing in result['missing_from_correct_answers']:
                        f.write(f"    ❌ '{missing['name']}' - Q{missing['question_num']}\n")
                        f.write(f"       Question: \"{missing['question_text']}\"\n")
                        f.write(f"       Correct Answer ({missing['correct_answer_letter']}): \"{missing['correct_answer_text']}\"\n")
                    f.write("\n")
        
        # Error details
        error_results = [r for r in results if r['error']]
        if error_results:
            f.write("3. NOVELS WITH LOADING ERRORS:\n")
            f.write("-" * 30 + "\n\n")
            for result in error_results:
                f.write(f"Novel {result['novel_id']}: {result['error']}\n")
    
    print(f"\n📊 Analysis complete! Report saved to: {output_file}")
    print(f"   📈 {novels_clean}/{total_novels} novels are clean")
    print(f"   ❓ {novels_with_question_issues_count} novels have missing names in QUESTIONS")
    print(f"   ✅ {novels_with_answer_issues_count} novels have missing names in CORRECT ANSWERS")
    if novels_with_errors:
        print(f"   ❌ {novels_with_errors} novels had loading errors")
    
    # Show samples of both issue types
    if novels_with_question_issues:
        print(f"\n🔍 Sample QUESTION issues:")
        for result in novels_with_question_issues[:2]:  # Show first 2
            print(f"   Novel {result['novel_id']} ({result['title']}):")
            for missing in result['missing_from_questions'][:2]:  # Show first 2 missing names
                print(f"     - Missing '{missing['name']}' from question {missing['question_num']}")
    
    if novels_with_answer_issues:
        print(f"\n🔍 Sample CORRECT ANSWER issues:")
        for result in novels_with_answer_issues[:2]:  # Show first 2
            print(f"   Novel {result['novel_id']} ({result['title']}):")
            for missing in result['missing_from_correct_answers'][:2]:  # Show first 2 missing names
                print(f"     - Missing '{missing['name']}' from answer {missing['correct_answer_letter']}")

if __name__ == "__main__":
    main()