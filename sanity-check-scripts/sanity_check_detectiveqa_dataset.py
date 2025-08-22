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
    
    # Also check if it's a compound name and any part matches
    name_parts = name_lower.split()
    if len(name_parts) > 1:
        variations.extend(name_parts)
    
    for variation in variations:
        if variation in content_lower:
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
            'missing_names': [],
            'question_names': [],
            'answer_names': []
        }
    
    novel_id = data['item_metadata']['item_id']
    content = data['documents'][0]['content']
    metadata = data['documents'][0]['metadata']
    title = metadata.get('title', 'Unknown')
    author = metadata.get('author', 'Unknown')
    questions_data = metadata.get('questions', [])
    
    all_missing_names = []
    all_question_names = []
    all_answer_names = []
    
    for i, q_data in enumerate(questions_data):
        question = q_data['question']
        options = q_data.get('options', {})
        correct_answer_letter = q_data.get('answer', '')
        
        # Extract names from question
        question_names = extract_names_spacy(question, nlp)
        all_question_names.extend(question_names)
        
        # Extract names from ALL answer options (to catch question issues with wrong names)
        answer_names = []
        for option_key, option_text in options.items():
            option_names = extract_names_spacy(option_text, nlp)
            answer_names.extend(option_names)
        all_answer_names.extend(answer_names)
        
        # Check all names against content
        all_names = set(question_names + answer_names)
        for name in all_names:
            if not check_name_in_content(name, content):
                all_missing_names.append({
                    'name': name,
                    'question_num': i + 1,
                    'in_question': name in question_names,
                    'in_any_answer': name in answer_names,
                    'question_text': question,
                    'all_options': str(options)
                })
    
    return {
        'novel_id': novel_id,
        'title': title,
        'author': author,
        'missing_names': all_missing_names,
        'question_names': list(set(all_question_names)),
        'answer_names': list(set(all_answer_names)),
        'total_questions': len(questions_data),
        'error': None
    }

def main():
    """Main function to analyze all detectiveqa items."""
    print("🔍 Starting DetectiveQA Dataset Sanity Check...")
    print("   Checking if names in questions/ALL answer options appear in story content...")
    
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
    novels_with_issues = []
    
    # Analyze each file
    for json_file in sorted(json_files):
        print(f"   Analyzing {json_file.stem}...", end="")
        
        result = analyze_item(json_file, nlp)
        results.append(result)
        
        if result['error']:
            print(f" ❌ {result['error']}")
        elif result['missing_names']:
            print(f" ⚠️  {len(result['missing_names'])} missing names")
            novels_with_issues.append(result)
        else:
            print(" ✅")
    
    # Generate report
    output_file = "detectiveqa_sanity_check_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DETECTIVEQA DATASET SANITY CHECK REPORT\n")
        f.write("(Checking questions and ALL answer options)\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary statistics
        total_novels = len(results)
        novels_with_errors = len([r for r in results if r['error']])
        novels_with_missing_names = len(novels_with_issues)
        novels_clean = total_novels - novels_with_errors - novels_with_missing_names
        
        f.write(f"SUMMARY:\n")
        f.write(f"  Total novels analyzed: {total_novels}\n")
        f.write(f"  Novels with errors: {novels_with_errors}\n") 
        f.write(f"  Novels with missing names: {novels_with_missing_names}\n")
        f.write(f"  Clean novels: {novels_clean}\n\n")
        
        # Detailed issues
        if novels_with_issues:
            f.write("NOVELS WITH MISSING NAMES:\n")
            f.write("-" * 30 + "\n\n")
            
            for result in novels_with_issues:
                f.write(f"Novel {result['novel_id']} - {result['title']} by {result['author']}\n")
                f.write(f"  Total questions: {result['total_questions']}\n")
                f.write(f"  Missing names: {len(result['missing_names'])}\n")
                
                for missing in result['missing_names']:
                    location = []
                    if missing['in_question']:
                        location.append("question")
                    if missing['in_any_answer']:
                        location.append("answer options")
                    
                    f.write(f"    ❌ '{missing['name']}' (in {'/'.join(location)}) - Q{missing['question_num']}\n")
                    f.write(f"       Question: \"{missing['question_text']}\"\n")
                    f.write(f"       All Options: {missing['all_options']}\n")
                
                f.write("\n")
        
        # Error details
        error_results = [r for r in results if r['error']]
        if error_results:
            f.write("NOVELS WITH ERRORS:\n")
            f.write("-" * 20 + "\n\n")
            for result in error_results:
                f.write(f"Novel {result['novel_id']}: {result['error']}\n")
    
    print(f"\n📊 Analysis complete! Report saved to: {output_file}")
    print(f"   📈 {novels_clean}/{total_novels} novels are clean")
    print(f"   ⚠️  {novels_with_missing_names} novels have missing names in questions/answers")
    if novels_with_errors:
        print(f"   ❌ {novels_with_errors} novels had loading errors")
    
    # Show a sample of issues if any
    if novels_with_issues:
        print(f"\n🔍 Sample issues found (missing names in questions/answers):")
        for result in novels_with_issues[:3]:  # Show first 3
            print(f"   Novel {result['novel_id']} ({result['title']}):")
            for missing in result['missing_names'][:2]:  # Show first 2 missing names
                location = "question" if missing['in_question'] else "answer options"
                print(f"     - Missing '{missing['name']}' in {location}")
        if len(novels_with_issues) > 3:
            print(f"   ... and {len(novels_with_issues) - 3} more novels with issues")

if __name__ == "__main__":
    main()