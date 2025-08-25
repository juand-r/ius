#!/usr/bin/env python3
"""
Analyze Name Variants in DetectiveQA Dataset

This script uses spaCy NER to extract all person names from each novel,
then analyzes them for potential variants and misspellings.
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from difflib import SequenceMatcher
import spacy
from typing import Dict, List, Set, Tuple

def load_spacy_model():
    """Load spaCy model for NER."""
    try:
        nlp = spacy.load('en_core_web_sm')
        return nlp
    except OSError:
        print("❌ Error: spaCy English model not found!")
        print("   Install it with: python -m spacy download en_core_web_sm")
        return None

def extract_person_entities(text: str, nlp) -> Set[str]:
    """Extract PERSON entities from text using spaCy NER."""
    if not nlp:
        return set()
    
    # Process text in chunks to handle large documents
    chunk_size = 1000000  # 1MB chunks
    entities = set()
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        doc = nlp(chunk)
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Clean the entity text
                name = ent.text.strip()
                # Filter out obviously non-name entities
                if is_likely_person_name(name):
                    entities.add(name)
    
    return entities

def is_likely_person_name(name: str) -> bool:
    """Filter out obvious non-person entities."""
    # Skip very short or very long names
    if len(name) < 2 or len(name) > 50:
        return False
    
    # Skip names that are mostly numbers or punctuation
    if not re.search(r'[A-Za-z]', name):
        return False
    
    # Skip common false positives
    false_positives = {
        'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'Who', 'What', 
        'Where', 'When', 'Why', 'How', 'Which', 'Yes', 'No', 'Oh', 'Ah',
        'Mr', 'Mrs', 'Ms', 'Dr', 'Professor', 'Sir', 'Lady', 'Lord', 
        'Captain', 'Colonel', 'Detective', 'Inspector', 'Officer',
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
    }
    
    if name in false_positives:
        return False
    
    # Skip names that look like titles or locations
    if name.upper() == name or name.lower() == name:
        return False
    
    return True

def calculate_similarity(name1: str, name2: str) -> float:
    """Calculate similarity ratio between two names."""
    return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

def find_similar_names(names: List[str], similarity_threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    """Find pairs of similar names that might be variants."""
    similar_pairs = []
    
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            similarity = calculate_similarity(name1, name2)
            if similarity >= similarity_threshold:
                similar_pairs.append((name1, name2, similarity))
    
    return sorted(similar_pairs, key=lambda x: x[2], reverse=True)

def analyze_novel(json_path: Path, nlp) -> Dict:
    """Analyze a single novel for person names."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {'error': f"Failed to load {json_path}: {e}"}
    
    novel_id = data['item_metadata']['item_id']
    title = data['documents'][0]['metadata'].get('title', 'Unknown')
    author = data['documents'][0]['metadata'].get('author', 'Unknown')
    content = data['documents'][0]['content']
    
    # Extract person entities
    person_entities = extract_person_entities(content, nlp)
    
    # Convert to sorted list for analysis
    sorted_names = sorted(list(person_entities))
    
    # Find similar name pairs
    similar_pairs = find_similar_names(sorted_names)
    
    # Count occurrences of each name in the text
    name_counts = {}
    content_lower = content.lower()
    for name in sorted_names:
        name_counts[name] = content_lower.count(name.lower())
    
    return {
        'novel_id': novel_id,
        'title': title,
        'author': author,
        'person_entities': sorted_names,
        'entity_count': len(sorted_names),
        'similar_pairs': similar_pairs,
        'name_counts': name_counts,
        'content_length': len(content)
    }

def main():
    """Main analysis function."""
    print("🔍 Analyzing DetectiveQA Dataset for Name Variants...")
    
    # Load spaCy model
    nlp = load_spacy_model()
    if not nlp:
        return
    
    # Find all JSON files
    items_dir = Path("datasets/detectiveqa/items")
    if not items_dir.exists():
        print(f"❌ Error: Directory {items_dir} not found!")
        return
    
    json_files = list(items_dir.glob("*.json"))
    print(f"📚 Found {len(json_files)} novels to analyze...")
    
    # Analyze each novel
    all_results = {}
    all_names = defaultdict(list)  # name -> list of novels it appears in
    
    for json_file in sorted(json_files):
        print(f"   Processing {json_file.stem}...")
        result = analyze_novel(json_file, nlp)
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
            continue
        
        all_results[result['novel_id']] = result
        
        # Track names across novels
        for name in result['person_entities']:
            all_names[name].append(result['novel_id'])
    
    # Save individual novel results
    output_dir = Path("name_analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📝 Saving results to {output_dir}/...")
    
    # Save detailed results for each novel
    for novel_id, result in all_results.items():
        novel_file = output_dir / f"novel_{novel_id}_names.json"
        with open(novel_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Create summary report
    summary_file = output_dir / "name_analysis_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("DETECTIVEQA NAME ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total novels analyzed: {len(all_results)}\n")
        f.write(f"Total unique person names found: {len(all_names)}\n\n")
        
        # Most common names across all novels
        f.write("TOP 50 MOST COMMON NAMES ACROSS ALL NOVELS:\n")
        f.write("-" * 50 + "\n")
        name_freq = {name: len(novels) for name, novels in all_names.items()}
        for name, freq in sorted(name_freq.items(), key=lambda x: x[1], reverse=True)[:50]:
            f.write(f"{name}: {freq} novels\n")
        
        f.write("\n\nNOVEL SUMMARIES:\n")
        f.write("-" * 50 + "\n")
        
        for novel_id in sorted(all_results.keys()):
            result = all_results[novel_id]
            f.write(f"\nNovel {novel_id}: {result['title']} by {result['author']}\n")
            f.write(f"  Person entities found: {result['entity_count']}\n")
            f.write(f"  Content length: {result['content_length']:,} chars\n")
            
            if result['similar_pairs']:
                f.write(f"  Similar name pairs found: {len(result['similar_pairs'])}\n")
                for name1, name2, similarity in result['similar_pairs'][:5]:  # Top 5
                    f.write(f"    {name1} ≈ {name2} ({similarity:.2f})\n")
    
    # Create potential variants report
    variants_file = output_dir / "potential_name_variants.txt"
    with open(variants_file, 'w', encoding='utf-8') as f:
        f.write("POTENTIAL NAME VARIANTS ACROSS ALL NOVELS\n")
        f.write("=" * 50 + "\n\n")
        
        # Find names that might be variants across different novels
        all_unique_names = list(all_names.keys())
        cross_novel_variants = find_similar_names(all_unique_names, similarity_threshold=0.8)
        
        f.write(f"Found {len(cross_novel_variants)} potential cross-novel variants:\n\n")
        
        for name1, name2, similarity in cross_novel_variants:
            novels1 = set(all_names[name1])
            novels2 = set(all_names[name2])
            f.write(f"{name1} ≈ {name2} (similarity: {similarity:.3f})\n")
            f.write(f"  {name1} appears in novels: {sorted(novels1)}\n")
            f.write(f"  {name2} appears in novels: {sorted(novels2)}\n")
            f.write(f"  Overlap: {sorted(novels1 & novels2)}\n\n")
    
    # Create high-frequency names report
    freq_file = output_dir / "high_frequency_names.txt"  
    with open(freq_file, 'w', encoding='utf-8') as f:
        f.write("HIGH FREQUENCY NAMES BY NOVEL\n")
        f.write("=" * 50 + "\n\n")
        
        for novel_id in sorted(all_results.keys()):
            result = all_results[novel_id]
            f.write(f"Novel {novel_id}: {result['title']}\n")
            f.write("-" * 30 + "\n")
            
            # Sort names by frequency in this novel
            sorted_names = sorted(result['name_counts'].items(), 
                                key=lambda x: x[1], reverse=True)
            
            f.write("Top 20 most frequent names:\n")
            for name, count in sorted_names[:20]:
                f.write(f"  {name}: {count} occurrences\n")
            f.write("\n")
    
    print(f"✅ Analysis complete!")
    print(f"   Results saved to: {output_dir}/")
    print(f"   - Individual novel files: novel_*_names.json")
    print(f"   - Summary: name_analysis_summary.txt")
    print(f"   - Potential variants: potential_name_variants.txt")
    print(f"   - High frequency names: high_frequency_names.txt")

if __name__ == "__main__":
    main()