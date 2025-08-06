"""
First run python download_detectiveqa_novels.py to download the novels into data-source/detectiveqa_novels/\
Each novel is in a .txt file labels like 87-<...>.txt, i.e., the first number is the book id. The mapping of
book id to title and author (in English) is in the book_ids dictionary below.

This script will then do the following:

1. Filter out questions that are not in English.
2. Filter out questions that are not about who the murderer/culprit is. (show what fraction was kept)
3. Write out the data to datasets/detectiveqa/, 
   Note I want one .json file per book, and one .json file for the collection.
   IMPORTANT: Note there are 172 samples in the dataset, but only 62 novels. So we will have 62 json files, and the
   "questions" list will aggregate from multiple samples.

   Specifically we will have:
   - a collection.json file with keys "domain", "source", "created", "num_items", "total_documents", "description", "items" (list of book ids, which are just str(ints)).
   - in items/, write each book as a json file named <book_id>.json, with keys: (TODO: I think we should also include the novel_id, novel_title, novel_author, novel_text, time_cost, num_paragraphs)
     - "title-and-author": str (obtained from book_ids)
     - "text": str (obtained from data-source/detectiveqa_novels/<book_id>.txt, verbatim)
     - "questions": list of dicts with keys:
       - "question": str (obtained from the questions list in the sample)
       - "answer": str (obtained from the questions list in the sample)
       - "reasoning": List[str] (obtained from the questions list in the sample)
       - "answer_position": int (obtained from the questions list in the sample)
       - "clue_position": List[int] (obtained from the questions list in the sample)
       - "options": dict (mapping A, B, C, D to str) (obtained from the questions list in the sample)
       - "distraction": dict (mapping A, B, C, D to str) (obtained from the questions list in the sample)

Note structure of the ORIGINAL data we are loading from is:
# 172 shards in the data
dataset = load_dataset("Phospheneser/DetectiveQA", streaming=True)
samples = list(dataset.take(172)) #TODO is this correct or am I missing any data?
# Structure: each sample is a dict with keys:
# - novel_id: int
# - num_paragraphs: int (Let's not use this for now)
# - time_cost
# - questions: list of dicts with keys:
#   - question: str
#   - answer: str
#   - reasoning: List[str]
#   - answer_position: int
#   - clue_position: List[int]
#   - options: dict (mapping A, B, C, D to str)
#   - distraction: dict (mapping A, B, C, D to str); TODO not sure what this is for.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from datasets import load_dataset
from langdetect import detect
from difflib import SequenceMatcher
import re

# Note: my attempt to get these in English may not be completely accurate
# https://huggingface.co/datasets/Phospheneser/DetectiveQA/tree/main/novel_data_en
book_ids = dict([
    (100, "Alphabet Puzzle - Seiichiro Oyama"),
    (103, "The Murder on the Links - Agatha Christie"),
    (104, "The Kidnapped Prime Minister - Agatha Christie"),
    (105, "The Mystery of the Blue Train - Agatha Christie"),
    (106, "Peril at End House - Agatha Christie"),
    (107, "Death in the Clouds - Agatha Christie"),
    (108, "Three Act Tragedy - Agatha Christie"),
    (109, "Cards on the Table - Agatha Christie"),
    (110, "Murder in Mesopotamia - Agatha Christie"),
    (114, "Hercule Poirot's Christmas - Agatha Christie"),
    (116, "Sad Cypress - Agatha Christie"),
    (117, "Evil Under the Sun - Agatha Christie"),
    (118, "Taken at the Flood - Agatha Christie"),
    (120, "After the Funeral - Agatha Christie"),
    (121, "Hickory Dickory Dock - Agatha Christie"),
    (124, "The Clocks - Agatha Christie"),
    (126, "Halloween Party - Agatha Christie"),
    (127, "Elephants Can Remember - Agatha Christie"),
    (128, "Curtain: Poirot's Last Case - Agatha Christie"),
    (130, "The Murder at the Vicarage - Agatha Christie"),
    (132, "The Body in the Library - Agatha Christie"),
    (133, "By the Pricking of My Thumbs - Agatha Christie"),
    (134, "A Murder is Announced - Agatha Christie"),
    (136, "4.50 from Paddington - Agatha Christie"),
    (137, "The Mirror Crack'd from Side to Side - Agatha Christie"),
    (138, "A Caribbean Mystery - Agatha Christie"),
    (140, "Sleeping Murder - Agatha Christie"),
    (142, "The Secret of Chimneys - Agatha Christie"),
    (144, "The Seven Dials Mystery - Agatha Christie"),
    (145, "The Witness for the Prosecution - Agatha Christie"),
    (149, "Sparkling Cyanide - Agatha Christie"),
    (15, "The Spider Man - Edogawa Ranpo"),
    (150, "Crooked House - Agatha Christie"),
    (151, "Three Blind Mice - Agatha Christie"),
    (16, "The Magician - Edogawa Ranpo"),
    (198, "The ABC Murders - Agatha Christie"),
    (203, "One, Two, Buckle My Shoe - Agatha Christie"),
    (209, "Mrs McGinty's Dead - Agatha Christie"),
    (219, "The Monogram Murders - Sophie Hannah"),
    (241, "Why Didn't They Ask Evans? - Agatha Christie"),
    (25, "The Roman Hat Mystery - Ellery Queen"),
    (252, "Murder by Command - Agatha Christie"),
    (26, "The Greek Coffin Mystery - Ellery Queen"),
    (27, "The Egyptian Cross Mystery - Ellery Queen"),
    (28, "The Tragedy of X - Ellery Queen"),
    (29, "The Tragedy of Y - Ellery Queen"),
    (30, "The Tragedy of Z - Ellery Queen"),
    (31, "The Three Coffins - John Dickson Carr"),
    (33, "The Crooked Hinge - John Dickson Carr"),
    (40, "The Klein Bottle - Okajima Futari"),
    (53, "The Dutch Shoe Mystery - Ellery Queen"),
    (56, "Drury Lane's Last Case - Ellery Queen"),
    (79, "Winged Darkness -- Maya Yutaka"),
    (81, "The One-Eyed Girl -- Yutaka Maya"),
    (82, "The Disease of Murder - Takemaru Abiko"),
    (83, "The Two-Headed Devil - Arisu Arisugawa"),
    (84, "The Decagon House Murders - Yukito Ayatsuji"),
    (87, "Everything Becomes F - Hiroshi Mori"),
    (90, "The Hyperbolic Murder - Kyotaro Nishimura"),
    (93, "The Red Museum - Seiichiro Oyama"),
    (97, "The Scottish Game - Yasuhiko Nishizawa"),
    (99, "The Night She Died - Yasuhiko Nishizawa")
])


def is_english(text):
    """Check if text is in English using langdetect."""
    try:
        return detect(text) == 'en'
    except:
        # If detection fails, assume it's not English
        return False


def is_murderer_question(question):
    """Check if question is about who the murderer/culprit is."""
    question_lower = question.lower()
    murderer_keywords = [
        'murderer', 'killer', 'culprit', 'perpetrator', 'criminal',
        'who killed', 'who murdered', 'who committed', 'who did it',
        'who is responsible', 'who was the', 'guilty person', 'suspect'
    ]
    return any(keyword in question_lower for keyword in murderer_keywords)


def similarity(a, b):
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def main():
    """Main ingestion process."""
    print("🔍 Starting DetectiveQA ingestion...")
    
    # Load dataset
    print("📥 Loading dataset...")
    dataset = load_dataset("Phospheneser/DetectiveQA", streaming=True)
    samples = list(dataset['train'].take(172))  # All 172 samples
    print(f"✅ Loaded {len(samples)} samples")
    
    # Create output directories
    output_dir = Path("datasets/detectiveqa")
    items_dir = output_dir / "items"
    output_dir.mkdir(exist_ok=True)
    items_dir.mkdir(exist_ok=True)
    
    # Open log files for inspection
    murder_questions_file = open("murder_questions.txt", "w", encoding="utf-8")
    non_murder_questions_file = open("non_murder_questions.txt", "w", encoding="utf-8")
    warnings_file = open("ingestion_warnings.txt", "w", encoding="utf-8")
    
    # Group samples by novel_id
    print("📚 Grouping samples by novel...")
    novels_data = defaultdict(list)
    for sample in samples:
        novels_data[sample['novel_id']].append(sample)
    
    print(f"📊 Found {len(novels_data)} unique novels")
    
    # Process each novel
    total_questions = 0
    english_questions = 0
    murder_questions = 0
    processed_novels = []
    
    for novel_id, novel_samples in novels_data.items():
        print(f"\n📖 Processing novel {novel_id}...")
        
        # Check if we have the book mapping
        if novel_id not in book_ids:
            warnings_file.write(f"WARNING: Novel ID {novel_id} not found in book_ids mapping\n")
            continue
        
        # Load novel text
        novel_file = Path(f"data-source/detectiveqa/{novel_id}-*.txt")
        novel_files = list(Path("data-source/detectiveqa").glob(f"{novel_id}-*.txt"))
        
        if not novel_files:
            warnings_file.write(f"WARNING: Novel text file not found for ID {novel_id}\n")
            continue
        
        novel_text_file = novel_files[0]
        try:
            with open(novel_text_file, 'r', encoding='utf-8') as f:
                novel_text = f.read()
        except Exception as e:
            warnings_file.write(f"ERROR: Could not read novel file {novel_text_file}: {e}\n")
            continue
        
        # Aggregate questions from all samples for this novel
        all_questions = []
        seen_questions = []  # For duplicate detection
        
        for sample in novel_samples:
            for question_data in sample['questions']:
                total_questions += 1
                
                # Check if English
                if not is_english(question_data['question']):
                    continue
                english_questions += 1
                
                # Check for duplicates/near-duplicates
                question_text = question_data['question']
                for seen_q in seen_questions:
                    if similarity(question_text, seen_q) > 0.8:  # 80% similarity threshold
                        warnings_file.write(f"WARNING: Potential duplicate question in novel {novel_id}:\n")
                        warnings_file.write(f"  Existing: {seen_q}\n")
                        warnings_file.write(f"  New: {question_text}\n\n")
                        break
                seen_questions.append(question_text)
                
                # Check if it's about murderer/culprit
                if is_murderer_question(question_data['question']):
                    murder_questions += 1
                    murder_questions_file.write(f"Novel {novel_id}: {question_data['question']}\n")
                    
                    # Add to filtered questions
                    all_questions.append({
                        "question": question_data['question'],
                        "answer": question_data['answer'],
                        "reasoning": question_data['reasoning'],
                        "answer_position": question_data['answer_position'],
                        "clue_position": question_data['clue_position'],
                        "options": question_data['options'],
                        "distraction": question_data['distraction']
                    })
                else:
                    non_murder_questions_file.write(f"Novel {novel_id}: {question_data['question']}\n")
        
        # Skip novels with no qualifying questions
        if not all_questions:
            print(f"  ⏭️  Skipping novel {novel_id} (no qualifying questions)")
            continue
        
        print(f"  ✅ Novel {novel_id}: {len(all_questions)} qualifying questions")
        
        # Create item JSON following README structure
        item_data = {
            "item_metadata": {
                "item_id": str(novel_id),
                "num_documents": 1
            },
            "documents": [
                {
                    "doc_id": f"novel_{novel_id}",
                    "content": novel_text,
                    "metadata": {
                        "title": book_ids[novel_id].split(" - ")[0],
                        "author": book_ids[novel_id].split(" - ")[1],
                        "novel_id": novel_id,
                        "num_paragraphs": novel_samples[0]['num_paragraphs'],
                        "time_cost": novel_samples[0]['time_cost'],
                        "questions": all_questions
                    }
                }
            ]
        }
        
        # Write item file
        item_file = items_dir / f"{novel_id}.json"
        with open(item_file, 'w', encoding='utf-8') as f:
            json.dump(item_data, f, indent=2, ensure_ascii=False)
        
        processed_novels.append(str(novel_id))
    
    # Create collection.json
    collection_data = {
        "domain": "detective_stories",
        "source": "DetectiveQA dataset - Phospheneser/DetectiveQA from Hugging Face",
        "created": datetime.now().isoformat(),
        "num_items": len(processed_novels),
        "total_documents": len(processed_novels),
        "description": "Detective novels with questions about murderer/culprit identification, filtered from DetectiveQA dataset",
        "items": processed_novels
    }
    
    collection_file = output_dir / "collection.json"
    with open(collection_file, 'w', encoding='utf-8') as f:
        json.dump(collection_data, f, indent=2, ensure_ascii=False)
    
    # Close log files
    murder_questions_file.close()
    non_murder_questions_file.close()
    warnings_file.close()
    
    # Print summary
    print(f"\n📊 INGESTION SUMMARY:")
    print(f"Total questions processed: {total_questions}")
    print(f"English questions: {english_questions} ({english_questions/total_questions*100:.1f}%)")
    print(f"Murder/culprit questions: {murder_questions} ({murder_questions/english_questions*100:.1f}% of English)")
    print(f"Novels processed: {len(processed_novels)}")
    print(f"Output directory: {output_dir}")
    print(f"\n📝 Log files created:")
    print(f"  - murder_questions.txt: {murder_questions} questions about murderers/culprits")
    print(f"  - non_murder_questions.txt: {english_questions - murder_questions} other English questions")
    print(f"  - ingestion_warnings.txt: Warnings and errors during processing")


if __name__ == "__main__":
    main()