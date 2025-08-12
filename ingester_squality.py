#!/usr/bin/env python3
"""
SQuALITY Dataset Ingester

This script ingests the SQuALITY dataset from Hugging Face (pszemraj/SQuALITY-v1.3) 
and converts it to the standardized IUS dataset format.

SQuALITY contains reading comprehension stories with questions and human-written responses.
Each story has 5 questions with multiple human responses per question.

The script will:
1. Load all splits (train, validation, test) from SQuALITY
2. Use passage_id as the unique item identifier
3. Convert to the standard IUS format with collection.json and items/{passage_id}.json
4. Include all questions and responses in metadata
5. Add split information to metadata
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from datasets import load_dataset


metadata_map = {
  "23942": {"title": "Unborn Tomorrow", "author": "Mack Reynolds"},
  "24192": {"title": "The First One", "author": "Herbert D. Kastle"},
  "24949": {"title": "Control Group", "author": "Roger Dee"},
  "24966": {"title": "Survival Tactics", "author": "Al Sevcik"},
  "29159": {"title": "Acid Bath", "author": "Vaseleos Garson"},
  "29170": {"title": "The Hoofer", "author": "Walter M. Miller, Jr."},
  "29193": {"title": "Dream Town", "author": "Henry Slesar"},
  "30004": {"title": "A Bottle of Old Wine", "author": "Richard O. Lewis"},
  "30029": {"title": "Lost in Translation", "author": "Larry M. Harris"},
  "30062": {"title": "The Plague", "author": "Teddy Keller"},
  "32667": {"title": "The Holes and John Smith", "author": "Edward W. Ludwig"},
  "32744": {"title": "The Valley", "author": "Richard Stockham"},
  "32890": {"title": "Home is Where You Left It", "author": "Adam Chase"},
  "40968": {"title": "Desire No More", "author": "Algis Budrys"},
  "41562": {"title": "The Hanging Stranger", "author": "Philip K. Dick"},
  "47841": {"title": "The Haunted Fountain", "author": "Margaret Sutton"},
  "48513": {"title": "His Master's Voice", "author": "Randall Garrett"},
  "49165": {"title": "Brightside Crossing", "author": "Alan E. Nourse"},
  "49838": {"title": "Jack of No Trades", "author": "Evelyn E. Smith"},
  "49897": {"title": "The Gravity Business", "author": "James E. Gunn"},
  "49901": {"title": "The Snare", "author": "Richard R. Smith"},
  "50441": {"title": "Master of Life and Death", "author": "Robert Silverberg"},
  "50774": {"title": "Contagion", "author": "Katherine MacLean"},
  "50802": {"title": "A City Near Centaurus", "author": "Bill Doede"},
  "50818": {"title": "How to Make Friends", "author": "Jim Harmon"},
  "50827": {"title": "Orphans of the Void", "author": "Michael Shaara"},
  "50847": {"title": "Tea Tray in the Sky", "author": "Evelyn E. Smith"},
  "50848": {"title": "Soldier Boy", "author": "Michael Shaara"},
  "50868": {"title": "The Highest Mountain", "author": "Bryce Walton"},
  "50869": {"title": "A Gleeb for Earth", "author": "Charles Shafhauser"},
  "50905": {"title": "Yesterday House", "author": "Fritz Leiber"},
  "50923": {"title": "The Serpent River", "author": "Don Wilcox"},
  "50969": {"title": "Big Ancestor", "author": "F. L. Wallace"},
  "50988": {"title": "Bodyguard", "author": "Christopher Grimm"},
  "50998": {"title": "Delay in Transit", "author": "F. L. Wallace"},
  "51129": {"title": "A Gift From Earth", "author": "Manly Banister"},
  "51150": {"title": "Venus Is a Man's World", "author": "William Tenn"},
  "51152": {"title": "Appointment in Tomorrow", "author": "Fritz Leiber"},
  "51167": {"title": "Butterfly 9", "author": "Donald Keith"},
  "51203": {"title": "A Coffin for Jacob", "author": "Edward W. Ludwig"},
  "51210": {"title": "I, the Unspeakable", "author": "Walt Sheldon"},
  "51241": {"title": "Bridge Crossing", "author": "Dave Dryfoos"},
  "51249": {"title": "Spacemen Die at Home", "author": "Edward W. Ludwig"},
  "51267": {"title": "End as a Hero", "author": "Keith Laumer"},
  "51268": {"title": "The Girls from Earth", "author": "Frank M. Robinson"},
  "51274": {"title": "Ambition", "author": "William L. Bade"},
  "51296": {"title": "The Sense of Wonder", "author": "Milton Lesser"},
  "51305": {"title": "Confidence Game", "author": "Jim Harmon"},
  "51310": {"title": "My Lady Greensleeves", "author": "Frederik Pohl"},
  "51321": {"title": "Prime Difference", "author": "Alan E. Nourse"},
  "51330": {"title": "I Am a Nucleus", "author": "Stephen Barr"},
  "51336": {"title": "What is POSAT?", "author": "Phyllis Sterling Smith"},
  "51337": {"title": "The Man Outside", "author": "Evelyn E. Smith"},
  "51351": {"title": "The Spicy Sound of Success", "author": "Jim Harmon"},
  "51353": {"title": "Dr. Kometevsky's Day", "author": "Fritz Leiber"},
  "51361": {"title": "Birds of a Feather", "author": "Robert Silverberg"},
  "51362": {"title": "Lex", "author": "W. T. Haggert"},
  "51380": {"title": "Time in the Round", "author": "Fritz Leiber"},
  "51398": {"title": "Growing Up on Big Muddy", "author": "Charles V. De Vet"},
  "51413": {"title": "The Ignoble Savages", "author": "Evelyn E. Smith"},
  "51433": {"title": "Hunt the Hunter", "author": "Kris Neville"},
  "51494": {"title": "Beach Scene", "author": "Marshall King"},
  "51597": {"title": "Gourmet", "author": "Allen Kim Lang"},
  "51609": {"title": "A Fall of Glass", "author": "Stanley R. Lee"},
  "51650": {"title": "Innocent at Large", "author": "Poul and Karen Anderson"},
  "51656": {"title": "Pick a Crime", "author": "Richard R. Smith"},
  "51662": {"title": "Breakdown", "author": "Herbert D. Kastle"},
  "51687": {"title": "The Spy in the Elevator", "author": "Donald E. Westlake"},
  "52844": {"title": "The Long Remembered Thunder", "author": "Keith Laumer"},
  "52855": {"title": "The Star-Sent Knaves", "author": "Keith Laumer"},
  "52995": {"title": "Spaceman on a Spree", "author": "Mack Reynolds"},
  "55801": {"title": "The First Man into Space", "author": "Richard M. Elam, Jr."},
  "60747": {"title": "The Little Red Bag", "author": "Jerry Sohl"},
  "61007": {"title": "In the Garden", "author": "R. A. Lafferty"},
  "61048": {"title": "The Girls from Fieu Dayol", "author": "Robert F. Young"},
  "61052": {"title": "Spawning Ground", "author": "Lester del Rey"},
  "61053": {"title": "Tolliver's Orbit", "author": "H. B. Fyfe"},
  "61081": {"title": "Cinderella Story", "author": "Allen Kim Lang"},
  "61090": {"title": "Call Him Nemesis", "author": "Donald E. Westlake"},
  "61097": {"title": "The Frozen Planet", "author": "Keith Laumer"},
  "61139": {"title": "The Madman from Earth", "author": "Keith Laumer"},
  "61146": {"title": "Retief of the Red-Tape Mountain", "author": "Keith Laumer"},
  "61171": {"title": "The Expendables", "author": "Jim Harmon"},
  "61198": {"title": "Aide Memoire", "author": "Keith Laumer"},
  "61204": {"title": "The Recruit", "author": "Bryce Walton"},
  "61228": {"title": "The Big Headache", "author": "Jim Harmon"},
  "61242": {"title": "The Winning of the Moon", "author": "Kris Neville"},
  "61263": {"title": "Cultural Exchange", "author": "Keith Laumer"},
  "61285": {"title": "The Desert and the Stars", "author": "Keith Laumer"},
  "61380": {"title": "The Five Hells of Orion", "author": "Frederik Pohl"},
  "61405": {"title": "Down to the Worlds of Men", "author": "Alexei Panshin"},
  "61434": {"title": "Mightiest Qorn", "author": "Keith Laumer"},
  "61467": {"title": "Muck Man", "author": "Fremont Dodge"},
  "62039": {"title": "The Lorelei Death", "author": "Nelson S. Bond"},
  "62198": {"title": "Quest of Thig", "author": "Basil Wells"},
  "62212": {"title": "Prison Planet", "author": "Bob Tucker"},
  "62244": {"title": "Galactic Ghost", "author": "Walter Kubilius"},
  "62260": {"title": "Trouble on Tycho", "author": "Nelson S. Bond"},
  "62324": {"title": "Grifters' Asteroid", "author": "H. L. Gold"},
  "62349": {"title": "The Blue Behemoth", "author": "Leigh Brackett"},
  "62569": {"title": "The Monster Maker", "author": "Ray Bradbury"},
  "62619": {"title": "The Avenger", "author": "Stuart Fleming"},
  "62997": {"title": "Saboteur of Space", "author": "Robert Abernathy"},
  "63041": {"title": "Morgue Ship", "author": "Ray Bradbury"},
  "63048": {"title": "Wanderers of the Wolf Moon", "author": "Nelson S. Bond"},
  "63109": {"title": "Doctor Universe", "author": "Carl Jacobi"},
  "63150": {"title": "The Soul Eaters", "author": "William Conover"},
  "63304": {"title": "Doublecross", "author": "James Mac Creigh"},
  "63392": {"title": "Doorway to Kal-Jmar", "author": "Stuart Fleming"},
  "63401": {"title": "The Happy Castaway", "author": "Robert E. McDowell"},
  "63419": {"title": "Death Star", "author": "Tom Pace"},
  "63442": {"title": "Double Trouble", "author": "Carl Jacobi"},
  "63473": {"title": "Dust Unto Dust", "author": "Lyman D. Hinckley"},
  "63477": {"title": "Image of Splendor", "author": "Lu Kella"},
  "63521": {"title": "Raiders of the Second Moon", "author": "Gene Ellerman"},
  "63527": {"title": "Cosmic Yo-Yo", "author": "Ross Rocklynne"},
  "63605": {"title": "The Beast-Jewel of Mars", "author": "V. E. Thiessen"},
  "63616": {"title": "Hagerty's Enzymes", "author": "A. L. Haley"},
  "63833": {"title": "Jinx Ship to the Rescue", "author": "Alfred Coppel, Jr."},
  "63860": {"title": "Signal Red", "author": "Henry Guth"},
  "63862": {"title": "Stalemate in Space", "author": "Charles L. Harness"},
  "63867": {"title": "Captain Midas", "author": "Alfred Coppel, Jr."},
  "63874": {"title": "The Creatures That Time Forgot", "author": "Ray Bradbury"},
  "63890": {"title": "A Planet Named Joe", "author": "S. A. Lombino"},
  "63899": {"title": "The Giants Return", "author": "Robert Abernathy"},
  "63916": {"title": "The Conjurer of Venus", "author": "Conan T. Troy"},
  "63932": {"title": "The Lost Tribes of Venus", "author": "Erik Fennel"}
}




def create_output_directory(output_dir: Path) -> None:
    """Create the output directory structure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    items_dir = output_dir / "items"
    items_dir.mkdir(exist_ok=True)
    return items_dir


def process_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process questions to extract structured information."""
    processed_questions = []
    
    for question in questions:
        # Extract responses
        responses = []
        for response in question.get('responses', []):
            responses.append({
                'worker_id': response.get('worker_id'),
                'uid': response.get('uid'),
                'response_text': response.get('response_text')
            })
        
        processed_question = {
            'question_text': question.get('question_text'),
            'question_number': question.get('question_number'),
            'responses': responses
        }
        processed_questions.append(processed_question)
    
    return processed_questions


def process_squality_item(item: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Convert a SQuALITY item to IUS format."""
    passage_id = str(item['metadata']['passage_id'])
    
    # Process questions
    processed_questions = process_questions(item['questions'])
    
    # Create item in IUS format
    ius_item = {
        "item_metadata": {
            "item_id": passage_id,
            "num_documents": 1
        },
        "documents": [
            {
                "doc_id": passage_id,
                "content": item['document'],
                "metadata": {
                    "title": metadata_map.get(passage_id, {}).get("title", ""),
                    "author": metadata_map.get(passage_id, {}).get("author", ""),
                    "passage_id": passage_id,
                    "uid": item['metadata']['uid'],
                    "license": item['metadata']['license'],
                    "split": split,  # Add split information
                    "questions": processed_questions,
                    "num_questions": len(processed_questions),
                    "document_length_words": len(item['document'].split()),
                    "document_length_chars": len(item['document'])
                }
            }
        ]
    }
    
    return ius_item


def main():
    """Main ingestion process."""
    print("🚀 Starting SQuALITY dataset ingestion...")
    
    # Set up output directory
    output_dir = Path("datasets/squality")
    items_dir = create_output_directory(output_dir)
    
    # Load SQuALITY dataset
    print("📥 Loading SQuALITY dataset from Hugging Face...")
    dataset = load_dataset("pszemraj/SQuALITY-v1.3")
    
    print(f"📊 Dataset splits: {list(dataset.keys())}")
    total_items = sum(len(dataset[split]) for split in dataset.keys())
    print(f"📊 Total items across all splits: {total_items}")
    
    # Process all items
    processed_items = []
    split_counts = {}
    
    for split_name in dataset.keys():
        print(f"\n🔄 Processing {split_name} split...")
        split_data = dataset[split_name]
        split_counts[split_name] = len(split_data)
        
        for i, item in enumerate(split_data):
            passage_id = str(item['metadata']['passage_id'])
            
            # Convert to IUS format
            ius_item = process_squality_item(item, split_name)
            
            # Write individual item file
            item_file = items_dir / f"{passage_id}.json"
            with open(item_file, 'w', encoding='utf-8') as f:
                json.dump(ius_item, f, indent=2, ensure_ascii=False)
            
            processed_items.append(passage_id)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(split_data):
                print(f"  ✅ Processed {i + 1}/{len(split_data)} items from {split_name}")
    
    # Create collection.json
    print(f"\n📝 Creating collection metadata...")
    collection_data = {
        "domain": "sci-fi story",
        "source": "SQuALITY dataset - pszemraj/SQuALITY-v1.3 from Hugging Face",
        "created": datetime.now().isoformat(),
        "num_items": len(processed_items),
        "total_documents": len(processed_items),
        "description": "Reading comprehension stories from SQuALITY dataset with questions and human responses",
        "items": processed_items,
        "split_distribution": split_counts,
        "dataset_info": {
            "total_items": total_items,
            "splits": list(dataset.keys()),
            "questions_per_story": 5,
            "responses_per_question": "~4 (varies)"
        }
    }
    
    collection_file = output_dir / "collection.json"
    with open(collection_file, 'w', encoding='utf-8') as f:
        json.dump(collection_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n📊 INGESTION SUMMARY:")
    print(f"✅ Total items processed: {len(processed_items)}")
    print(f"✅ Split distribution:")
    for split, count in split_counts.items():
        print(f"   - {split}: {count} items")
    print(f"✅ Output directory: {output_dir}")
    print(f"✅ Collection file: {collection_file}")
    print(f"✅ Items directory: {items_dir}")
    
    # Verify passage_id uniqueness
    if len(processed_items) == len(set(processed_items)):
        print(f"✅ All passage_ids are unique")
    else:
        print(f"⚠️  Warning: Found duplicate passage_ids!")
    
    print(f"\n🎉 SQuALITY ingestion completed successfully!")
    print(f"📁 Dataset ready at: {output_dir}")


if __name__ == "__main__":
    main()