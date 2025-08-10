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
                    "title": "",  # Empty as requested
                    "author": "",  # Empty as requested
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