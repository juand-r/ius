#!/usr/bin/env python3
"""
Book data ingester.

Loads books from data-source/all_books_booookscore.pkl and converts them
to individual text files for processing.
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Union


def load_books_pickle(pickle_path: Union[str, Path]) -> Any:
    """Load books from pickle file."""
    pickle_path = Path(pickle_path)
    
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
    
    print(f"Loading books from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Loaded pickle file")
    return data


def inspect_data_structure(data: Any, max_depth: int = 2) -> None:
    """Inspect the structure of the loaded data."""
    print("\n📊 Data Structure:")
    
    def inspect_recursive(obj, depth=0, name="root"):
        if depth > max_depth:
            return
            
        indent = "  " * depth
        
        if isinstance(obj, dict):
            print(f"{indent}{name}: dict with {len(obj)} keys")
            if depth < max_depth:
                for key in list(obj.keys())[:5]:  # Show first 5 keys
                    inspect_recursive(obj[key], depth + 1, f"['{key}']")
                if len(obj) > 5:
                    print(f"{indent}  ... and {len(obj) - 5} more keys")
                    
        elif isinstance(obj, list):
            print(f"{indent}{name}: list with {len(obj)} items")
            if depth < max_depth and len(obj) > 0:
                inspect_recursive(obj[0], depth + 1, "[0]")
                if len(obj) > 1:
                    print(f"{indent}  ... and {len(obj) - 1} more items")
                    
        elif isinstance(obj, str):
            preview = obj[:100] + "..." if len(obj) > 100 else obj
            print(f"{indent}{name}: str (len={len(obj)}) '{preview}'")
            
        else:
            print(f"{indent}{name}: {type(obj).__name__}")
    
    inspect_recursive(data)


def extract_books_to_txt(data: Any, output_dir: Union[str, Path], dry_run: bool = False) -> None:
    """
    Extract books from data structure and save as individual txt files.
    
    Args:
        data: Loaded pickle data
        output_dir: Directory to save txt files
        dry_run: If True, don't actually write files, just show what would be done
    """
    output_dir = Path(output_dir)
    
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
    
    # This is where we'll need to adapt based on the actual data structure
    # For now, let's handle common patterns
    
    books_found = 0
    
    if isinstance(data, dict):
        # Case 1: Dictionary with book IDs as keys
        for book_id, book_content in data.items():
            books_found += 1
            
            # Extract text content (adapt as needed)
            if isinstance(book_content, str):
                text = book_content
            elif isinstance(book_content, dict) and 'text' in book_content:
                text = book_content['text']
            elif isinstance(book_content, dict) and 'content' in book_content:
                text = book_content['content']
            else:
                print(f"⚠️  Skipping {book_id}: Unknown content structure")
                continue
            
            # Clean book ID for filename
            safe_book_id = str(book_id).replace('/', '_').replace('\\', '_')
            filename = f"{safe_book_id}.txt"
            filepath = output_dir / filename
            
            if dry_run:
                print(f"📄 Would save: {filename} ({len(text)} chars)")
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"📄 Saved: {filename} ({len(text)} chars)")
    
    elif isinstance(data, list):
        # Case 2: List of books
        for i, book_content in enumerate(data):
            books_found += 1
            
            # Extract book ID and text
            if isinstance(book_content, dict):
                book_id = book_content.get('id', book_content.get('book_id', f"book_{i+1}"))
                text = book_content.get('text', book_content.get('content', ''))
            else:
                book_id = f"book_{i+1}"
                text = str(book_content)
            
            safe_book_id = str(book_id).replace('/', '_').replace('\\', '_')
            filename = f"{safe_book_id}.txt"
            filepath = output_dir / filename
            
            if dry_run:
                print(f"📄 Would save: {filename} ({len(text)} chars)")
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"📄 Saved: {filename} ({len(text)} chars)")
    
    else:
        print(f"❌ Unsupported data structure: {type(data)}")
        return
    
    print(f"\n✅ Processed {books_found} books")


def main():
    parser = argparse.ArgumentParser(description="Ingest books from pickle file")
    parser.add_argument(
        "--pickle-path", 
        default="data-source/all_books_booookscore.pkl",
        help="Path to pickle file (default: data-source/all_books_booookscore.pkl)"
    )
    parser.add_argument(
        "--output-dir",
        default="data-source/books_txt",
        help="Output directory for txt files (default: data-source/books_txt)"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect data structure without extracting"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Show what would be done without actually writing files"
    )
    
    args = parser.parse_args()
    
    try:
        # Load the pickle file
        data = load_books_pickle(args.pickle_path)
        
        # Always inspect the structure first
        inspect_data_structure(data)
        
        if args.inspect:
            print("\n🔍 Inspection complete. Use without --inspect to extract books.")
            return
        
        # Extract books to txt files
        print(f"\n📚 Extracting books...")
        extract_books_to_txt(data, args.output_dir, dry_run=args.dry_run)
        
        if args.dry_run:
            print("\n💡 This was a dry run. Remove --dry-run to actually save files.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 