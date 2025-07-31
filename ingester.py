#!/usr/bin/env python3
"""
BooookScore book data ingester.

Loads books from data-source/booookscore/all_books.pkl (dict[filename, text])
and extracts them to individual text files for processing.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Union


def load_books_pickle(pickle_path: Union[str, Path]) -> Dict[str, str]:
    """Load books from pickle file."""
    pickle_path = Path(pickle_path)

    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    print(f"Loading books from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    print("✅ Loaded pickle file")
    return data


def inspect_data_structure(data: Dict[str, str]) -> None:
    """Inspect the BooookScore pickle data structure."""
    print("\n📊 Data Structure:")
    print(f"root: dict with {len(data)} books")

    # Show first 5 book entries as examples
    for i, (epub_filename, book_text) in enumerate(data.items()):
        if i >= 5:
            print(f"  ... and {len(data) - 5} more books")
            break
        preview = book_text[:100] + "..." if len(book_text) > 100 else book_text
        print(f"  ['{epub_filename}']: str (len={len(book_text)}) '{preview}'")


def extract_books_to_txt(data: Dict[str, str], output_dir: Union[str, Path], dry_run: bool = False, show_samples: bool = False) -> None:
    """
    Extract books from BooookScore pickle data (dict[filename, text]) to individual txt files.

    Args:
        data: Dict mapping epub filenames to book text content
        output_dir: Directory to save txt files
        dry_run: If True, don't actually write files, just show what would be done
        show_samples: If True, show sample content from first few books
    """
    output_dir = Path(output_dir)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")

    text_lengths = []
    empty_books = []
    sample_count = 0
    max_samples = 3

    for epub_filename, book_text in data.items():
        # Sanity checks
        text = str(book_text).strip()
        text_lengths.append(len(text))

        if len(text) == 0:
            empty_books.append(epub_filename)
            print(f"⚠️  Warning: {epub_filename} is empty")
        elif len(text) < 1000:
            print(f"⚠️  Warning: {epub_filename} is very short ({len(text)} chars)")

        # Show sample content for first few books
        if show_samples and sample_count < max_samples and len(text) > 0:
            sample_count += 1
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"\n📖 Sample from {epub_filename}:")
            print(f"   {preview}\n")

        # Clean filename: remove .epub extension and clean for filesystem
        clean_name = epub_filename.replace('.epub', '').replace('/', '_').replace('\\', '_').replace(' ', '_')
        filename = f"{clean_name}.txt"
        filepath = output_dir / filename

        if dry_run:
            print(f"📄 Would save: {filename} ({len(text)} chars)")
        else:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"📄 Saved: {filename} ({len(text)} chars)")
            except Exception as e:
                print(f"❌ Error saving {epub_filename}: {e}")

    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   Total books: {len(data)}")

    if text_lengths:
        min_length = min(text_lengths)
        max_length = max(text_lengths)
        avg_length = sum(text_lengths) // len(text_lengths)
        total_chars = sum(text_lengths)

        print(f"   Text length - Min: {min_length:,}, Max: {max_length:,}, Avg: {avg_length:,}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Empty books: {len(empty_books)}")

        # Flag potential issues
        very_short = [i for i, length in enumerate(text_lengths) if 0 < length < 1000]
        very_long = [i for i, length in enumerate(text_lengths) if length > 1_000_000]

        if very_short:
            print(f"   ⚠️  Very short books (< 1K chars): {len(very_short)}")
        if very_long:
            print(f"   ⚠️  Very long books (> 1M chars): {len(very_long)}")
        if empty_books:
            print(f"   ❌ Empty books: {empty_books}")

    print("\n✅ Extraction complete!")


def main():
    parser = argparse.ArgumentParser(description="Ingest books from pickle file")
    parser.add_argument(
        "--pickle-path",
        default="data-source/booookscore/all_books.pkl",
        help="Path to pickle file (default: data-source/booookscore/all_books.pkl)"
    )
    parser.add_argument(
        "--output-dir",
        default="data-source/booookscore/booookscore_txt",
        help="Output directory for txt files (default: data-source/booookscore/booookscore_txt)"
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
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Show sample content from first few books"
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
        print("\n📚 Extracting books...")
        extract_books_to_txt(data, args.output_dir, dry_run=args.dry_run, show_samples=args.show_samples)

        if args.dry_run:
            print("\n💡 This was a dry run. Remove --dry-run to actually save files.")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
