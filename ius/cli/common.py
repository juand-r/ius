"""
Common utilities for CLI modules.
"""

import json
import sys
from pathlib import Path
from typing import Any


def setup_output_dir(output_path: str | None = None) -> Path:
    """
    Set up output directory, creating it if it doesn't exist.

    Args:
        output_path: Optional custom output path

    Returns:
        Path object for the output directory
    """
    output_dir = Path(output_path).parent if output_path else Path("outputs")

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json_output(
    data: dict[str, Any], output_path: str, pretty: bool = True
) -> None:
    """
    Save data to JSON file with error handling.

    Args:
        data: Data to save
        output_path: Path to save the JSON file
        pretty: Whether to pretty-print the JSON
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)

        print(f"✅ Output saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error saving output to {output_path}: {e}", file=sys.stderr)
        sys.exit(1)


def save_chunked_collection_and_items(
    chunked_collection: dict[str, Any], chunked_items: dict[str, Any], 
    output_directory: str, pretty: bool = True
) -> None:
    """
    Save chunked collection and individual items in the proper directory structure.

    Args:
        chunked_collection: Collection-level metadata and statistics
        chunked_items: Dictionary of item_id -> chunked item data
        output_directory: Directory to save the chunked data
        pretty: Whether to pretty-print the JSON
    """
    try:
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create items subdirectory
        items_dir = output_dir / "items"
        items_dir.mkdir(exist_ok=True)

        # Save collection.json
        collection_file = output_dir / "collection.json"
        with open(collection_file, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(chunked_collection, f, indent=2, ensure_ascii=False)
            else:
                json.dump(chunked_collection, f, ensure_ascii=False)

        # Save individual items
        saved_count = 0
        for item_id, item_data in chunked_items.items():
            item_file = items_dir / f"{item_id}.json"
            
            with open(item_file, "w", encoding="utf-8") as f:
                if pretty:
                    json.dump(item_data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(item_data, f, ensure_ascii=False)
            
            saved_count += 1

        print(f"✅ Collection and {saved_count} chunked items saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Error saving chunked data to {output_directory}: {e}", file=sys.stderr)
        sys.exit(1)


def print_summary_stats(stats: dict[str, Any]) -> None:
    """
    Print formatted summary statistics.

    Args:
        stats: Statistics dictionary to display
    """
    print("\n📊 Summary Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
