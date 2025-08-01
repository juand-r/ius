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


def save_json_output(data: dict[str, Any], output_path: str, pretty: bool = True) -> None:
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

        with open(output_file, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)

        print(f"✅ Output saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error saving output to {output_path}: {e}", file=sys.stderr)
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

