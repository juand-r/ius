"""
Common utilities for CLI modules.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def setup_output_dir(output_path: Optional[str] = None) -> Path:
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


def save_json_output(data: Dict[str, Any], output_path: str, pretty: bool = True) -> None:
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


def print_summary_stats(stats: Dict[str, Any]) -> None:
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


def validate_dataset_exists(dataset_name: str) -> bool:
    """
    Check if dataset exists in the datasets directory.

    Args:
        dataset_name: Name of the dataset

    Returns:
        True if dataset exists, False otherwise
    """
    dataset_path = Path("datasets") / dataset_name
    return dataset_path.exists() and dataset_path.is_dir()


def list_available_datasets() -> List[str]:
    """
    List all available datasets in the datasets directory.

    Returns:
        List of dataset names
    """
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        return []

    return [d.name for d in datasets_dir.iterdir() if d.is_dir()]
