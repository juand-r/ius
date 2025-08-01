#!/usr/bin/env python3
"""
Tests for CLI common utilities.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ius.cli.common import print_summary_stats, save_json_output, setup_output_dir


class TestCLICommon(unittest.TestCase):
    """Test cases for CLI common utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_setup_output_dir_default(self):
        """Test setting up default output directory."""
        result = setup_output_dir()
        self.assertEqual(result, Path("outputs"))
        self.assertTrue(result.exists())

    def test_setup_output_dir_custom_path(self):
        """Test setting up custom output directory."""
        custom_path = str(self.temp_path / "custom" / "output.json")
        result = setup_output_dir(custom_path)
        expected = Path(custom_path).parent
        self.assertEqual(result, expected)
        self.assertTrue(result.exists())

    def test_setup_output_dir_nested_directories(self):
        """Test setting up nested output directories."""
        nested_path = str(self.temp_path / "deep" / "nested" / "dirs" / "output.json")
        result = setup_output_dir(nested_path)
        self.assertTrue(result.exists())
        self.assertEqual(result, Path(nested_path).parent)

    def test_save_json_output_success(self):
        """Test successful JSON output saving."""
        test_data = {"test": "data", "number": 42}
        output_path = str(self.temp_path / "test_output.json")

        with patch('builtins.print') as mock_print:
            save_json_output(test_data, output_path)

        # Check file was created and contains correct data
        self.assertTrue(Path(output_path).exists())
        with open(output_path, encoding='utf-8') as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, test_data)

        # Check success message was printed
        mock_print.assert_called_with(f"✅ Output saved to: {Path(output_path)}")

    def test_save_json_output_pretty_formatting(self):
        """Test JSON output with pretty formatting."""
        test_data = {"test": "data", "nested": {"key": "value"}}
        output_path = str(self.temp_path / "pretty_output.json")

        save_json_output(test_data, output_path, pretty=True)

        with open(output_path, encoding='utf-8') as f:
            content = f.read()

        # Pretty formatted JSON should contain newlines and indentation
        self.assertIn('\n', content)
        self.assertIn('  ', content)  # Indentation

    def test_save_json_output_compact_formatting(self):
        """Test JSON output with compact formatting."""
        test_data = {"test": "data", "nested": {"key": "value"}}
        output_path = str(self.temp_path / "compact_output.json")

        save_json_output(test_data, output_path, pretty=False)

        with open(output_path, encoding='utf-8') as f:
            content = f.read()

        # Compact JSON should be single line
        self.assertEqual(content.count('\n'), 0)

    def test_save_json_output_creates_directories(self):
        """Test that save_json_output creates parent directories."""
        nested_path = str(self.temp_path / "new" / "nested" / "output.json")
        test_data = {"test": "data"}

        save_json_output(test_data, nested_path)

        self.assertTrue(Path(nested_path).exists())
        self.assertTrue(Path(nested_path).parent.exists())

    def test_save_json_output_unicode_handling(self):
        """Test JSON output with unicode characters."""
        test_data = {"unicode": "café", "emoji": "🎉", "chinese": "测试"}
        output_path = str(self.temp_path / "unicode_output.json")

        save_json_output(test_data, output_path)

        with open(output_path, encoding='utf-8') as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, test_data)

    @patch('sys.exit')
    @patch('builtins.print')
    def test_save_json_output_permission_error(self, mock_print, mock_exit):
        """Test save_json_output handles permission errors."""
        # Try to write to root directory (should fail)
        invalid_path = "/root/cannot_write_here.json"
        test_data = {"test": "data"}

        save_json_output(test_data, invalid_path)

        mock_exit.assert_called_with(1)
        # Check error message was printed to stderr
        mock_print.assert_called()
        call_args = mock_print.call_args
        self.assertIn("❌ Error saving output", str(call_args))

    def test_print_summary_stats_mixed_types(self):
        """Test printing summary statistics with mixed data types."""
        stats = {
            "total_items": 42,
            "avg_score": 85.7,
            "name": "test_dataset",
            "success_rate": 0.95
        }

        with patch('builtins.print') as mock_print:
            print_summary_stats(stats)

        # Check that print was called with header
        calls = mock_print.call_args_list
        self.assertTrue(any("📊 Summary Statistics:" in str(call) for call in calls))

        # Check float formatting
        self.assertTrue(any("avg_score: 85.7" in str(call) for call in calls))
        self.assertTrue(any("success_rate: 0.9" in str(call) for call in calls))

        # Check non-float values
        self.assertTrue(any("total_items: 42" in str(call) for call in calls))
        self.assertTrue(any("name: test_dataset" in str(call) for call in calls))

    def test_print_summary_stats_empty_dict(self):
        """Test printing empty statistics dictionary."""
        with patch('builtins.print') as mock_print:
            print_summary_stats({})

        # Should still print header
        mock_print.assert_called_with("\n📊 Summary Statistics:")

    def test_print_summary_stats_float_precision(self):
        """Test float precision in summary statistics."""
        stats = {
            "precise_value": 3.14159265359,
            "whole_number": 5.0,
            "small_decimal": 0.001
        }

        with patch('builtins.print') as mock_print:
            print_summary_stats(stats)

        calls = mock_print.call_args_list
        # All floats should be formatted to 1 decimal place
        self.assertTrue(any("precise_value: 3.1" in str(call) for call in calls))
        self.assertTrue(any("whole_number: 5.0" in str(call) for call in calls))
        self.assertTrue(any("small_decimal: 0.0" in str(call) for call in calls))


if __name__ == "__main__":
    unittest.main(verbosity=2)
