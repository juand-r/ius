"""
Tests for the chunking CLI functionality.
"""

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ius.cli.chunk import chunk_dataset, main
from ius.cli.common import save_json_output


class TestChunkingCLI(unittest.TestCase):
    """Test cases for chunking CLI functionality."""

    def setUp(self):
        """Set up test environment."""
        # Mock data in the format that load_data() actually returns
        self.test_dataset = {
            "items": {
                "item_1": {
                    "documents": [
                        {"content": "This is a test document.\nIt has multiple lines.\nFor testing purposes.", "doc_id": "item_1"}
                    ]
                },
                "item_2": {
                    "documents": [
                        {"content": "Another document.\nWith different content.\nShort lines here.", "doc_id": "item_2"}
                    ]
                },
            },
            "collection_metadata": {"num_items": 2},
            "num_items_loaded": 2
        }

        self.multi_doc_dataset = {
            "items": {
                "story_1": {
                    "documents": [
                        {"content": "Chapter 1: The beginning.\nSomething happened.", "doc_id": "story_1_doc1"},
                        {"content": "Chapter 2: The middle.\nMore events occurred.", "doc_id": "story_1_doc2"},
                    ]
                }
            },
            "collection_metadata": {"num_items": 1},
            "num_items_loaded": 1
        }

        self.empty_dataset = {
            "items": {},
            "collection_metadata": {"num_items": 0},
            "num_items_loaded": 0
        }

    @patch('ius.cli.chunk.load_data')
    def test_chunk_dataset_fixed_size(self, mock_load_data):
        """Test dataset chunking with fixed size strategy."""
        mock_load_data.return_value = self.test_dataset

        result = chunk_dataset(
            dataset_name="test",
            strategy="fixed_size",
            chunk_size=30,
            delimiter="\n",

        )

        self.assertEqual(result["dataset"], "test")
        self.assertEqual(result["strategy"], "fixed_size")
        self.assertEqual(len(result["items"]), 2)

        # Check item_1 results
        item_1 = result["items"]["item_1"]
        self.assertTrue(item_1["validation_passed"])
        self.assertGreater(item_1["overall_stats"]["total_chunks"], 1)
        self.assertEqual(item_1["strategy"], "fixed_size")

    @patch('ius.cli.chunk.load_data')
    def test_chunk_dataset_fixed_count(self, mock_load_data):
        """Test dataset chunking with fixed count strategy."""
        mock_load_data.return_value = self.test_dataset

        result = chunk_dataset(
            dataset_name="test",
            strategy="fixed_count",
            num_chunks=2,
            delimiter="\n",

        )

        self.assertEqual(result["strategy"], "fixed_count")

        # Each item should have exactly 2 chunks (or fewer if not enough delimiters)
        for _item_id, item_data in result["items"].items():
            self.assertLessEqual(item_data["overall_stats"]["total_chunks"], 2)

    @patch('ius.cli.chunk.load_data')
    def test_chunk_dataset_multi_document(self, mock_load_data):
        """Test chunking dataset with multi-document items."""
        mock_load_data.return_value = self.multi_doc_dataset

        result = chunk_dataset(
            dataset_name="test",
            strategy="fixed_size",
            chunk_size=50,
            delimiter="\n",

        )

        # Should handle multi-document items by concatenating
        self.assertEqual(len(result["items"]), 1)
        story_1 = result["items"]["story_1"]
        self.assertTrue(story_1["validation_passed"])
        self.assertGreater(story_1["original_length"], 50)  # Combined text

    @patch('ius.cli.chunk.load_data')
    def test_chunk_dataset_with_output(self, mock_load_data):
        """Test chunking with output file saving."""
        mock_load_data.return_value = self.test_dataset

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            output_path = tmp.name

        try:
            chunk_dataset(
                dataset_name="test",
                strategy="fixed_size",
                chunk_size=30,
                output_path=output_path,

            )

            # Check that file was created and contains expected data
            self.assertTrue(Path(output_path).exists())

            with open(output_path) as f:
                saved_data = json.load(f)

            self.assertEqual(saved_data["dataset"], "test")
            self.assertEqual(len(saved_data["items"]), 2)

        finally:
            # Cleanup
            Path(output_path).unlink(missing_ok=True)

    @patch('ius.cli.chunk.load_data')
    def test_chunk_dataset_empty_items(self, mock_load_data):
        """Test handling of empty or invalid items."""
        mock_dataset = {
            "items": {
                "empty": {"documents": [{"content": "", "doc_id": "empty"}]},
                "missing_documents": {"title": "No documents field"},
                "valid": {"documents": [{"content": "Valid content here.\nWith multiple lines.\nFor proper chunking.", "doc_id": "valid"}]},
            },
            "collection_metadata": {"num_items": 3},
            "num_items_loaded": 3
        }
        mock_load_data.return_value = mock_dataset

        result = chunk_dataset(
            dataset_name="test",
            strategy="fixed_size",
            chunk_size=10,

        )

        # Should process only the valid item with proper content and delimiters
        # empty content and missing_documents should be caught by validation
        self.assertEqual(len(result["items"]), 1)
        self.assertIn("valid", result["items"])

        # Should have 2 errors (empty and missing_documents)
        self.assertEqual(len(result["errors"]), 2)
        self.assertIn("empty", result["errors"])
        self.assertIn("missing_documents", result["errors"])

    @patch('ius.cli.chunk.load_data')
    def test_chunk_dataset_validation_failure(self, mock_load_data):
        """Test handling of validation failures."""
        mock_load_data.return_value = self.test_dataset

        # Mock chunking function to raise validation error
        with patch('ius.chunk.chunkers.chunk_fixed_size', side_effect=ValueError("Content validation failed")):
            result = chunk_dataset(
                dataset_name="test",
                strategy="fixed_size",
                chunk_size=30,

            )

            # Should handle validation failures gracefully
            # Items with validation errors should be skipped
            self.assertEqual(result["overall_stats"]["total_items"], 0)
            self.assertEqual(len(result["items"]), 0)


class TestCLICommands(unittest.TestCase):
    """Test CLI command-line interface."""

    def test_main_help(self):
        """Test main help functionality."""
        with (
            patch('sys.argv', ['ius', '--help']),
            patch('sys.stdout', new_callable=StringIO) as mock_stdout,
            self.assertRaises(SystemExit) as cm,
        ):
            main()
            self.assertEqual(cm.exception.code, 0)
            output = mock_stdout.getvalue()
            self.assertIn("Chunk documents for incremental summarization", output)

    def test_main_list_datasets(self):
        """Test listing available datasets."""
        test_args = ['ius', '--list-datasets']
        with (
            patch('sys.argv', test_args),
            patch('ius.cli.chunk.list_datasets', return_value=['bmds', 'test']),
            patch('sys.stdout', new_callable=StringIO) as mock_stdout,
        ):
            main()
            output = mock_stdout.getvalue()
            self.assertIn("Available datasets:", output)
            self.assertIn("bmds", output)
            self.assertIn("test", output)

    @patch('ius.cli.chunk.list_datasets')
    @patch('ius.cli.chunk.chunk_dataset')
    def test_main_successful_run(self, mock_chunk_dataset, mock_list_datasets):
        """Test successful CLI execution."""
        mock_list_datasets.return_value = ['test']
        mock_chunk_dataset.return_value = {"dataset": "test", "items": {"item1": {}}}

        test_args = [
            'ius', '--dataset', 'test',
            '--strategy', 'fixed_size',
            '--size', '1000'
        ]

        with (
            patch('sys.argv', test_args),
            patch('sys.stdout', new_callable=StringIO),
        ):
            main()  # Should not raise

        # Verify chunk_dataset was called with correct arguments
        mock_chunk_dataset.assert_called_once()
        call_args = mock_chunk_dataset.call_args
        self.assertEqual(call_args.kwargs['dataset_name'], 'test')
        self.assertEqual(call_args.kwargs['strategy'], 'fixed_size')
        self.assertEqual(call_args.kwargs['chunk_size'], 1000)

    def test_main_missing_required_args(self):
        """Test error handling for missing required arguments."""
        with (
            patch('sys.argv', ['ius', '--strategy', 'fixed_size']),
            patch('sys.stderr', new_callable=StringIO),
            self.assertRaises(SystemExit) as cm,
        ):
            main()
            self.assertNotEqual(cm.exception.code, 0)

    def test_main_invalid_strategy_args(self):
        """Test error handling for invalid strategy arguments."""
        # fixed_size without size
        with (
            patch('sys.argv', ['ius', '--dataset', 'test', '--strategy', 'fixed_size']),
            patch('sys.stderr', new_callable=StringIO),
            self.assertRaises(SystemExit) as cm,
        ):
            main()
            self.assertNotEqual(cm.exception.code, 0)

        # fixed_count without count
        with (
            patch('sys.argv', ['ius', '--dataset', 'test', '--strategy', 'fixed_count']),
            patch('sys.stderr', new_callable=StringIO),
            self.assertRaises(SystemExit) as cm,
        ):
            main()
            self.assertNotEqual(cm.exception.code, 0)

    @patch('ius.cli.chunk.list_datasets')
    def test_main_nonexistent_dataset(self, mock_list_datasets):
        """Test error handling for nonexistent dataset."""
        mock_list_datasets.return_value = ['bmds', 'other']

        test_args = [
            'ius', '--dataset', 'nonexistent',
            '--strategy', 'fixed_size',
            '--size', '1000'
        ]

        with (
            patch('sys.argv', test_args),
            patch('sys.stderr', new_callable=StringIO) as mock_stderr,
            self.assertRaises(SystemExit) as cm,
        ):
            main()
            self.assertNotEqual(cm.exception.code, 0)

            error_output = mock_stderr.getvalue()
            self.assertIn("Dataset 'nonexistent' not found", error_output)
            self.assertIn("Available datasets:", error_output)


class TestCLICommon(unittest.TestCase):
    """Test common CLI utilities."""

    def test_save_json_output(self):
        """Test JSON output saving."""
        test_data = {"test": "data", "number": 42}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            output_path = tmp.name

        try:
            with patch('sys.stdout', new_callable=StringIO):
                save_json_output(test_data, output_path)

            # Verify file was created and contains correct data
            self.assertTrue(Path(output_path).exists())

            with open(output_path) as f:
                loaded_data = json.load(f)

            self.assertEqual(loaded_data, test_data)

        finally:
            Path(output_path).unlink(missing_ok=True)




class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI with real chunking functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test data for integration tests."""
        # Load real BMDS data for integration testing
        try:
            from ius.data import load_data
            cls.bmds_data = load_data("bmds")
            cls.has_bmds = True
        except Exception:
            cls.has_bmds = False

    def test_real_bmds_chunking(self):
        """Integration test with real BMDS data."""
        if not self.has_bmds:
            self.skipTest("BMDS dataset not available")

        # Use the actual BMDS items structure
        from ius.data import load_data
        dataset = load_data("bmds")

        # BMDS has nested structure - extract the actual items
        if 'items' in dataset:
            items = dataset['items']
            # Create a small test with just one item
            test_item_id = list(items.keys())[0]
            test_dataset = {
                "items": {test_item_id: items[test_item_id]},
                "collection_metadata": dataset.get("collection_metadata", {}),
                "num_items_loaded": 1
            }

            # Mock load_data to return our test dataset
            with patch('ius.cli.chunk.load_data', return_value=test_dataset):
                result = chunk_dataset(
                    dataset_name="bmds",
                    strategy="fixed_size",
                    chunk_size=1000,
                    delimiter="\n",

                )

                self.assertEqual(result["dataset"], "bmds")
                self.assertGreater(len(result["items"]), 0)

                # All items should pass validation
                for item_data in result["items"].values():
                    self.assertTrue(item_data["validation_passed"])
                    self.assertGreater(item_data["overall_stats"]["total_chunks"], 0)
        else:
            self.skipTest("BMDS dataset format not as expected")


if __name__ == "__main__":
    unittest.main()
