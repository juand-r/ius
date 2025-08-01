"""
Tests for the chunking CLI functionality.
"""

import json
import sys
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
                        {
                            "content": "This is a test document.\nIt has multiple lines.\nFor testing purposes.",
                            "doc_id": "item_1",
                        }
                    ]
                },
                "item_2": {
                    "documents": [
                        {
                            "content": "Another document.\nWith different content.\nShort lines here.",
                            "doc_id": "item_2",
                        }
                    ]
                },
            },
            "collection_metadata": {"num_items": 2},
            "num_items_loaded": 2,
        }

        self.multi_doc_dataset = {
            "items": {
                "story_1": {
                    "documents": [
                        {
                            "content": "Chapter 1: The beginning.\nSomething happened.",
                            "doc_id": "story_1_doc1",
                        },
                        {
                            "content": "Chapter 2: The middle.\nMore events occurred.",
                            "doc_id": "story_1_doc2",
                        },
                    ]
                }
            },
            "collection_metadata": {"num_items": 1},
            "num_items_loaded": 1,
        }

        self.empty_dataset = {
            "items": {},
            "collection_metadata": {"num_items": 0},
            "num_items_loaded": 0,
        }

    @patch("ius.cli.chunk.load_data")
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

    @patch("ius.cli.chunk.load_data")
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

    @patch("ius.cli.chunk.load_data")
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

    @patch("ius.cli.chunk.load_data")
    def test_chunk_dataset_with_output(self, mock_load_data):
        """Test chunking with output file saving."""
        mock_load_data.return_value = self.test_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
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

    @patch("ius.cli.chunk.load_data")
    def test_chunk_dataset_empty_items(self, mock_load_data):
        """Test handling of empty or invalid items."""
        mock_dataset = {
            "items": {
                "empty": {"documents": [{"content": "", "doc_id": "empty"}]},
                "missing_documents": {"title": "No documents field"},
                "valid": {
                    "documents": [
                        {
                            "content": "Valid content here.\nWith multiple lines.\nFor proper chunking.",
                            "doc_id": "valid",
                        }
                    ]
                },
            },
            "collection_metadata": {"num_items": 3},
            "num_items_loaded": 3,
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

    @patch("ius.cli.chunk.load_data")
    def test_chunk_dataset_validation_failure(self, mock_load_data):
        """Test handling of validation failures."""
        mock_load_data.return_value = self.test_dataset

        # Mock chunking function to raise validation error
        with patch(
            "ius.chunk.chunkers.chunk_fixed_size",
            side_effect=ValueError("Content validation failed"),
        ):
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
            patch("sys.argv", ["ius", "--help"]),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
            self.assertRaises(SystemExit) as cm,
        ):
            main()
            self.assertEqual(cm.exception.code, 0)
            output = mock_stdout.getvalue()
            self.assertIn("Chunk documents for incremental summarization", output)

    def test_main_list_datasets(self):
        """Test listing available datasets."""
        test_args = ["ius", "--list-datasets"]
        with (
            patch("sys.argv", test_args),
            patch("ius.cli.chunk.list_datasets", return_value=["bmds", "test"]),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            main()
            output = mock_stdout.getvalue()
            self.assertIn("Available datasets:", output)
            self.assertIn("bmds", output)
            self.assertIn("test", output)

    @patch("ius.cli.chunk.list_datasets")
    @patch("ius.cli.chunk.chunk_dataset")
    def test_main_successful_run(self, mock_chunk_dataset, mock_list_datasets):
        """Test successful CLI execution."""
        mock_list_datasets.return_value = ["test"]
        mock_chunk_dataset.return_value = {"dataset": "test", "items": {"item1": {}}}

        test_args = [
            "ius",
            "--dataset",
            "test",
            "--strategy",
            "fixed_size",
            "--size",
            "1000",
        ]

        with (
            patch("sys.argv", test_args),
            patch("sys.stdout", new_callable=StringIO),
        ):
            main()  # Should not raise

        # Verify chunk_dataset was called with correct arguments
        mock_chunk_dataset.assert_called_once()
        call_args = mock_chunk_dataset.call_args
        self.assertEqual(call_args.kwargs["dataset_name"], "test")
        self.assertEqual(call_args.kwargs["strategy"], "fixed_size")
        self.assertEqual(call_args.kwargs["chunk_size"], 1000)

    def test_main_missing_required_args(self):
        """Test error handling for missing required arguments."""
        with (
            patch("sys.argv", ["ius", "--strategy", "fixed_size"]),
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as cm,
        ):
            main()
            self.assertNotEqual(cm.exception.code, 0)

    def test_main_invalid_strategy_args(self):
        """Test error handling for invalid strategy arguments."""
        # fixed_size without size
        with (
            patch("sys.argv", ["ius", "--dataset", "test", "--strategy", "fixed_size"]),
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as cm,
        ):
            main()
            self.assertNotEqual(cm.exception.code, 0)

        # fixed_count without count
        with (
            patch(
                "sys.argv", ["ius", "--dataset", "test", "--strategy", "fixed_count"]
            ),
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as cm,
        ):
            main()
            self.assertNotEqual(cm.exception.code, 0)

    @patch("ius.cli.chunk.list_datasets")
    def test_main_nonexistent_dataset(self, mock_list_datasets):
        """Test error handling for nonexistent dataset."""
        mock_list_datasets.return_value = ["bmds", "other"]

        test_args = [
            "ius",
            "--dataset",
            "nonexistent",
            "--strategy",
            "fixed_size",
            "--size",
            "1000",
        ]

        with (
            patch("sys.argv", test_args),
            patch("sys.stderr", new_callable=StringIO) as mock_stderr,
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            output_path = tmp.name

        try:
            with patch("sys.stdout", new_callable=StringIO):
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
        if "items" in dataset:
            items = dataset["items"]
            # Create a small test with just one item
            test_item_id = list(items.keys())[0]
            test_dataset = {
                "items": {test_item_id: items[test_item_id]},
                "collection_metadata": dataset.get("collection_metadata", {}),
                "num_items_loaded": 1,
            }

            # Mock load_data to return our test dataset
            with patch("ius.cli.chunk.load_data", return_value=test_dataset):
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


class TestCLIFlags(unittest.TestCase):
    """Test cases for CLI flags like --verbose and --dry-run."""

    def setUp(self):
        """Set up test fixtures."""
        self.original_argv = sys.argv.copy()

    def tearDown(self):
        """Clean up test fixtures."""
        sys.argv = self.original_argv

    @patch("ius.cli.chunk.setup_logging")
    def test_verbose_flag_enables_verbose_logging(self, mock_setup_logging):
        """Test that --verbose flag enables verbose logging."""
        sys.argv = ["test", "--verbose", "--list-datasets"]

        with (
            patch("ius.cli.chunk.list_datasets", return_value=["test"]),
            patch("ius.cli.chunk.logger"),
        ):
            main()

        # Should call setup_logging with verbose=True
        mock_setup_logging.assert_called_with(log_level="INFO", verbose=True)

    @patch("ius.cli.chunk.setup_logging")
    def test_short_verbose_flag_enables_verbose_logging(self, mock_setup_logging):
        """Test that -v flag enables verbose logging."""
        sys.argv = ["test", "-v", "--list-datasets"]

        with (
            patch("ius.cli.chunk.list_datasets", return_value=["test"]),
            patch("ius.cli.chunk.logger"),
        ):
            main()

        # Should call setup_logging with verbose=True
        mock_setup_logging.assert_called_with(log_level="INFO", verbose=True)

    @patch("ius.cli.chunk.setup_logging")
    def test_no_verbose_flag_uses_normal_logging(self, mock_setup_logging):
        """Test that normal mode uses non-verbose logging."""
        sys.argv = ["test", "--list-datasets"]

        with (
            patch("ius.cli.chunk.list_datasets", return_value=["test"]),
            patch("ius.cli.chunk.logger"),
        ):
            main()

        # Should call setup_logging with verbose=False
        mock_setup_logging.assert_called_with(log_level="INFO", verbose=False)

    @patch("ius.cli.chunk._load_and_validate_dataset")
    @patch("ius.cli.chunk.logger")
    def test_dry_run_flag_shows_preview_without_processing(
        self, mock_logger, mock_load_dataset
    ):
        """Test that --dry-run shows what would be processed without doing it."""
        # Mock dataset loading
        mock_dataset = {
            "items": {
                "item1": {"test": "data1"},
                "item2": {"test": "data2"},
                "item3": {"test": "data3"},
            }
        }
        mock_load_dataset.return_value = mock_dataset

        sys.argv = [
            "test",
            "--dataset",
            "test",
            "--strategy",
            "fixed_size",
            "--size",
            "1000",
            "--dry-run",
        ]

        # Should return without calling chunk_dataset
        with (
            patch("ius.cli.chunk.chunk_dataset") as mock_chunk,
            patch("ius.cli.chunk.list_datasets", return_value=["test"]),
        ):
            main()

            # chunk_dataset should NOT be called in dry-run mode
            mock_chunk.assert_not_called()

        # Should log dry-run messages
        mock_logger.info.assert_any_call("📋 Would process 3 items from dataset 'test'")
        mock_logger.info.assert_any_call("🔧 Would use chunking strategy: fixed_size")
        mock_logger.info.assert_any_call("🔧 Target chunk size: 1000 characters")
        mock_logger.info.assert_any_call(
            "✨ Dry run completed - no files were modified"
        )

    @patch("ius.cli.chunk._load_and_validate_dataset")
    @patch("ius.cli.chunk.logger")
    def test_dry_run_with_fixed_count_strategy(self, mock_logger, mock_load_dataset):
        """Test dry-run with fixed_count strategy shows correct parameters."""
        mock_dataset = {"items": {"item1": {"test": "data"}}}
        mock_load_dataset.return_value = mock_dataset

        sys.argv = [
            "test",
            "--dataset",
            "test",
            "--strategy",
            "fixed_count",
            "--count",
            "5",
            "--dry-run",
        ]

        with (
            patch("ius.cli.chunk.chunk_dataset") as mock_chunk,
            patch("ius.cli.chunk.list_datasets", return_value=["test"]),
        ):
            main()
            mock_chunk.assert_not_called()

        # Should show count-specific information
        mock_logger.info.assert_any_call("🔧 Target number of chunks: 5")

    @patch("ius.cli.chunk._load_and_validate_dataset")
    @patch("ius.cli.chunk.logger")
    def test_dry_run_shows_many_items_correctly(self, mock_logger, mock_load_dataset):
        """Test that dry-run handles many items correctly (shows first 5 + ellipsis)."""
        # Create dataset with more than 5 items
        mock_dataset = {"items": {f"item{i}": {"test": f"data{i}"} for i in range(10)}}
        mock_load_dataset.return_value = mock_dataset

        sys.argv = [
            "test",
            "--dataset",
            "test",
            "--strategy",
            "fixed_size",
            "--size",
            "1000",
            "--dry-run",
        ]

        with (
            patch("ius.cli.chunk.chunk_dataset") as mock_chunk,
            patch("ius.cli.chunk.list_datasets", return_value=["test"]),
        ):
            main()
            mock_chunk.assert_not_called()

        # Should show first 5 items with ellipsis
        items_call = None
        for call in mock_logger.info.call_args_list:
            if "📋 Items:" in str(call):
                items_call = str(call)
                break

        self.assertIsNotNone(items_call)
        self.assertIn("...", items_call)  # Should have ellipsis for many items

    @patch("ius.cli.chunk._load_and_validate_dataset")
    @patch("ius.cli.chunk.logger")
    @patch("sys.exit")
    def test_dry_run_handles_dataset_loading_failure(
        self, mock_exit, mock_logger, mock_load_dataset
    ):
        """Test that dry-run handles dataset loading failures gracefully."""
        mock_load_dataset.return_value = None  # Simulate loading failure

        sys.argv = [
            "test",
            "--dataset",
            "test",
            "--strategy",
            "fixed_size",
            "--size",
            "1000",
            "--dry-run",
        ]

        main()

        # Should log error and exit
        mock_logger.error.assert_any_call(
            "Cannot show dry run preview - dataset loading failed"
        )
        mock_exit.assert_called_with(1)

    def test_dry_run_and_verbose_work_together(self):
        """Test that --dry-run and --verbose flags can be used together."""
        sys.argv = [
            "test",
            "--dataset",
            "test",
            "--strategy",
            "fixed_size",
            "--size",
            "1000",
            "--dry-run",
            "--verbose",
        ]

        with (
            patch("ius.cli.chunk.setup_logging") as mock_setup,
            patch(
                "ius.cli.chunk._load_and_validate_dataset",
                return_value={"items": {"test": {}}},
            ),
            patch("ius.cli.chunk.logger"),
            patch("ius.cli.chunk.list_datasets", return_value=["test"]),
        ):
            main()

        # Should enable verbose logging
        mock_setup.assert_called_with(log_level="INFO", verbose=True)


class TestProgressBars(unittest.TestCase):
    """Test cases for tqdm progress bars."""

    def setUp(self):
        """Set up test fixtures."""
        self.original_argv = sys.argv.copy()

    def tearDown(self):
        """Clean up test fixtures."""
        sys.argv = self.original_argv

    @patch("ius.chunk.chunkers.tqdm")
    def test_progress_bar_enabled_for_multiple_items(self, mock_tqdm):
        """Test that progress bars appear when processing multiple items."""
        from ius.chunk.chunkers import process_dataset_items

        # Create mock items (multiple items should show progress)
        mock_items = {
            f"item{i}": {
                "documents": [
                    {"doc_id": f"doc{i}", "content": "Line 1\nLine 2\nLine 3"}
                ]
            }
            for i in range(5)
        }

        # Mock tqdm to just pass through the iterator
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

        # Process items
        process_dataset_items(items=mock_items, strategy="fixed_size", chunk_size=10)

        # Should have called tqdm for items progress
        mock_tqdm.assert_called()
        # Check that the first call (items progress) had disable=False for multiple items
        first_call = mock_tqdm.call_args_list[0]
        self.assertFalse(
            first_call.kwargs.get("disable", True)
        )  # Should not be disabled

    @patch("ius.chunk.chunkers.tqdm")
    def test_progress_bar_disabled_for_single_item(self, mock_tqdm):
        """Test that progress bars are disabled for single items."""
        from ius.chunk.chunkers import process_dataset_items

        # Create single item
        mock_items = {
            "item1": {
                "documents": [{"doc_id": "doc1", "content": "Line 1\nLine 2\nLine 3"}]
            }
        }

        # Mock tqdm to just pass through the iterator
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

        # Process items
        process_dataset_items(items=mock_items, strategy="fixed_size", chunk_size=10)

        # Should have called tqdm but with disable=True for single item
        mock_tqdm.assert_called()
        first_call = mock_tqdm.call_args_list[0]
        self.assertTrue(first_call.kwargs.get("disable", False))  # Should be disabled

    @patch("ius.chunk.chunkers.tqdm")
    def test_chunk_fixed_size_progress_bar_for_large_text(self, mock_tqdm):
        """Test that chunk_fixed_size shows progress for large texts."""
        from ius.chunk.chunkers import chunk_fixed_size

        # Create large text with many units (> 100 to trigger progress bar)
        large_text = "\n".join([f"Line {i}" for i in range(150)])

        # Mock tqdm to just pass through the iterator
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

        # Process chunking
        chunk_fixed_size(large_text, chunk_size=100, delimiter="\n")

        # Should have called tqdm for units progress
        mock_tqdm.assert_called()
        call_args = mock_tqdm.call_args
        self.assertFalse(
            call_args.kwargs.get("disable", True)
        )  # Should not be disabled
        self.assertEqual(call_args.kwargs.get("desc"), "Chunking text")

    @patch("ius.chunk.chunkers.tqdm")
    def test_chunk_fixed_size_no_progress_for_small_text(self, mock_tqdm):
        """Test that chunk_fixed_size doesn't show progress for small texts."""
        from ius.chunk.chunkers import chunk_fixed_size

        # Create small text (< 100 units)
        small_text = "\n".join([f"Line {i}" for i in range(10)])

        # Mock tqdm to just pass through the iterator
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

        # Process chunking
        chunk_fixed_size(small_text, chunk_size=50, delimiter="\n")

        # Should have called tqdm but with disable=True for small text
        mock_tqdm.assert_called()
        call_args = mock_tqdm.call_args
        self.assertTrue(call_args.kwargs.get("disable", False))  # Should be disabled

    @patch("ius.chunk.chunkers.tqdm")
    def test_chunk_fixed_count_progress_bar_for_many_chunks(self, mock_tqdm):
        """Test that chunk_fixed_count shows progress for many chunks."""
        from ius.chunk.chunkers import chunk_fixed_count

        # Create text and request many chunks (> 10 to trigger progress bar)
        text = "\n".join([f"Line {i}" for i in range(100)])
        num_chunks = 15

        # Mock tqdm to just pass through the iterator
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

        # Process chunking
        chunk_fixed_count(text, num_chunks=num_chunks, delimiter="\n")

        # Should have called tqdm for chunk creation progress
        mock_tqdm.assert_called()
        call_args = mock_tqdm.call_args
        self.assertFalse(
            call_args.kwargs.get("disable", True)
        )  # Should not be disabled
        self.assertEqual(call_args.kwargs.get("desc"), "Creating chunks")

    @patch("ius.chunk.chunkers.tqdm")
    def test_chunk_fixed_count_no_progress_for_few_chunks(self, mock_tqdm):
        """Test that chunk_fixed_count doesn't show progress for few chunks."""
        from ius.chunk.chunkers import chunk_fixed_count

        # Create text and request few chunks (< 10)
        text = "\n".join([f"Line {i}" for i in range(50)])
        num_chunks = 5

        # Mock tqdm to just pass through the iterator
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

        # Process chunking
        chunk_fixed_count(text, num_chunks=num_chunks, delimiter="\n")

        # Should have called tqdm but with disable=True for few chunks
        mock_tqdm.assert_called()
        call_args = mock_tqdm.call_args
        self.assertTrue(call_args.kwargs.get("disable", False))  # Should be disabled

    @patch("ius.chunk.chunkers.tqdm")
    def test_document_progress_bar_for_multiple_documents(self, mock_tqdm):
        """Test that document processing shows progress for multiple documents."""
        from ius.chunk.chunkers import process_dataset_items

        # Create item with multiple documents
        mock_items = {
            "item1": {
                "documents": [
                    {"doc_id": f"doc{i}", "content": "Line 1\nLine 2\nLine 3"}
                    for i in range(5)
                ]
            }
        }

        # Mock tqdm to just pass through the iterator for all calls
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

        # Process items
        process_dataset_items(items=mock_items, strategy="fixed_size", chunk_size=10)

        # Should have called tqdm multiple times (items + documents progress bars)
        self.assertGreater(mock_tqdm.call_count, 1)


if __name__ == "__main__":
    unittest.main()
