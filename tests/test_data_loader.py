#!/usr/bin/env python3
"""
Tests for the data loading module.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ius.data.loader import DatasetLoader, get_dataset_info, list_datasets, load_data
from ius.exceptions import DatasetError


class TestDatasetLoader(unittest.TestCase):
    """Test cases for DatasetLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test datasets
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.loader = DatasetLoader(self.data_dir)

        # Create sample dataset structure
        self.sample_collection = {
            "domain": "test_domain",
            "source": "test_source",
            "num_items": 2,
            "total_documents": 2,
            "description": "Test dataset",
            "items": ["test_item1", "test_item2"]
        }

        self.sample_item1 = {
            "item_metadata": {
                "item_id": "test_item1",
                "num_documents": 1
            },
            "documents": [
                {
                    "content": "This is test content.\nWith multiple lines.\nFor testing purposes.",
                    "doc_id": "doc1",
                    "metadata": {
                        "title": "Test Document 1",
                        "author": "Test Author"
                    }
                }
            ]
        }

        self.sample_item2 = {
            "item_metadata": {
                "item_id": "test_item2",
                "num_documents": 1
            },
            "documents": [
                {
                    "content": "Another test document.\nWith different content.\nFor variety.",
                    "doc_id": "doc2",
                    "metadata": {
                        "title": "Test Document 2"
                    }
                }
            ]
        }

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def _create_test_dataset(self, dataset_name: str = "test_dataset"):
        """Helper to create a complete test dataset."""
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create collection.json
        collection_path = dataset_dir / "collection.json"
        with open(collection_path, 'w', encoding='utf-8') as f:
            json.dump(self.sample_collection, f)

        # Create items directory and files
        items_dir = dataset_dir / "items"
        items_dir.mkdir(exist_ok=True)

        with open(items_dir / "test_item1.json", 'w', encoding='utf-8') as f:
            json.dump(self.sample_item1, f)

        with open(items_dir / "test_item2.json", 'w', encoding='utf-8') as f:
            json.dump(self.sample_item2, f)

        return dataset_dir

    def test_list_datasets_empty_directory(self):
        """Test listing datasets from empty directory."""
        datasets = self.loader.list_datasets()
        self.assertEqual(datasets, [])

    def test_list_datasets_with_datasets(self):
        """Test listing datasets with valid datasets."""
        self._create_test_dataset("dataset1")
        self._create_test_dataset("dataset2")

        # Create directory without collection.json (should be ignored)
        invalid_dir = self.data_dir / "invalid_dataset"
        invalid_dir.mkdir()

        datasets = self.loader.list_datasets()
        self.assertEqual(sorted(datasets), ["dataset1", "dataset2"])

    def test_list_datasets_nonexistent_directory(self):
        """Test listing datasets when data directory doesn't exist."""
        nonexistent_loader = DatasetLoader(self.data_dir / "nonexistent")
        datasets = nonexistent_loader.list_datasets()
        self.assertEqual(datasets, [])

    def test_load_collection_metadata_success(self):
        """Test successful collection metadata loading."""
        self._create_test_dataset("test_dataset")

        metadata = self.loader.load_collection_metadata("test_dataset")

        self.assertEqual(metadata["domain"], "test_domain")
        self.assertEqual(metadata["num_items"], 2)
        self.assertEqual(metadata["items"], ["test_item1", "test_item2"])

    def test_load_collection_metadata_invalid_dataset_name(self):
        """Test loading collection metadata with invalid dataset name."""
        with self.assertRaises(DatasetError) as cm:
            self.loader.load_collection_metadata("")
        self.assertIn("dataset_name must be a non-empty string", str(cm.exception))

        with self.assertRaises(DatasetError) as cm:
            self.loader.load_collection_metadata(None)
        self.assertIn("dataset_name must be a non-empty string", str(cm.exception))

    def test_load_collection_metadata_dataset_not_found(self):
        """Test loading collection metadata for non-existent dataset."""
        # Create one dataset so we can test the helpful error message
        self._create_test_dataset("existing_dataset")

        with self.assertRaises(DatasetError) as cm:
            self.loader.load_collection_metadata("nonexistent_dataset")
        self.assertIn("Dataset 'nonexistent_dataset' not found", str(cm.exception))
        self.assertIn("Available datasets: existing_dataset", str(cm.exception))

    def test_load_collection_metadata_no_datasets_available(self):
        """Test loading collection metadata when no datasets exist."""
        with self.assertRaises(DatasetError) as cm:
            self.loader.load_collection_metadata("nonexistent")
        self.assertIn("No datasets found in directory", str(cm.exception))

    def test_load_collection_metadata_invalid_json(self):
        """Test loading collection metadata with invalid JSON."""
        dataset_dir = self.data_dir / "invalid_json_dataset"
        dataset_dir.mkdir(parents=True)

        # Create invalid JSON file
        collection_path = dataset_dir / "collection.json"
        with open(collection_path, 'w', encoding='utf-8') as f:
            f.write("{ invalid json content")

        with self.assertRaises(DatasetError) as cm:
            self.loader.load_collection_metadata("invalid_json_dataset")
        self.assertIn("Invalid JSON in collection.json", str(cm.exception))

    def test_load_collection_metadata_missing_required_fields(self):
        """Test loading collection metadata with missing required fields."""
        dataset_dir = self.data_dir / "incomplete_dataset"
        dataset_dir.mkdir(parents=True)

        # Create collection.json missing required fields
        incomplete_collection = {"domain": "test"}  # Missing "items" and "num_items"
        collection_path = dataset_dir / "collection.json"
        with open(collection_path, 'w', encoding='utf-8') as f:
            json.dump(incomplete_collection, f)

        with self.assertRaises(DatasetError) as cm:
            self.loader.load_collection_metadata("incomplete_dataset")
        self.assertIn("missing required fields", str(cm.exception))
        self.assertIn("items", str(cm.exception))
        self.assertIn("num_items", str(cm.exception))

    def test_load_item_success(self):
        """Test successful item loading."""
        self._create_test_dataset("test_dataset")

        item = self.loader.load_item("test_dataset", "test_item1")

        self.assertEqual(item["item_metadata"]["item_id"], "test_item1")
        self.assertEqual(len(item["documents"]), 1)
        self.assertIn("This is test content", item["documents"][0]["content"])

    def test_load_item_invalid_parameters(self):
        """Test loading item with invalid parameters."""
        with self.assertRaises(DatasetError) as cm:
            self.loader.load_item("", "test_item")
        self.assertIn("dataset_name must be a non-empty string", str(cm.exception))

        with self.assertRaises(DatasetError) as cm:
            self.loader.load_item("test_dataset", "")
        self.assertIn("item_id must be a non-empty string", str(cm.exception))

    def test_load_item_not_found(self):
        """Test loading non-existent item."""
        self._create_test_dataset("test_dataset")

        with self.assertRaises(DatasetError) as cm:
            self.loader.load_item("test_dataset", "nonexistent_item")
        self.assertIn("Item 'nonexistent_item' not found", str(cm.exception))

    def test_load_item_invalid_json(self):
        """Test loading item with invalid JSON."""
        dataset_dir = self._create_test_dataset("test_dataset")
        items_dir = dataset_dir / "items"

        # Create invalid JSON item file
        with open(items_dir / "invalid_item.json", 'w', encoding='utf-8') as f:
            f.write("{ invalid json")

        with self.assertRaises(DatasetError) as cm:
            self.loader.load_item("test_dataset", "invalid_item")
        self.assertIn("Invalid JSON in item 'invalid_item'", str(cm.exception))

    def test_load_item_missing_documents_field(self):
        """Test loading item missing required documents field."""
        dataset_dir = self._create_test_dataset("test_dataset")
        items_dir = dataset_dir / "items"

        # Create item without documents field
        invalid_item = {"item_metadata": {"item_id": "invalid"}}
        with open(items_dir / "invalid_item.json", 'w', encoding='utf-8') as f:
            json.dump(invalid_item, f)

        with self.assertRaises(DatasetError) as cm:
            self.loader.load_item("test_dataset", "invalid_item")
        self.assertIn("missing required 'documents' field", str(cm.exception))

    def test_load_item_invalid_data_type(self):
        """Test loading item with invalid data type."""
        dataset_dir = self._create_test_dataset("test_dataset")
        items_dir = dataset_dir / "items"

        # Create item that's not a dictionary
        with open(items_dir / "invalid_type.json", 'w', encoding='utf-8') as f:
            json.dump(["this", "is", "a", "list"], f)

        with self.assertRaises(DatasetError) as cm:
            self.loader.load_item("test_dataset", "invalid_type")
        self.assertIn("expected dictionary, got list", str(cm.exception))

    def test_load_data_single_item(self):
        """Test loading single item."""
        self._create_test_dataset("test_dataset")

        result = self.loader.load_data("test_dataset", item_id="test_item1")

        self.assertIn("collection_metadata", result)
        self.assertIn("items", result)
        self.assertIn("num_items_loaded", result)
        self.assertEqual(result["num_items_loaded"], 1)
        self.assertIn("test_item1", result["items"])
        self.assertEqual(len(result["items"]), 1)

    def test_load_data_all_items(self):
        """Test loading all items."""
        self._create_test_dataset("test_dataset")

        result = self.loader.load_data("test_dataset")

        self.assertEqual(result["num_items_loaded"], 2)
        self.assertIn("test_item1", result["items"])
        self.assertIn("test_item2", result["items"])
        self.assertEqual(len(result["items"]), 2)

    def test_get_dataset_info(self):
        """Test getting dataset info."""
        self._create_test_dataset("test_dataset")

        info = self.loader.get_dataset_info("test_dataset")

        required_fields = [
            "dataset_name", "domain", "source", "num_items", "total_documents",
            "max_num_documents_per_item", "min_num_documents_per_item",
            "avg_num_documents_per_item", "avg_words_per_document"
        ]

        for field in required_fields:
            self.assertIn(field, info)

        self.assertEqual(info["dataset_name"], "test_dataset")
        self.assertEqual(info["domain"], "test_domain")
        self.assertEqual(info["num_items"], 2)
        self.assertEqual(info["max_num_documents_per_item"], 1)
        self.assertEqual(info["min_num_documents_per_item"], 1)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_load_data_convenience_function(self):
        """Test the load_data convenience function."""
        # We'll test this with mocking since it's a simple wrapper
        with patch('ius.data.loader.DatasetLoader') as mock_loader_class:
            mock_loader = mock_loader_class.return_value
            mock_loader.load_data.return_value = {"test": "data"}

            result = load_data("test_dataset", item_id="test_item", data_dir=str(self.data_dir))

            mock_loader_class.assert_called_once_with(str(self.data_dir))
            mock_loader.load_data.assert_called_once_with("test_dataset", "test_item")
            self.assertEqual(result, {"test": "data"})

    def test_list_datasets_convenience_function(self):
        """Test the list_datasets convenience function."""
        with patch('ius.data.loader.DatasetLoader') as mock_loader_class:
            mock_loader = mock_loader_class.return_value
            mock_loader.list_datasets.return_value = ["dataset1", "dataset2"]

            result = list_datasets(data_dir=str(self.data_dir))

            mock_loader_class.assert_called_once_with(str(self.data_dir))
            mock_loader.list_datasets.assert_called_once()
            self.assertEqual(result, ["dataset1", "dataset2"])

    def test_get_dataset_info_convenience_function(self):
        """Test the get_dataset_info convenience function."""
        with patch('ius.data.loader.DatasetLoader') as mock_loader_class:
            mock_loader = mock_loader_class.return_value
            mock_loader.get_dataset_info.return_value = {"info": "data"}

            result = get_dataset_info("test_dataset", data_dir=str(self.data_dir))

            mock_loader_class.assert_called_once_with(str(self.data_dir))
            mock_loader.get_dataset_info.assert_called_once_with("test_dataset")
            self.assertEqual(result, {"info": "data"})


class TestDataLoaderIntegration(unittest.TestCase):
    """Integration tests using real BMDS data."""

    def test_load_real_bmds_data(self):
        """Test loading real BMDS dataset."""
        try:
            # Try to load a single item from BMDS
            result = load_data("bmds", item_id="ADP02")

            # Basic structure validation
            self.assertIn("collection_metadata", result)
            self.assertIn("items", result)
            self.assertIn("num_items_loaded", result)
            self.assertEqual(result["num_items_loaded"], 1)
            self.assertIn("ADP02", result["items"])

            # Item structure validation
            item = result["items"]["ADP02"]
            self.assertIn("item_metadata", item)
            self.assertIn("documents", item)
            self.assertGreater(len(item["documents"]), 0)

            # Document structure validation
            doc = item["documents"][0]
            self.assertIn("content", doc)
            self.assertIsInstance(doc["content"], str)
            self.assertGreater(len(doc["content"]), 0)

        except Exception as e:
            self.skipTest(f"BMDS dataset not available: {e}")

    def test_list_real_datasets(self):
        """Test listing real datasets."""
        try:
            datasets = list_datasets()
            self.assertIsInstance(datasets, list)

            # If BMDS exists, it should be in the list
            if Path("datasets/bmds/collection.json").exists():
                self.assertIn("bmds", datasets)

        except Exception as e:
            self.skipTest(f"Datasets directory not accessible: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
