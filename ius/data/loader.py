"""
Data loader for common dataset format.

Loads datasets from the standardized format with collection.json + items/ structure.
"""

import json
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import numpy as np


class DatasetLoader:
    """Loader for datasets in common format."""
    
    def __init__(self, data_dir: Union[str, Path] = "datasets"):
        """
        Initialize loader.
        
        Args:
            data_dir: Base directory containing datasets
        """
        self.data_dir = Path(data_dir)
    
    def list_datasets(self) -> List[str]:
        """List available datasets."""
        if not self.data_dir.exists():
            return []
        
        datasets = []
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                collection_file = subdir / "collection.json"
                if collection_file.exists():
                    datasets.append(subdir.name)
        
        return sorted(datasets)
    
    def load_collection_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load collection metadata.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Collection metadata dict
        """
        collection_path = self.data_dir / dataset_name / "collection.json"
        
        if not collection_path.exists():
            raise FileNotFoundError(f"Collection file not found: {collection_path}")
        
        with open(collection_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_item(self, dataset_name: str, item_id: str) -> Dict[str, Any]:
        """
        Load a single item.
        
        Args:
            dataset_name: Name of the dataset
            item_id: ID of the item to load
        
        Returns:
            Item data dict
        """
        item_path = self.data_dir / dataset_name / "items" / f"{item_id}.json"
        
        if not item_path.exists():
            raise FileNotFoundError(f"Item file not found: {item_path}")
        
        with open(item_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_data(self, dataset_name: str, item_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load dataset with collection metadata and items.
        
        Args:
            dataset_name: Name of the dataset (e.g., "bmds")
            item_id: Optional specific item ID to load. If None, loads all items.
        
        Returns:
            Dict with:
            - collection_metadata: metadata from collection.json
            - items: dict mapping item_id -> item_data
            - num_items_loaded: number of items actually loaded
        """
        # Load collection metadata
        collection_metadata = self.load_collection_metadata(dataset_name)
        
        # Load items
        if item_id is not None:
            # Load single item
            items = {item_id: self.load_item(dataset_name, item_id)}
        else:
            # Load all items
            item_ids = collection_metadata["items"]
            items = {}
            for item_id in item_ids:
                try:
                    items[item_id] = self.load_item(dataset_name, item_id)
                except FileNotFoundError as e:
                    print(f"Warning: Could not load item {item_id}: {e}")
        
        return {
            "collection_metadata": collection_metadata,
            "items": items,
            "num_items_loaded": len(items)
        }
    
   
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get summary information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Summary info dict
        """
        collection_metadata = self.load_collection_metadata(dataset_name)
        data = self.load_data(dataset_name)
        items = data["items"]
        num_documents_per_item = [len(item["documents"]) for item in items.values()]
        max_num_documents = max(num_documents_per_item)
        min_num_documents = min(num_documents_per_item)
        avg_num_documents = int(np.mean(num_documents_per_item))

        avg_words_per_document = int(np.mean([len(document["content"].split()) for item in items.values() for document in item["documents"]]))

        return {
            "dataset_name": dataset_name,
            "domain": collection_metadata.get("domain"),
            "source": collection_metadata.get("source"),
            "num_items": collection_metadata.get("num_items"),
            "total_documents": collection_metadata.get("total_documents"),
            "created": collection_metadata.get("created"),
            "description": collection_metadata.get("description"),
            "max_num_documents_per_item": max_num_documents,
            "min_num_documents_per_item": min_num_documents,
            "avg_num_documents_per_item": avg_num_documents,
            "avg_words_per_document": avg_words_per_document,
        }


# Convenience function matching the requested API
def load_data(dataset_name: str, item_id: Optional[str] = None, data_dir: Union[str, Path] = "datasets") -> Dict[str, Any]:
    """
    Convenience function to load dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., "bmds")
        item_id: Optional specific item ID to load. If None, loads all items.
        data_dir: Base directory containing datasets
    
    Returns:
        Dict with collection_metadata, items, and num_items_loaded
    """
    loader = DatasetLoader(data_dir)
    return loader.load_data(dataset_name, item_id)


# Additional convenience functions
def list_datasets(data_dir: Union[str, Path] = "datasets") -> List[str]:
    """List available datasets."""
    loader = DatasetLoader(data_dir)
    return loader.list_datasets()


def get_dataset_info(dataset_name: str, data_dir: Union[str, Path] = "datasets") -> Dict[str, Any]:
    """Get dataset summary info."""
    loader = DatasetLoader(data_dir)
    return loader.get_dataset_info(dataset_name) 