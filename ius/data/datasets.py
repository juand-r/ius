"""
Unified data collection classes for the IUS pipeline.

Provides clean, structured access to datasets, chunked data, and summaries
following the pipeline: Dataset -> ChunkedDataset -> SummaryDataset

All classes inherit from BaseDataCollection to follow DRY principles.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..exceptions import DatasetError, ValidationError
from ..logging_config import get_logger

logger = get_logger(__name__)


class BaseDataCollection:
    """
    Base class for all data collections in the IUS pipeline.
    
    Provides common functionality for path handling, JSON loading, and item management.
    Subclasses override methods for specific data formats (original, chunked, summarized).
    """
    
    def __init__(self, collection_path: str):
        """
        Initialize collection from a directory path.
        
        Args:
            collection_path: Path to collection directory
        
        Raises:
            DatasetError: If collection cannot be loaded or is invalid
        """
        self.collection_path = Path(collection_path)
        self._collection_metadata = None
        self._item_ids = None
        
        # Validate and initialize collection
        self._validate_and_load()
    
    def _validate_and_load(self) -> None:
        """Validate collection structure and load metadata. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _validate_and_load()")
    
    def _load_json_file(self, file_path: Path, required: bool = True) -> Dict[str, Any]:
        """
        Load and parse JSON file with error handling.
        
        Args:
            file_path: Path to JSON file
            required: Whether file is required to exist
            
        Returns:
            Parsed JSON data, empty dict if file doesn't exist and not required
            
        Raises:
            DatasetError: If file cannot be loaded or parsed
        """
        if not file_path.exists():
            if required:
                raise DatasetError(f"Required file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON in file {file_path}: {e}")
        except Exception as e:
            raise DatasetError(f"Error loading file {file_path}: {e}")
    
    def _validate_item_structure(self, item_data: Dict[str, Any], item_id: str, 
                                required_field: str) -> None:
        """
        Validate that item has required structure.
        
        Args:
            item_data: Item data dictionary
            item_id: Item identifier for error messages
            required_field: Field that must be present (e.g., "documents", "chunks", "summaries")
            
        Raises:
            DatasetError: If item structure is invalid
        """
        if required_field not in item_data:
            raise DatasetError(f"Item '{item_id}' missing required '{required_field}' field")
    
    @property
    def name(self) -> str:
        """Returns the collection name (derived from directory name)."""
        return self.collection_path.name
    
    @property
    def item_ids(self) -> List[str]:
        """Returns list of item IDs available in this collection."""
        return self._item_ids.copy() if self._item_ids else []
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns collection metadata if available."""
        return self._collection_metadata.copy() if self._collection_metadata else {}
    
    def load_item(self, item_id: str) -> Dict[str, Any]:
        """
        Load individual item JSON file from items/ directory.
        
        Args:
            item_id: ID of the item to load
            
        Returns:
            Full item structure with item_metadata and documents
            
        Raises:
            DatasetError: If item cannot be loaded or doesn't exist
        """
        if item_id not in self._item_ids:
            raise DatasetError(f"Item '{item_id}' not found in collection '{self.name}'")
        
        item_file = self.collection_path / "items" / f"{item_id}.json"
        item_data = self._load_json_file(item_file, required=True)
        self._validate_item_structure(item_data, item_id, "documents")
        
        return item_data
    
    def __len__(self) -> int:
        """Returns number of items in collection."""
        return len(self._item_ids) if self._item_ids else 0
    
    def __contains__(self, item_id: str) -> bool:
        """Check if item exists in collection."""
        return item_id in (self._item_ids or [])


class Dataset(BaseDataCollection):
    """
    Represents an original dataset with collection metadata and individual items.
    
    Provides clean access to dataset structure, item loading, and document extraction.
    """
    
    def _validate_and_load(self) -> None:
        """Load and validate original dataset structure."""
        collection_file = self.collection_path / "collection.json"
        self._collection_metadata = self._load_json_file(collection_file, required=True)
        
        # Validate required fields
        if "items" not in self._collection_metadata:
            raise DatasetError("Collection metadata missing required 'items' field")
        
        self._item_ids = self._collection_metadata["items"]
        logger.debug(f"Loaded dataset with {len(self._item_ids)} items")
    

    
    def get_item_documents(self, item_id: str) -> List[Dict[str, Any]]:
        """
        Get just the documents array for an item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            List of documents, each with "content" and "metadata"
        """
        item_data = self.load_item(item_id)
        return item_data["documents"]
    
    def __repr__(self) -> str:
        return f"Dataset(name='{self.name}', items={len(self)})"


class ChunkedDataset(BaseDataCollection):
    """
    Represents a chunked dataset, providing clean access to text chunks.
    
    Handles the chunked format where each item is saved as a separate JSON file
    with the same structure as original items but with "chunks" instead of "content".
    """
    
    def _validate_and_load(self) -> None:
        """Discover and validate chunked items in the directory."""
        if not self.collection_path.exists():
            raise DatasetError(f"Chunked directory not found: {self.collection_path}")
        
        if not self.collection_path.is_dir():
            raise DatasetError(f"Path is not a directory: {self.collection_path}")
        
        # Look for items in the items/ subdirectory
        items_dir = self.collection_path / "items"
        if not items_dir.exists() or not items_dir.is_dir():
            raise DatasetError(f"Items subdirectory not found: {items_dir}")
        
        json_files = list(items_dir.glob("*.json"))
        if not json_files:
            raise DatasetError(f"No chunked items found in: {items_dir}")
        
        # Extract item IDs from filenames
        self._item_ids = [json_file.stem for json_file in json_files]
        self._item_ids.sort()  # Keep consistent ordering
        
        # Load collection metadata if available (optional for chunked datasets)
        collection_file = self.collection_path / "collection.json"
        self._collection_metadata = self._load_json_file(collection_file, required=False)
        
        logger.debug(f"Discovered {len(self._item_ids)} chunked items")
    

    
    # Alias for backward compatibility
    def load_chunked_item(self, item_id: str) -> Dict[str, Any]:
        """Alias for load_item() to maintain backward compatibility."""
        return self.load_item(item_id)
    
    def get_item_chunks(self, item_id: str) -> List[str]:
        """
        Get all chunks for a specific item (flattened from all documents).
        
        Args:
            item_id: ID of the item
            
        Returns:
            List of text chunks (strings)
        """
        chunked_item = self.load_item(item_id)
        
        # Flatten chunks from all documents
        all_chunks = []
        for document in chunked_item["documents"]:
            all_chunks.extend(document["chunks"])
        
        return all_chunks
    
    def get_document_chunks(self, item_id: str, doc_index: int) -> List[str]:
        """
        Get chunks from a specific document within an item.
        
        Args:
            item_id: ID of the item
            doc_index: Index of the document (0-based)
            
        Returns:
            List of text chunks from the specified document
            
        Raises:
            ValidationError: If document index is invalid
        """
        chunked_item = self.load_item(item_id)
        documents = chunked_item["documents"]
        
        if doc_index < 0 or doc_index >= len(documents):
            raise ValidationError(f"Document index {doc_index} out of range for item '{item_id}' (has {len(documents)} documents)")
        
        return documents[doc_index]["chunks"]
    
    def get_item_stats(self, item_id: str) -> Dict[str, Any]:
        """
        Get chunking statistics for an item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            Dictionary with item-specific chunking statistics
        """
        chunked_item = self.load_item(item_id)
        
        # Calculate item-specific stats from document chunking stats
        documents = chunked_item.get("documents", [])
        if not documents:
            return {}
        
        total_chunks = sum(len(doc.get("chunks", [])) for doc in documents) 
        total_chars = sum(
            doc.get("metadata", {}).get("chunking_stats", {}).get("total_chars", 0) 
            for doc in documents
        )
        
        if total_chunks > 0:
            avg_chunk_size = total_chars / total_chunks
        else:
            avg_chunk_size = 0
            
        return {
            "num_documents": len(documents),
            "total_chunks": total_chunks,
            "total_chars": total_chars,
            "avg_chunk_size": round(avg_chunk_size, 1),
            "chunking_method": chunked_item.get("item_metadata", {}).get("chunking_method", "unknown"),
            "chunking_params": chunked_item.get("item_metadata", {}).get("chunking_params", {}),
        }
    
    def get_chunks_for_scope(self, scope: str, **kwargs) -> List[str]:
        """
        Main method for scope-based chunk extraction.
        
        Args:
            scope: Type of scope ("item", "dataset", "doc_range")
            **kwargs: Additional arguments based on scope:
                - scope="item": requires item_id
                - scope="dataset": no additional args (returns chunks from all items)
                - scope="doc_range": requires item_id and doc_range (e.g., "0-2" or "1")
        
        Returns:
            List of text chunks for the specified scope
            
        Raises:
            ValidationError: If scope or arguments are invalid
        """
        if scope == "item":
            item_id = kwargs.get("item_id")
            if not item_id:
                raise ValidationError("scope='item' requires 'item_id' argument")
            return self.get_item_chunks(item_id)
        
        elif scope == "dataset":
            # Return chunks from all items
            all_chunks = []
            for item_id in self._item_ids:
                all_chunks.extend(self.get_item_chunks(item_id))
            return all_chunks
        
        elif scope == "doc_range":
            item_id = kwargs.get("item_id")
            doc_range = kwargs.get("doc_range")
            
            if not item_id:
                raise ValidationError("scope='doc_range' requires 'item_id' argument")
            if not doc_range:
                raise ValidationError("scope='doc_range' requires 'doc_range' argument")
            
            return self._get_doc_range_chunks(item_id, doc_range)
        
        else:
            raise ValidationError(f"Invalid scope: '{scope}'. Must be one of: 'item', 'dataset', 'doc_range'")
    
    def _get_doc_range_chunks(self, item_id: str, doc_range: str) -> List[str]:
        """
        Get chunks from a document range within an item.
        
        Args:
            item_id: ID of the item
            doc_range: Document range specification (e.g., "0-2" or "1")
        
        Returns:
            List of text chunks from the specified document range
        """
        chunked_item = self.load_item(item_id)
        documents = chunked_item["documents"]
        total_docs = len(documents)
        
        # Parse document range
        doc_indices = self._parse_doc_range(doc_range, total_docs)
        
        # Collect chunks from specified documents
        range_chunks = []
        for doc_index in doc_indices:
            range_chunks.extend(documents[doc_index]["chunks"])
        
        return range_chunks
    
    def _parse_doc_range(self, doc_range: str, total_docs: int) -> List[int]:
        """
        Parse document range string into list of document indices.
        
        Args:
            doc_range: Range specification (e.g., "0-2", "1", "0,2,3")
            total_docs: Total number of documents available
        
        Returns:
            List of document indices
            
        Raises:
            ValidationError: If range is invalid
        """
        try:
            if "-" in doc_range:
                # Range format: "0-2"
                start_str, end_str = doc_range.split("-", 1)
                start = int(start_str)
                end = int(end_str)
                
                if start < 0 or end >= total_docs or start > end:
                    raise ValidationError(f"Invalid document range '{doc_range}' for {total_docs} documents")
                
                return list(range(start, end + 1))
            
            elif "," in doc_range:
                # Comma-separated format: "0,2,3"
                indices = [int(idx.strip()) for idx in doc_range.split(",")]
                
                for idx in indices:
                    if idx < 0 or idx >= total_docs:
                        raise ValidationError(f"Document index {idx} out of range for {total_docs} documents")
                
                return sorted(set(indices))  # Remove duplicates and sort
            
            else:
                # Single document: "1"
                idx = int(doc_range)
                
                if idx < 0 or idx >= total_docs:
                    raise ValidationError(f"Document index {idx} out of range for {total_docs} documents")
                
                return [idx]
        
        except ValueError as e:
            raise ValidationError(f"Invalid document range format '{doc_range}': {e}")
    
    def __repr__(self) -> str:
        return f"ChunkedDataset(name='{self.name}', items={len(self)})"


class SummaryDataset(BaseDataCollection):
    """
    Represents a summary dataset, providing clean access to text summaries.
    
    Uses the unified structure: collection.json + items/{item_id}.json
    where each item contains documents with "summaries" arrays.
    """
    
    def _validate_and_load(self) -> None:
        """Validate and load summary collection structure (unified format)."""
        if not self.collection_path.exists():
            raise DatasetError(f"Summary directory not found: {self.collection_path}")
        
        if not self.collection_path.is_dir():
            raise DatasetError(f"Path is not a directory: {self.collection_path}")
        
        # Check for items directory (unified structure)
        items_dir = self.collection_path / "items"
        if not items_dir.exists() or not items_dir.is_dir():
            raise DatasetError(f"Items directory not found: {items_dir}")
        
        # Find all summary items (JSON files in items/ directory)
        item_files = list(items_dir.glob("*.json"))
        if not item_files:
            raise DatasetError(f"No summary items found in: {items_dir}")
        
        # Extract item IDs from filenames
        self._item_ids = [item_file.stem for item_file in item_files]
        self._item_ids.sort()  # Keep consistent ordering
        
        # Load collection metadata (prefer collection.json, fallback to config.json for backward compatibility)
        collection_file = self.collection_path / "collection.json"
        config_file = self.collection_path / "config.json"
        
        if collection_file.exists():
            self._collection_metadata = self._load_json_file(collection_file, required=True)
        elif config_file.exists():
            self._collection_metadata = self._load_json_file(config_file, required=True)
        else:
            raise DatasetError(f"No collection.json or config.json found in {self.collection_path}")
        
        logger.debug(f"Discovered {len(self._item_ids)} summary items")
    

    
    def get_item_summaries(self, item_id: str) -> List[str]:
        """
        Get all summaries for a specific item (flattened from all documents).
        
        Args:
            item_id: ID of the item
            
        Returns:
            List of summary texts (strings)
        """
        summary_item = self.load_item(item_id)
        
        # Flatten summaries from all documents (mirrors get_item_chunks from ChunkedDataset)
        all_summaries = []
        for document in summary_item["documents"]:
            if "summaries" in document:
                all_summaries.extend(document["summaries"])
        
        return all_summaries
    
    def get_item_stats(self, item_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for an item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            Dictionary with item-specific summary statistics
        """
        summary_item = self.load_item(item_id)
        
        # Calculate item-specific stats from document summary stats
        documents = summary_item.get("documents", [])
        if not documents:
            return {}
        
        total_summaries = sum(len(doc.get("summaries", [])) for doc in documents)
        total_chars = sum(
            doc.get("metadata", {}).get("summary_stats", {}).get("summary_length_chars", 0)
            for doc in documents
        )
        
        return {
            "num_documents": len(documents),
            "total_summaries": total_summaries,
            "total_chars": total_chars,
            "experiment_id": summary_item["item_metadata"]["experiment_id"],
            "strategy": summary_item["item_metadata"]["strategy"],
            "model": summary_item["item_metadata"]["model"],
        }
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """
        Get the full experiment configuration.
        
        Returns:
            Dictionary with experiment configuration
        """
        return self.metadata
    
    def __repr__(self) -> str:
        return f"SummaryDataset(name='{self.name}', items={len(self)})"


# Utility functions
def list_datasets(data_dir: Union[str, Path] = "datasets") -> List[str]:
    """
    List available datasets by looking for directories containing collection.json.
    
    Args:
        data_dir: Base directory containing datasets
        
    Returns:
        List of dataset names (directory names)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    datasets = []
    for subdir in data_path.iterdir():
        if subdir.is_dir():
            collection_file = subdir / "collection.json"
            if collection_file.exists():
                datasets.append(subdir.name)

    return sorted(datasets)