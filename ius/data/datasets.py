"""
Dataset and ChunkedDataset classes for clean data access.

These classes provide clean, structured access to original datasets and chunked data,
abstracting away the complexity of nested JSON structures and file operations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..exceptions import DatasetError, ValidationError
from ..logging_config import get_logger

logger = get_logger(__name__)


class Dataset:
    """
    Represents an original dataset with collection metadata and individual items.
    
    Provides clean access to dataset structure, item loading, and document extraction.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize Dataset from a dataset directory path.
        
        Args:
            dataset_path: Path to dataset directory (e.g., "datasets/bmds/")
        
        Raises:
            DatasetError: If dataset cannot be loaded or is invalid
        """
        self.dataset_path = Path(dataset_path)
        self._collection_metadata = None
        self._item_ids = None
        
        # Validate and load collection metadata
        self._load_collection_metadata()
    
    def _load_collection_metadata(self) -> None:
        """Load and validate collection.json metadata."""
        collection_file = self.dataset_path / "collection.json"
        
        if not collection_file.exists():
            raise DatasetError(f"Collection file not found: {collection_file}")
        
        try:
            with open(collection_file, 'r', encoding='utf-8') as f:
                self._collection_metadata = json.load(f)
            
            # Validate required fields
            if "items" not in self._collection_metadata:
                raise DatasetError("Collection metadata missing required 'items' field")
            
            self._item_ids = self._collection_metadata["items"]
            logger.debug(f"Loaded dataset with {len(self._item_ids)} items")
            
        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON in collection file: {e}")
        except Exception as e:
            raise DatasetError(f"Error loading collection metadata: {e}")
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns collection metadata (domain, source, description, etc.)."""
        return self._collection_metadata.copy()
    
    @property
    def item_ids(self) -> List[str]:
        """Returns list of item IDs from collection.json."""
        return self._item_ids.copy()
    
    @property
    def name(self) -> str:
        """Returns the dataset name (derived from directory name)."""
        return self.dataset_path.name
    
    def load_item(self, item_id: str) -> Dict[str, Any]:
        """
        Load individual item JSON file.
        
        Args:
            item_id: ID of the item to load
            
        Returns:
            Full item structure with item_metadata and documents
            
        Raises:
            DatasetError: If item cannot be loaded or doesn't exist
        """
        if item_id not in self._item_ids:
            raise DatasetError(f"Item '{item_id}' not found in dataset '{self.name}'")
        
        item_file = self.dataset_path / "items" / f"{item_id}.json"
        
        if not item_file.exists():
            raise DatasetError(f"Item file not found: {item_file}")
        
        try:
            with open(item_file, 'r', encoding='utf-8') as f:
                item_data = json.load(f)
            
            # Validate item structure
            if "documents" not in item_data:
                raise DatasetError(f"Item '{item_id}' missing required 'documents' field")
            
            return item_data
            
        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON in item file {item_file}: {e}")
        except Exception as e:
            raise DatasetError(f"Error loading item '{item_id}': {e}")
    
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
        return f"Dataset(name='{self.name}', items={len(self._item_ids)})"


class ChunkedDataset:
    """
    Represents a chunked dataset, providing clean access to text chunks.
    
    Handles the new chunked format where each item is saved as a separate JSON file
    with the same structure as original items but with "chunks" instead of "content".
    """
    
    def __init__(self, chunked_directory_path: str):
        """
        Initialize ChunkedDataset from a chunked directory path.
        
        Args:
            chunked_directory_path: Path to chunked directory (e.g., "outputs/chunks/bmds_fixed_count_3/")
        
        Raises:
            DatasetError: If chunked directory cannot be accessed or is invalid
        """
        self.chunked_directory = Path(chunked_directory_path)
        self._item_ids = None
        
        # Validate directory and discover items
        self._discover_items()
    
    def _discover_items(self) -> None:
        """Discover available chunked items in the directory."""
        if not self.chunked_directory.exists():
            raise DatasetError(f"Chunked directory not found: {self.chunked_directory}")
        
        if not self.chunked_directory.is_dir():
            raise DatasetError(f"Path is not a directory: {self.chunked_directory}")
        
        # Look for items in the items/ subdirectory
        items_dir = self.chunked_directory / "items"
        if not items_dir.exists() or not items_dir.is_dir():
            raise DatasetError(f"Items subdirectory not found: {items_dir}")
        
        json_files = list(items_dir.glob("*.json"))
        if not json_files:
            raise DatasetError(f"No chunked items found in: {items_dir}")
        
        # Extract item IDs from filenames
        self._item_ids = [json_file.stem for json_file in json_files]
        self._item_ids.sort()  # Keep consistent ordering
        
        logger.debug(f"Discovered {len(self._item_ids)} chunked items")
    
    @property
    def item_ids(self) -> List[str]:
        """Returns list of item IDs available in chunked data."""
        return self._item_ids.copy()
    
    @property
    def name(self) -> str:
        """Returns the chunked dataset name (derived from directory name)."""
        return self.chunked_directory.name
    
    def load_collection_metadata(self) -> Dict[str, Any]:
        """
        Load collection metadata if available.
        
        Returns:
            Collection metadata dictionary, empty dict if not found
        """
        collection_file = self.chunked_directory / "collection.json"
        
        if not collection_file.exists():
            logger.debug(f"No collection.json found in {self.chunked_directory}")
            return {}
        
        try:
            with open(collection_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in collection file {collection_file}: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Error loading collection metadata: {e}")
            return {}
    
    def load_chunked_item(self, item_id: str) -> Dict[str, Any]:
        """
        Load individual chunked item JSON file.
        
        Args:
            item_id: ID of the chunked item to load
            
        Returns:
            Chunked item structure with item_metadata and documents (containing chunks)
            
        Raises:
            DatasetError: If chunked item cannot be loaded or doesn't exist
        """
        if item_id not in self._item_ids:
            raise DatasetError(f"Chunked item '{item_id}' not found in '{self.name}'")
        
        # Load from items/ subdirectory
        items_dir = self.chunked_directory / "items"
        item_file = items_dir / f"{item_id}.json"
        
        try:
            with open(item_file, 'r', encoding='utf-8') as f:
                item_data = json.load(f)
            
            # Validate chunked item structure
            if "documents" not in item_data:
                raise DatasetError(f"Chunked item '{item_id}' missing required 'documents' field")
            
            return item_data
            
        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON in chunked item file {item_file}: {e}")
        except Exception as e:
            raise DatasetError(f"Error loading chunked item '{item_id}': {e}")
    
    def get_item_chunks(self, item_id: str) -> List[str]:
        """
        Get all chunks for a specific item (flattened from all documents).
        
        Args:
            item_id: ID of the item
            
        Returns:
            List of text chunks (strings)
        """
        chunked_item = self.load_chunked_item(item_id)
        
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
        chunked_item = self.load_chunked_item(item_id)
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
        chunked_item = self.load_chunked_item(item_id)
        
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
        chunked_item = self.load_chunked_item(item_id)
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
        return f"ChunkedDataset(name='{self.name}', items={len(self._item_ids)})"