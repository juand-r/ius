"""
Custom exceptions for the IUS package.

These exceptions provide more specific error handling and better debugging
information than generic Python exceptions.
"""


class IUSError(Exception):
    """Base exception for all IUS package errors."""
    pass


class DatasetError(IUSError):
    """Dataset-related errors (loading, validation, format issues)."""
    pass


class ChunkingError(IUSError):
    """Chunking-related errors (invalid parameters, processing failures)."""
    pass


class ValidationError(IUSError):
    """Data validation errors (content preservation, format validation)."""
    pass


class ConfigurationError(IUSError):
    """Configuration-related errors (invalid settings, missing required config)."""
    pass