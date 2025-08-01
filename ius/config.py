"""Configuration management for IUS application."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .exceptions import ValidationError


@dataclass
class Config:
    """Application configuration with sensible defaults."""
    
    datasets_dir: Path = Path("datasets")
    outputs_dir: Path = Path("outputs")
    default_chunk_size: int = 1000
    default_num_chunks: int = 4
    max_memory_usage: int = 1024 * 1024 * 500  # 500MB
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        return cls(
            datasets_dir=Path(os.getenv('IUS_DATASETS_DIR', 'datasets')),
            outputs_dir=Path(os.getenv('IUS_OUTPUTS_DIR', 'outputs')),
            default_chunk_size=int(os.getenv('IUS_DEFAULT_CHUNK_SIZE', '1000')),
            default_num_chunks=int(os.getenv('IUS_DEFAULT_NUM_CHUNKS', '4')),
            max_memory_usage=int(os.getenv('IUS_MAX_MEMORY', str(1024 * 1024 * 500))),
            log_level=os.getenv('IUS_LOG_LEVEL', 'INFO'),
        )
    
    def validate(self) -> None:
        """Validate configuration values."""
        # Validate directories exist or can be created
        if not self.datasets_dir.exists():
            raise ValidationError(f"Datasets directory does not exist: {self.datasets_dir}")
        
        # Validate numeric values
        if self.default_chunk_size <= 0:
            raise ValidationError("default_chunk_size must be positive")
        
        if self.default_num_chunks <= 0:
            raise ValidationError("default_num_chunks must be positive")
        
        if self.max_memory_usage <= 0:
            raise ValidationError("max_memory_usage must be positive")
        
        # Validate log level
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            raise ValidationError(f"Invalid log_level: {self.log_level}. Must be one of: {valid_log_levels}")
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create outputs subdirectories
        (self.outputs_dir / "chunks").mkdir(parents=True, exist_ok=True)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
        _config.validate()
        _config.ensure_directories()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    config.validate()
    config.ensure_directories()
    _config = config


def reset_config() -> None:
    """Reset configuration to defaults (useful for testing)."""
    global _config
    _config = None