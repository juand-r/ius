"""Tests for configuration management."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ius.config import Config, get_config, reset_config, set_config
from ius.exceptions import ValidationError


class TestConfig(unittest.TestCase):
    """Test cases for Config dataclass."""

    def test_default_values(self):
        """Test that Config has sensible default values."""
        config = Config()

        self.assertEqual(config.datasets_dir, Path("datasets"))
        self.assertEqual(config.outputs_dir, Path("outputs"))
        self.assertEqual(config.default_chunk_size, 1000)
        self.assertEqual(config.default_num_chunks, 4)
        self.assertEqual(config.max_memory_usage, 1024 * 1024 * 500)  # 500MB
        self.assertEqual(config.log_level, "INFO")

    def test_from_env_with_defaults(self):
        """Test from_env() with no environment variables (uses defaults)."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env()

            self.assertEqual(config.datasets_dir, Path("datasets"))
            self.assertEqual(config.outputs_dir, Path("outputs"))
            self.assertEqual(config.default_chunk_size, 1000)
            self.assertEqual(config.default_num_chunks, 4)
            self.assertEqual(config.max_memory_usage, 1024 * 1024 * 500)
            self.assertEqual(config.log_level, "INFO")

    def test_from_env_with_custom_values(self):
        """Test from_env() with custom environment variables."""
        env_vars = {
            "IUS_DATASETS_DIR": "/custom/datasets",
            "IUS_OUTPUTS_DIR": "/custom/outputs",
            "IUS_DEFAULT_CHUNK_SIZE": "2000",
            "IUS_DEFAULT_NUM_CHUNKS": "8",
            "IUS_MAX_MEMORY": "1073741824",  # 1GB
            "IUS_LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.from_env()

            self.assertEqual(config.datasets_dir, Path("/custom/datasets"))
            self.assertEqual(config.outputs_dir, Path("/custom/outputs"))
            self.assertEqual(config.default_chunk_size, 2000)
            self.assertEqual(config.default_num_chunks, 8)
            self.assertEqual(config.max_memory_usage, 1073741824)
            self.assertEqual(config.log_level, "DEBUG")

    def test_from_env_partial_override(self):
        """Test from_env() with only some environment variables set."""
        env_vars = {"IUS_DEFAULT_CHUNK_SIZE": "1500", "IUS_LOG_LEVEL": "WARNING"}

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.from_env()

            # Overridden values
            self.assertEqual(config.default_chunk_size, 1500)
            self.assertEqual(config.log_level, "WARNING")

            # Default values for non-overridden
            self.assertEqual(config.datasets_dir, Path("datasets"))
            self.assertEqual(config.default_num_chunks, 4)


class TestConfigValidation(unittest.TestCase):
    """Test cases for Config validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.datasets_dir = Path(self.temp_dir) / "datasets"
        self.datasets_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_validate_success(self):
        """Test successful validation with valid configuration."""
        config = Config(
            datasets_dir=self.datasets_dir,
            default_chunk_size=1000,
            default_num_chunks=4,
            max_memory_usage=1024 * 1024 * 100,
            log_level="INFO",
        )

        # Should not raise an exception
        config.validate()

    def test_validate_missing_datasets_directory(self):
        """Test validation fails when datasets directory doesn't exist."""
        missing_dir = Path(self.temp_dir) / "nonexistent"

        config = Config(datasets_dir=missing_dir)

        with self.assertRaises(ValidationError) as cm:
            config.validate()

        self.assertIn("Datasets directory does not exist", str(cm.exception))
        self.assertIn(str(missing_dir), str(cm.exception))

    def test_validate_negative_chunk_size(self):
        """Test validation fails with negative chunk size."""
        config = Config(datasets_dir=self.datasets_dir, default_chunk_size=-100)

        with self.assertRaises(ValidationError) as cm:
            config.validate()

        self.assertEqual(str(cm.exception), "default_chunk_size must be positive")

    def test_validate_zero_chunk_size(self):
        """Test validation fails with zero chunk size."""
        config = Config(datasets_dir=self.datasets_dir, default_chunk_size=0)

        with self.assertRaises(ValidationError) as cm:
            config.validate()

        self.assertEqual(str(cm.exception), "default_chunk_size must be positive")

    def test_validate_negative_num_chunks(self):
        """Test validation fails with negative number of chunks."""
        config = Config(datasets_dir=self.datasets_dir, default_num_chunks=-5)

        with self.assertRaises(ValidationError) as cm:
            config.validate()

        self.assertEqual(str(cm.exception), "default_num_chunks must be positive")

    def test_validate_zero_num_chunks(self):
        """Test validation fails with zero number of chunks."""
        config = Config(datasets_dir=self.datasets_dir, default_num_chunks=0)

        with self.assertRaises(ValidationError) as cm:
            config.validate()

        self.assertEqual(str(cm.exception), "default_num_chunks must be positive")

    def test_validate_negative_memory_usage(self):
        """Test validation fails with negative memory usage."""
        config = Config(datasets_dir=self.datasets_dir, max_memory_usage=-1000)

        with self.assertRaises(ValidationError) as cm:
            config.validate()

        self.assertEqual(str(cm.exception), "max_memory_usage must be positive")

    def test_validate_zero_memory_usage(self):
        """Test validation fails with zero memory usage."""
        config = Config(datasets_dir=self.datasets_dir, max_memory_usage=0)

        with self.assertRaises(ValidationError) as cm:
            config.validate()

        self.assertEqual(str(cm.exception), "max_memory_usage must be positive")

    def test_validate_invalid_log_level(self):
        """Test validation fails with invalid log level."""
        config = Config(datasets_dir=self.datasets_dir, log_level="INVALID")

        with self.assertRaises(ValidationError) as cm:
            config.validate()

        error_msg = str(cm.exception)
        self.assertIn("Invalid log_level: INVALID", error_msg)
        self.assertIn("DEBUG", error_msg)
        self.assertIn("INFO", error_msg)
        self.assertIn("WARNING", error_msg)
        self.assertIn("ERROR", error_msg)
        self.assertIn("CRITICAL", error_msg)

    def test_validate_log_level_case_insensitive(self):
        """Test validation accepts log levels in different cases."""
        valid_levels = ["debug", "INFO", "Warning", "ERROR", "critical"]

        for level in valid_levels:
            config = Config(datasets_dir=self.datasets_dir, log_level=level)
            # Should not raise an exception
            config.validate()


class TestConfigDirectoryManagement(unittest.TestCase):
    """Test cases for Config directory management."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.datasets_dir = Path(self.temp_dir) / "datasets"
        self.datasets_dir.mkdir()
        self.outputs_dir = Path(self.temp_dir) / "outputs"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_ensure_directories_creates_outputs_dir(self):
        """Test ensure_directories creates outputs directory if it doesn't exist."""
        config = Config(datasets_dir=self.datasets_dir, outputs_dir=self.outputs_dir)

        # Directory shouldn't exist initially
        self.assertFalse(self.outputs_dir.exists())

        config.ensure_directories()

        # Directory should be created
        self.assertTrue(self.outputs_dir.exists())
        self.assertTrue(self.outputs_dir.is_dir())

    def test_ensure_directories_creates_chunks_subdir(self):
        """Test ensure_directories creates outputs/chunks subdirectory."""
        config = Config(datasets_dir=self.datasets_dir, outputs_dir=self.outputs_dir)

        config.ensure_directories()

        chunks_dir = self.outputs_dir / "chunks"
        self.assertTrue(chunks_dir.exists())
        self.assertTrue(chunks_dir.is_dir())

    def test_ensure_directories_with_existing_dirs(self):
        """Test ensure_directories works when directories already exist."""
        # Create directories first
        self.outputs_dir.mkdir(parents=True)
        (self.outputs_dir / "chunks").mkdir()

        config = Config(datasets_dir=self.datasets_dir, outputs_dir=self.outputs_dir)

        # Should not raise an exception
        config.ensure_directories()

        # Directories should still exist
        self.assertTrue(self.outputs_dir.exists())
        self.assertTrue((self.outputs_dir / "chunks").exists())

    def test_ensure_directories_creates_nested_paths(self):
        """Test ensure_directories creates nested directory paths."""
        nested_outputs = Path(self.temp_dir) / "deeply" / "nested" / "outputs"

        config = Config(datasets_dir=self.datasets_dir, outputs_dir=nested_outputs)

        config.ensure_directories()

        self.assertTrue(nested_outputs.exists())
        self.assertTrue((nested_outputs / "chunks").exists())


class TestGlobalConfigManagement(unittest.TestCase):
    """Test cases for global configuration management functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Always reset config before each test
        reset_config()

        self.temp_dir = tempfile.mkdtemp()
        self.datasets_dir = Path(self.temp_dir) / "datasets"
        self.datasets_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)
        # Reset config after each test
        reset_config()

    def test_get_config_lazy_initialization(self):
        """Test get_config() creates config on first call."""
        # Mock the datasets directory to exist (current working directory)
        with patch("ius.config.Config.from_env") as mock_from_env:
            mock_config = Config(datasets_dir=self.datasets_dir)
            mock_from_env.return_value = mock_config

            # First call should create config
            config1 = get_config()

            # Should call from_env()
            mock_from_env.assert_called_once()

            # Second call should return same instance
            config2 = get_config()

            self.assertIs(config1, config2)
            # from_env should still have been called only once
            self.assertEqual(mock_from_env.call_count, 1)

    def test_get_config_validates_and_ensures_directories(self):
        """Test get_config() validates config and ensures directories."""
        with (
            patch("ius.config.Config.from_env") as mock_from_env,
            patch.object(Config, "validate") as mock_validate,
            patch.object(Config, "ensure_directories") as mock_ensure,
        ):
            mock_config = Config(datasets_dir=self.datasets_dir)
            mock_from_env.return_value = mock_config

            get_config()

            mock_validate.assert_called_once()
            mock_ensure.assert_called_once()

    def test_set_config_overrides_global_config(self):
        """Test set_config() overrides the global configuration."""
        # First get default config
        with patch("ius.config.Config.from_env") as mock_from_env:
            default_config = Config(datasets_dir=self.datasets_dir)
            mock_from_env.return_value = default_config

            get_config()

            # Create and set custom config
            custom_config = Config(
                datasets_dir=self.datasets_dir,
                default_chunk_size=2000,
                log_level="DEBUG",
            )

            set_config(custom_config)

            # Get config again
            config2 = get_config()

            # Should be the custom config
            self.assertIs(config2, custom_config)
            self.assertEqual(config2.default_chunk_size, 2000)
            self.assertEqual(config2.log_level, "DEBUG")

    def test_set_config_validates_and_ensures_directories(self):
        """Test set_config() validates config and ensures directories."""
        with (
            patch.object(Config, "validate") as mock_validate,
            patch.object(Config, "ensure_directories") as mock_ensure,
        ):
            custom_config = Config(datasets_dir=self.datasets_dir)

            set_config(custom_config)

            mock_validate.assert_called_once()
            mock_ensure.assert_called_once()

    def test_reset_config_clears_global_config(self):
        """Test reset_config() clears the global configuration."""
        # First create a config
        with patch("ius.config.Config.from_env") as mock_from_env:
            # Use side_effect to return new instances each time
            mock_from_env.side_effect = [
                Config(datasets_dir=self.datasets_dir),
                Config(datasets_dir=self.datasets_dir),
            ]

            config1 = get_config()

            # Reset config
            reset_config()

            # Get config again - should create new instance
            config2 = get_config()

            # Should be different instances (new one created)
            self.assertIsNot(config1, config2)

            # from_env should have been called twice (once for each get_config)
            self.assertEqual(mock_from_env.call_count, 2)

    def test_set_config_validation_failure(self):
        """Test set_config() raises exception when validation fails."""
        # Create invalid config (missing datasets directory)
        missing_dir = Path(self.temp_dir) / "nonexistent"
        invalid_config = Config(datasets_dir=missing_dir)

        with self.assertRaises(ValidationError):
            set_config(invalid_config)


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for Config with real environment variables."""

    def setUp(self):
        """Set up test fixtures."""
        reset_config()
        self.temp_dir = tempfile.mkdtemp()
        self.datasets_dir = Path(self.temp_dir) / "datasets"
        self.datasets_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)
        reset_config()

    def test_environment_variable_integration(self):
        """Test complete integration with environment variables."""
        env_vars = {
            "IUS_DATASETS_DIR": str(self.datasets_dir),
            "IUS_OUTPUTS_DIR": str(Path(self.temp_dir) / "custom_outputs"),
            "IUS_DEFAULT_CHUNK_SIZE": "1500",
            "IUS_DEFAULT_NUM_CHUNKS": "6",
            "IUS_LOG_LEVEL": "WARNING",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = get_config()

            self.assertEqual(config.datasets_dir, Path(self.datasets_dir))
            self.assertEqual(config.outputs_dir, Path(self.temp_dir) / "custom_outputs")
            self.assertEqual(config.default_chunk_size, 1500)
            self.assertEqual(config.default_num_chunks, 6)
            self.assertEqual(config.log_level, "WARNING")

            # Outputs directory should be created
            self.assertTrue(config.outputs_dir.exists())
            self.assertTrue((config.outputs_dir / "chunks").exists())

    def test_invalid_environment_variable_values(self):
        """Test handling of invalid environment variable values."""
        env_vars = {
            "IUS_DATASETS_DIR": str(self.datasets_dir),
            "IUS_DEFAULT_CHUNK_SIZE": "not_a_number",
        }

        with (
            patch.dict(os.environ, env_vars, clear=True),
            self.assertRaises(ValueError),
        ):
            # Should fail when trying to convert 'not_a_number' to int
            get_config()


if __name__ == "__main__":
    unittest.main()
