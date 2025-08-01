#!/usr/bin/env python3
"""
Tests for logging configuration.
"""

import logging
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ius.logging_config import get_logger, setup_logging


class TestLoggingConfig(unittest.TestCase):
    """Test cases for logging configuration."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear any existing handlers to start fresh
        logging.getLogger().handlers.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        # Clear handlers after each test
        logging.getLogger().handlers.clear()
        self.temp_dir.cleanup()

    def test_setup_logging_basic(self):
        """Test basic logging setup with default parameters."""
        setup_logging()

        # Check root logger level
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.INFO)

        # Check that handler was added
        self.assertEqual(len(root_logger.handlers), 1)
        self.assertIsInstance(root_logger.handlers[0], logging.StreamHandler)

    def test_setup_logging_different_levels(self):
        """Test logging setup with different log levels."""
        test_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

        for level_str in test_levels:
            with self.subTest(level=level_str):
                # Clear previous handlers
                logging.getLogger().handlers.clear()

                setup_logging(log_level=level_str)
                root_logger = logging.getLogger()

                expected_level = getattr(logging, level_str)
                self.assertEqual(root_logger.level, expected_level)

    def test_setup_logging_verbose_formatting(self):
        """Test verbose logging format includes more details."""
        setup_logging(verbose=True)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        formatter = handler.formatter

        # Verbose format should include timestamp and logger name
        self.assertIn("%(asctime)s", formatter._fmt)
        self.assertIn("%(name)s", formatter._fmt)
        self.assertIn("%(levelname)s", formatter._fmt)
        self.assertIn("%(message)s", formatter._fmt)

    def test_setup_logging_normal_formatting(self):
        """Test normal logging format is simpler."""
        setup_logging(verbose=False)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        formatter = handler.formatter

        # Normal format should be simpler
        self.assertEqual(formatter._fmt, "%(levelname)s: %(message)s")

    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        log_file = self.temp_path / "test.log"

        setup_logging(log_file=log_file)

        root_logger = logging.getLogger()
        # Should have console handler + file handler
        self.assertEqual(len(root_logger.handlers), 2)

        # Check that log file was created
        self.assertTrue(log_file.parent.exists())

        # Test that logging to file works
        logger = get_logger("test")
        logger.info("Test message")

        # Check file contents
        with open(log_file) as f:
            content = f.read()
        self.assertIn("Test message", content)

    def test_setup_logging_file_directory_creation(self):
        """Test that logging creates parent directories for log file."""
        log_file = self.temp_path / "nested" / "dirs" / "test.log"

        setup_logging(log_file=log_file)

        # Directory should be created
        self.assertTrue(log_file.parent.exists())

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a proper logger instance."""
        logger = get_logger("test.module")

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test.module")

    def test_get_logger_different_names(self):
        """Test that get_logger returns different loggers for different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        logger3 = get_logger("module1")  # Same as logger1

        self.assertNotEqual(logger1, logger2)
        self.assertEqual(logger1, logger3)  # Should be same instance

    @patch('sys.stdout', new_callable=StringIO)
    def test_logging_output_levels(self, mock_stdout):
        """Test that different log levels produce appropriate output."""
        setup_logging(log_level="DEBUG")
        logger = get_logger("test")

        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        output = mock_stdout.getvalue()

        # All should appear with DEBUG level
        self.assertIn("DEBUG: Debug message", output)
        self.assertIn("INFO: Info message", output)
        self.assertIn("WARNING: Warning message", output)
        self.assertIn("ERROR: Error message", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_logging_level_filtering(self, mock_stdout):
        """Test that log level filtering works correctly."""
        setup_logging(log_level="WARNING")
        logger = get_logger("test")

        # Test different log levels
        logger.debug("Debug message")  # Should not appear
        logger.info("Info message")    # Should not appear
        logger.warning("Warning message")  # Should appear
        logger.error("Error message")      # Should appear

        output = mock_stdout.getvalue()

        # Only WARNING and ERROR should appear
        self.assertNotIn("Debug message", output)
        self.assertNotIn("Info message", output)
        self.assertIn("WARNING: Warning message", output)
        self.assertIn("ERROR: Error message", output)

    def test_third_party_logger_suppression(self):
        """Test that third-party loggers are properly suppressed."""
        setup_logging()

        # Check that urllib3 logger level is set to WARNING
        urllib3_logger = logging.getLogger('urllib3')
        self.assertEqual(urllib3_logger.level, logging.WARNING)

    def test_setup_logging_clears_existing_handlers(self):
        """Test that setup_logging clears any existing handlers."""
        # Add a dummy handler first
        root_logger = logging.getLogger()
        dummy_handler = logging.StreamHandler()
        root_logger.addHandler(dummy_handler)

        initial_count = len(root_logger.handlers)
        self.assertGreater(initial_count, 0)

        # Setup logging should clear and replace
        setup_logging()

        # Should have exactly 1 handler (the new one)
        self.assertEqual(len(root_logger.handlers), 1)
        self.assertNotIn(dummy_handler, root_logger.handlers)


class TestLoggingIntegration(unittest.TestCase):
    """Integration tests for logging in real usage scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        logging.getLogger().handlers.clear()

    def tearDown(self):
        """Clean up test fixtures."""
        logging.getLogger().handlers.clear()

    @patch('sys.stdout', new_callable=StringIO)
    def test_realistic_usage_pattern(self, mock_stdout):
        """Test logging in a realistic usage pattern like CLI modules."""
        # Simulate what happens in CLI modules
        setup_logging(log_level="INFO")
        logger = get_logger("ius.cli.chunk")

        # Simulate typical CLI messages
        logger.info("Loading dataset: test")
        logger.info("Loaded 5 items from test")
        logger.error("Dataset error: file not found")
        logger.warning("Chunking interrupted by user")

        output = mock_stdout.getvalue()

        # Check all messages appear correctly
        self.assertIn("INFO: Loading dataset: test", output)
        self.assertIn("INFO: Loaded 5 items from test", output)
        self.assertIn("ERROR: Dataset error: file not found", output)
        self.assertIn("WARNING: Chunking interrupted by user", output)

    def test_multiple_module_loggers(self):
        """Test that multiple modules can have their own loggers."""
        setup_logging()

        chunk_logger = get_logger("ius.cli.chunk")
        data_logger = get_logger("ius.data.loader")
        main_logger = get_logger("ius.__main__")

        # All should be different logger instances
        loggers = [chunk_logger, data_logger, main_logger]
        for i, logger1 in enumerate(loggers):
            for j, logger2 in enumerate(loggers):
                if i != j:
                    self.assertNotEqual(logger1, logger2)

        # But all should use the same root configuration
        self.assertEqual(chunk_logger.level, 0)  # Inherits from root
        self.assertEqual(data_logger.level, 0)   # Inherits from root
        self.assertEqual(main_logger.level, 0)   # Inherits from root


if __name__ == "__main__":
    unittest.main(verbosity=2)
