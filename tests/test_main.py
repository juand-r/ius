#!/usr/bin/env python3
"""
Tests for main entry point module.
"""

import sys
import unittest
from unittest.mock import patch

from ius.__main__ import main, print_help


class TestMainEntryPoint(unittest.TestCase):
    """Test cases for main entry point."""

    def setUp(self):
        """Set up test fixtures."""
        self.original_argv = sys.argv.copy()

    def tearDown(self):
        """Clean up test fixtures."""
        sys.argv = self.original_argv

    @patch('ius.__main__.print_help')
    def test_main_no_arguments(self, mock_print_help):
        """Test main with no command arguments."""
        sys.argv = ['ius']

        main()

        mock_print_help.assert_called_once()

    @patch('ius.cli.chunk.main')
    def test_main_chunk_command(self, mock_chunk_main):
        """Test main with chunk command."""
        sys.argv = ['ius', 'chunk', '--dataset', 'test']

        main()

        mock_chunk_main.assert_called_once()
        # Should remove 'chunk' from argv, leaving ['ius', '--dataset', 'test']
        self.assertEqual(sys.argv, ['ius', '--dataset', 'test'])

    @patch('ius.__main__.print_help')
    def test_main_help_command(self, mock_print_help):
        """Test main with help command."""
        test_cases = ['help', '-h', '--help']

        for help_cmd in test_cases:
            with self.subTest(command=help_cmd):
                mock_print_help.reset_mock()
                sys.argv = ['ius', help_cmd]

                main()

                mock_print_help.assert_called_once()

    @patch('ius.__main__.print_help')
    @patch('sys.exit')
    @patch('ius.__main__.logger')
    def test_main_unknown_command(self, mock_logger, mock_exit, mock_print_help):
        """Test main with unknown command."""
        sys.argv = ['ius', 'unknown_command']

        main()

        # Should log error message
        mock_logger.error.assert_called_with("Unknown command: unknown_command")
        mock_print_help.assert_called_once()
        mock_exit.assert_called_with(1)

    @patch('builtins.print')
    def test_print_help_content(self, mock_print):
        """Test print_help displays correct content."""
        print_help()

        calls = mock_print.call_args_list
        output = '\n'.join([str(call.args[0]) if call.args else str(call) for call in calls])

        # Check key components of help text
        self.assertIn("IUS - Incremental Update Summarization", output)
        self.assertIn("python -m ius <command>", output)
        self.assertIn("chunk", output)
        self.assertIn("help", output)
        self.assertIn("Examples:", output)

    @patch('ius.cli.chunk.main')
    def test_argv_manipulation_multiple_args(self, mock_chunk_main):
        """Test that argv is correctly manipulated with multiple arguments."""
        original_argv = ['ius', 'chunk', '--dataset', 'bmds', '--strategy', 'fixed_size']
        sys.argv = original_argv.copy()

        main()

        mock_chunk_main.assert_called_once()
        # Should have ['ius', '--dataset', 'bmds', '--strategy', 'fixed_size']
        expected_argv = [original_argv[0]] + original_argv[2:]
        self.assertEqual(sys.argv, expected_argv)

    @patch('ius.cli.chunk.main')
    def test_argv_manipulation_single_command(self, mock_chunk_main):
        """Test argv manipulation with just command and no additional args."""
        sys.argv = ['ius', 'chunk']

        main()

        mock_chunk_main.assert_called_once()
        # Should have just ['ius']
        self.assertEqual(sys.argv, ['ius'])

    @patch('ius.__main__.print_help')
    @patch('ius.__main__.logger')
    def test_main_empty_string_command(self, mock_logger, mock_print_help):
        """Test main behavior with empty string as second argument."""
        sys.argv = ['ius', '']

        with patch('sys.exit') as mock_exit:
            main()

        mock_logger.error.assert_called_with("Unknown command: ")
        mock_print_help.assert_called_once()
        mock_exit.assert_called_with(1)

    @patch('ius.cli.chunk.main')
    @patch('ius.__main__.logger')
    def test_chunk_command_case_sensitivity(self, mock_logger, mock_chunk_main):
        """Test that chunk command is case sensitive."""
        sys.argv = ['ius', 'CHUNK']  # Wrong case

        with patch('sys.exit'), patch('ius.__main__.print_help'):
            main()

        # Should treat as unknown command, not call chunk
        mock_chunk_main.assert_not_called()
        mock_logger.error.assert_called_with("Unknown command: CHUNK")

    def test_main_preserves_program_name(self):
        """Test that main preserves the program name in argv[0]."""
        original_program = 'custom_program_name'
        sys.argv = [original_program, 'chunk', '--test']

        with patch('ius.cli.chunk.main') as mock_chunk:
            main()

        # Program name should be preserved
        self.assertEqual(sys.argv[0], original_program)
        mock_chunk.assert_called_once()


class TestMainIntegration(unittest.TestCase):
    """Integration tests for main entry point."""

    def setUp(self):
        """Set up test fixtures."""
        self.original_argv = sys.argv.copy()

    def tearDown(self):
        """Clean up test fixtures."""
        sys.argv = self.original_argv

    @patch('builtins.print')
    def test_help_formatting(self, mock_print):
        """Test that help output is properly formatted."""
        print_help()

        calls = mock_print.call_args_list

        # Should have multiple print calls for formatting
        self.assertGreater(len(calls), 5)

        # Check for proper sections
        all_output = [str(call.args[0]) if call.args else str(call) for call in calls]

        # Should have header
        self.assertTrue(any("IUS" in output for output in all_output))

        # Should have usage section
        self.assertTrue(any("Usage:" in output for output in all_output))

        # Should have commands section
        self.assertTrue(any("Available commands:" in output for output in all_output))

        # Should have examples section
        self.assertTrue(any("Examples:" in output for output in all_output))

    def test_command_routing_isolation(self):
        """Test that command routing doesn't interfere with imports."""
        # This test ensures that importing the chunk module doesn't have side effects

        with patch('ius.cli.chunk.main') as mock_chunk:
            # Should be able to call main multiple times without issues
            sys.argv = ['ius', 'chunk']
            main()
            sys.argv = ['ius', 'chunk']
            main()

        # Should have been called twice
        self.assertEqual(mock_chunk.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
