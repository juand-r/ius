#!/usr/bin/env python3
"""
Simple test runner for the IUS project.

Usage:
    python run_tests.py [test_module]

Examples:
    python run_tests.py                    # Run all tests
    python run_tests.py test_chunking      # Run specific test module
"""

import os
import sys


# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test module
        test_module = sys.argv[1]
        if not test_module.startswith("test_"):
            test_module = f"test_{test_module}"

        try:
            exec(f"from tests.{test_module} import *")
            if hasattr(
                sys.modules[f"tests.{test_module}"], "test_chunking_integration"
            ):
                print(f"Running {test_module}...")
                sys.modules[f"tests.{test_module}"].test_chunking_integration()
            else:
                print(f"Running {test_module} with unittest...")
                import unittest

                loader = unittest.TestLoader()
                suite = loader.loadTestsFromName(f"tests.{test_module}")
                runner = unittest.TextTestRunner(verbosity=2)
                runner.run(suite)
        except ImportError as e:
            print(f"Could not import {test_module}: {e}")
            sys.exit(1)
    else:
        # Run all tests
        print("Running all tests...")
        from tests.test_chunking import test_chunking_integration

        test_chunking_integration()

        print("\n" + "=" * 50)
        print("Running All Unit Tests")
        print("=" * 50)

        import unittest

        loader = unittest.TestLoader()
        suite = loader.discover("tests", pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
