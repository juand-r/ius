"""
Main entry point for IUS package.

Enables: python -m ius [command] [args]
"""

import sys

from ius.logging_config import get_logger, setup_logging


# Set up logger for main entry point
logger = get_logger(__name__)


def main():
    """Main entry point that delegates to appropriate CLI modules."""
    # Set up basic logging for error messages
    setup_logging(log_level="INFO")

    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1]

    # Remove the command from argv so submodules see the right arguments
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "chunk":
        from ius.cli.chunk import main as chunk_main

        chunk_main()
    elif command == "summarize":
        from ius.cli.summarize import main as summarize_main

        summarize_main()
    elif command == "help" or command == "-h" or command == "--help":
        print_help()
    else:
        logger.error(f"Unknown command: {command}")
        print_help()
        sys.exit(1)


def print_help():
    """Print main help message."""
    print("IUS - Incremental Update Summarization")
    print()
    print("Usage:")
    print("  python -m ius <command> [options]")
    print()
    print("Available commands:")
    print("  chunk       Chunk documents for summarization")
    print("  summarize   Generate summaries from chunked data")
    print("  help        Show this help message")
    print()
    print("Examples:")
    print("  python -m ius chunk --dataset bmds --strategy fixed_size --size 2048")
    print("  python -m ius summarize --input outputs/chunks/ipython_test --output my_experiment")
    print("  python -m ius help")
    print()
    print("For command-specific help:")
    print("  python -m ius chunk --help")
    print("  python -m ius summarize --help")


if __name__ == "__main__":
    main()
