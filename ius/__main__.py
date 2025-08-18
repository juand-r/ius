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
    elif command == "claim-extract":
        from ius.cli.claim_extract import main as claim_extract_main

        claim_extract_main()
    elif command == "whodunit":
        from ius.cli.whodunit import main as whodunit_main

        whodunit_main()
    elif command == "entity-coverage":
        from ius.cli.entity_coverage import main as entity_coverage_main

        entity_coverage_main()
    elif command == "entity-coverage-multi":
        from ius.cli.entity_coverage_multi import main as entity_coverage_multi_main

        entity_coverage_multi_main()
    elif command == "supert":
        from ius.cli.supert import main as supert_main

        supert_main()
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
    print("  chunk           Chunk documents for summarization")
    print("  summarize       Generate summaries from chunked data")
    print("  claim-extract   Extract claims from summaries")
    print("  whodunit        Evaluate detective stories (whodunit analysis)")
    print("  entity-coverage       Evaluate entity coverage in summaries")
    print("  entity-coverage-multi Evaluate entity coverage across multiple ranges")
    print("  supert          Evaluate summaries using SUPERT (reference-free metric)")
    print("  help            Show this help message")
    print()
    print("Examples:")
    print("  python -m ius chunk --dataset bmds --strategy fixed_size --size 2048")
    print("  python -m ius summarize --input outputs/chunks/ipython_test")
    print("  python -m ius claim-extract --input outputs/summaries/bmds_summaries")
    print("  python -m ius whodunit --input outputs/summaries/bmds_summaries --range 1-3")
    print("  python -m ius entity-coverage --input outputs/summaries/bmds_summaries --range penultimate")
    print("  python -m ius entity-coverage-multi --input outputs/summaries/bmds_summaries --max-range 5")
    print("  python -m ius supert --input outputs/summaries/bmds_summaries --chunks outputs/chunks/bmds_chunks")
    print("  python -m ius help")
    print()
    print("For command-specific help:")
    print("  python -m ius chunk --help")
    print("  python -m ius summarize --help")
    print("  python -m ius claim-extract --help")
    print("  python -m ius whodunit --help")
    print("  python -m ius entity-coverage --help")
    print("  python -m ius entity-coverage-multi --help")
    print("  python -m ius supert --help")


if __name__ == "__main__":
    main()
