#!/usr/bin/env python3
"""
Sync requirements.txt with pyproject.toml using pip-tools.

This script ensures that requirements.txt stays in sync with pyproject.toml
by regenerating it from the dependency specifications.
"""

import subprocess
import sys


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"📦 {description}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Error: {e.stderr}")
        return False


def check_pip_tools() -> bool:
    """Check if pip-tools is installed."""
    try:
        subprocess.run(["pip-compile", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pip-tools not found. Installing...")
        return run_command(
            [sys.executable, "-m", "pip", "install", "pip-tools"],
            "Installing pip-tools",
        )


def main():
    """Main sync function."""
    print("🔄 Syncing requirements.txt with pyproject.toml")

    # Check prerequisites
    if not check_pip_tools():
        print("❌ Failed to install pip-tools")
        return 1

    # Generate production requirements.txt
    if not run_command(
        ["pip-compile", "--resolver=backtracking", "--upgrade", "requirements.in"],
        "Generating requirements.txt",
    ):
        return 1

    # Generate development requirements
    if not run_command(
        ["pip-compile", "--resolver=backtracking", "--upgrade", "requirements-dev.in"],
        "Generating requirements-dev.txt",
    ):
        return 1

    print("\n✅ Requirements files synced successfully!")
    print("📋 Files updated:")
    print("   - requirements.txt (production dependencies)")
    print("   - requirements-dev.txt (development dependencies)")
    print("\n💡 Commit these files to keep dependencies locked.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
