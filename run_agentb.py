#!/usr/bin/env python
"""
Convenience wrapper for running Agent B from the command line.

This script allows you to run Agent B without needing to install it or use 'python -m'.

Usage:
    python run_agentb.py "Your task here"
    python run_agentb.py "Create a database in Notion" --headless
"""

import sys
from pathlib import Path

# Add src directory to Python path so we can import agentb
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from agentb.cli import main

if __name__ == "__main__":
    sys.exit(main())
