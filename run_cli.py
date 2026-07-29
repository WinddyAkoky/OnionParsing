"""
Script runner for local development and testing.

Usage:
    python run_cli.py -i input.png -o output_dir -d
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

if __name__ == "__main__":
    from onion_parsing.cli.main import main as cli_main
    cli_main()
