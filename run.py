#!/usr/bin/env python
"""Quickstart Runner for MedClear AI.
Executes the CrewAI workflow directly from the project root directory.

Usage:
    python run.py                                           # Runs with sample prescription & patient profile
    python run.py --prescription my_rx.txt --patient my_pt.txt  # Runs with your custom files
"""

import sys
from pathlib import Path

# Add src to Python module path
project_root = Path(__file__).resolve().parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from medclear_crew.main import run

if __name__ == "__main__":
    run()
