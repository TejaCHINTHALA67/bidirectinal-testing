#!/usr/bin/env python3
"""
Wrapper script to run dataset preparation from the scripts directory.
Usage: python scripts/prepare_datasets.py --datasets nq triviaqa hotpotqa stackoverflow
"""
import sys
import os

# Add project root to Python path (one level up from scripts)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the main function from dataset_loader
try:
    from src.data.dataset_loader import main
except ImportError:
    # If src is not found, try adding the current directory (if run from root)
    sys.path.insert(0, os.getcwd())
    try:
        from src.data.dataset_loader import main
    except ImportError as e:
        print(f"Error: Could not import dataset_loader. Make sure you are in the cloud_run directory.")
        print(f"Python path: {sys.path}")
        print(f"Error details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
