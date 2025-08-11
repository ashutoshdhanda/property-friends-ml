#!/usr/bin/env python3
"""Script to train the property valuation model."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.train import main

if __name__ == "__main__":
    main()
