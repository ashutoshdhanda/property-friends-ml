#!/usr/bin/env python3
"""Script to run the API server."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.api.main import start_server

if __name__ == "__main__":
    start_server()
