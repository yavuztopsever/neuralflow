"""
Test suite for the LangGraph project.
This package contains all test modules organized by component.
"""

from pathlib import Path

# Define test directories
TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"
DATA_DIR = TEST_DIR / "data"

# Create directories if they don't exist
FIXTURES_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

from .unit import *
from .integration import *

__all__ = []
