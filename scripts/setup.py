#!/usr/bin/env python3
"""
Setup script for LangGraph.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the development environment."""
    # Create virtual environment
    subprocess.run([sys.executable, "-m", "venv", ".venv"])
    
    # Activate virtual environment
    if sys.platform == "win32":
        activate_script = ".venv\\Scripts\\activate"
    else:
        activate_script = ".venv/bin/activate"
    
    # Install dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements/base.txt"])
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements/dev.txt"])
    
    # Create necessary directories
    directories = [
        "logs",
        "storage/models",
        "storage/vector_store",
        "storage/cache",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Copy example environment file
    if not Path(".env").exists():
        Path(".env.example").copy(".env")
    
    print("Development environment setup complete!")

if __name__ == "__main__":
    setup_environment()
