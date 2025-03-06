"""
Script to reorganize the project structure.
This script moves files to their new locations in the improved directory structure.
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the new directory structure."""
    base_dirs = [
        "src/core",
        "src/core/tools",
        "src/core/state",
        "src/core/engine",
        "src/core/workflow",
        "src/core/models",
        "src/core/context",
        "src/core/graph",
        "src/core/api",
        "src/core/services",
        "src/core/events",
        "src/core/utils",
        "src/core/utils/validation",
        "src/core/utils/text_processing",
        "src/core/utils/error_handling",
        "src/core/utils/logging",
        "src/core/utils/document",
        "src/core/utils/storage",
        "src/infrastructure",
        "src/api",
        "src/models",
        "src/ui",
        "src/services"
    ]
    
    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def move_utils_to_core():
    """Move utility functions to core/utils."""
    # Move common utilities
    utils_src = Path("src/utils/common/utils")
    utils_dest = Path("src/core/utils")
    
    if utils_src.exists():
        # Move core utilities
        if (utils_src / "core").exists():
            for item in (utils_src / "core").iterdir():
                if item.is_file():
                    shutil.copy2(item, utils_dest / item.name)
                elif item.is_dir():
                    shutil.copytree(item, utils_dest / item.name, dirs_exist_ok=True)
        
        # Move other utilities
        for dir_name in ["common", "error", "logging", "document", "note", "storage"]:
            if (utils_src / dir_name).exists():
                shutil.copytree(utils_src / dir_name, utils_dest / dir_name, dirs_exist_ok=True)

def update_imports():
    """Update import statements in all Python files."""
    for root, _, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                with open(file_path, "r") as f:
                    content = f.read()
                
                # Update imports from utils to core.utils
                content = content.replace("from utils.", "from core.utils.")
                content = content.replace("import utils.", "import core.utils.")
                
                with open(file_path, "w") as f:
                    f.write(content)

def cleanup():
    """Remove old directories and files."""
    # Remove old utils directory
    if Path("src/utils").exists():
        shutil.rmtree("src/utils")

def main():
    """Main function to handle the reorganization."""
    print("Starting directory reorganization...")
    
    # Create new directory structure
    create_directory_structure()
    print("Created new directory structure")
    
    # Move utilities to core
    move_utils_to_core()
    print("Moved utilities to core")
    
    # Update imports
    update_imports()
    print("Updated imports")
    
    # Cleanup
    cleanup()
    print("Cleaned up old directories")
    
    print("Reorganization complete!")

if __name__ == "__main__":
    main() 