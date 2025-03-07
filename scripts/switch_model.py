#!/usr/bin/env python3
"""
Utility script to switch between different GGUF models.
"""

import argparse
import os
from pathlib import Path
from config.models import list_available_models, get_model_config, validate_model_path

def update_env_file(model_name: str):
    """Update the .env file with the selected model."""
    env_path = Path(__file__).parent.parent / '.env'
    
    # Read current .env content
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines = f.readlines()
    else:
        lines = []
    
    # Update or add LLM_MODEL line
    model_line = f"LLM_MODEL={model_name}\n"
    found = False
    
    for i, line in enumerate(lines):
        if line.startswith('LLM_MODEL='):
            lines[i] = model_line
            found = True
            break
    
    if not found:
        lines.append(model_line)
    
    # Write back to .env
    with open(env_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Updated .env file to use model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Switch between different GGUF models")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model to switch to"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable models:")
        for name, desc in list_available_models().items():
            print(f"- {name}: {desc}")
        return
    
    if args.model:
        model_config = get_model_config(args.model)
        if not model_config:
            print(f"Error: Model '{args.model}' not found.")
            print("\nAvailable models:")
            for name, desc in list_available_models().items():
                print(f"- {name}: {desc}")
            return
        
        if not validate_model_path(model_config.path):
            print(f"Error: Model file not found at {model_config.path}")
            return
        
        update_env_file(args.model)
        print(f"Successfully switched to model: {args.model}")
        return
    
    parser.print_help()

if __name__ == "__main__":
    main() 