#!/usr/bin/env python3
"""
Utility script to switch between different GGUF models.
"""

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any

def load_model_config() -> Dict[str, Any]:
    """Load model configuration."""
    config_path = Path("config/providers/models.yaml")
    if not config_path.exists():
        return {}
    
    with open(config_path) as f:
        return yaml.safe_load(f)

def save_model_config(config: Dict[str, Any]) -> None:
    """Save model configuration."""
    config_path = Path("config/providers/models.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def update_env_file(model_name: str) -> None:
    """Update the .env file with the selected model."""
    env_path = Path(".env")
    
    # Read current .env content
    if env_path.exists():
        with open(env_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []
    
    # Update or add LLM_MODEL line
    model_line = f"LLM_MODEL={model_name}\n"
    found = False
    
    for i, line in enumerate(lines):
        if line.startswith("LLM_MODEL="):
            lines[i] = model_line
            found = True
            break
    
    if not found:
        lines.append(model_line)
    
    # Write back to .env
    with open(env_path, "w") as f:
        f.writelines(lines)

def list_available_models() -> Dict[str, str]:
    """List available models."""
    config = load_model_config()
    models = config.get("models", {})
    return {
        name: model.get("description", "No description available")
        for name, model in models.items()
    }

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration."""
    config = load_model_config()
    models = config.get("models", {})
    return models.get(model_name)

def validate_model_path(model_path: str) -> bool:
    """Validate model path."""
    if not model_path:
        return False
    return Path(model_path).exists()

def add_model(
    name: str,
    path: str,
    description: str = "",
    parameters: Dict[str, Any] = None
) -> None:
    """Add a new model."""
    config = load_model_config()
    models = config.get("models", {})
    
    models[name] = {
        "path": str(Path(path).resolve()),
        "description": description,
        "parameters": parameters or {}
    }
    
    config["models"] = models
    save_model_config(config)
    print(f"Added model: {name}")

def remove_model(name: str) -> None:
    """Remove a model."""
    config = load_model_config()
    models = config.get("models", {})
    
    if name in models:
        del models[name]
        config["models"] = models
        save_model_config(config)
        print(f"Removed model: {name}")
    else:
        print(f"Model not found: {name}")

def main():
    parser = argparse.ArgumentParser(description="Switch between different GGUF models")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    
    # Switch command
    switch_parser = subparsers.add_parser("switch", help="Switch to a model")
    switch_parser.add_argument("model", help="Name of the model to switch to")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new model")
    add_parser.add_argument("name", help="Name of the model")
    add_parser.add_argument("path", help="Path to the model file")
    add_parser.add_argument("--description", help="Model description")
    add_parser.add_argument("--parameters", help="Model parameters (JSON string)")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a model")
    remove_parser.add_argument("name", help="Name of the model to remove")
    
    args = parser.parse_args()
    
    if args.command == "list":
        print("\nAvailable models:")
        for name, desc in list_available_models().items():
            print(f"- {name}: {desc}")
    
    elif args.command == "switch":
        model_config = get_model_config(args.model)
        if not model_config:
            print(f"Error: Model '{args.model}' not found.")
            return
        
        if not validate_model_path(model_config["path"]):
            print(f"Error: Model file not found at {model_config['path']}")
            return
        
        update_env_file(args.model)
        print(f"Successfully switched to model: {args.model}")
    
    elif args.command == "add":
        import json
        parameters = {}
        if args.parameters:
            try:
                parameters = json.loads(args.parameters)
            except json.JSONDecodeError:
                print("Error: Invalid JSON in parameters")
                return
        
        add_model(args.name, args.path, args.description or "", parameters)
    
    elif args.command == "remove":
        remove_model(args.name)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 