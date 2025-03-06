"""
Configuration management utilities specific to the LangGraph project.
These utilities handle configuration that is not covered by LangChain's built-in configuration.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ConfigMetadata:
    """Metadata about a configuration."""
    created_at: str
    updated_at: str
    version: str
    environment: str
    description: Optional[str] = None

class ConfigManager:
    """Manages configuration for the LangGraph project."""
    
    def __init__(
        self,
        config_dir: str = "config",
        default_env: str = "development",
        config_version: str = "1.0.0"
    ):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.default_env = default_env
        self.config_version = config_version
        
        # Initialize configuration
        self.config: Dict[str, Any] = {}
        self.metadata = ConfigMetadata(
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            version=config_version,
            environment=default_env
        )
        
        # Load default configuration if it exists
        self._load_default_config()

    def _load_default_config(self):
        """Load default configuration from file if it exists."""
        default_config_path = self.config_dir / "config.yaml"
        if default_config_path.exists():
            self.load_config(default_config_path)

    def set_config(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value
        self.metadata.updated_at = datetime.now().isoformat()

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)

    def delete_config(self, key: str):
        """Delete a configuration value."""
        if key in self.config:
            del self.config[key]
            self.metadata.updated_at = datetime.now().isoformat()

    def load_config(self, filepath: Union[str, Path]):
        """Load configuration from a file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            if filepath.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif filepath.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")
        
        # Update configuration
        self.config.update(data.get('config', {}))
        
        # Update metadata if present
        if 'metadata' in data:
            metadata_data = data['metadata']
            self.metadata = ConfigMetadata(
                created_at=metadata_data.get('created_at', self.metadata.created_at),
                updated_at=datetime.now().isoformat(),
                version=metadata_data.get('version', self.metadata.version),
                environment=metadata_data.get('environment', self.metadata.environment),
                description=metadata_data.get('description')
            )

    def save_config(self, filepath: Union[str, Path]):
        """Save configuration to a file."""
        filepath = Path(filepath)
        
        # Prepare data
        data = {
            'config': self.config,
            'metadata': asdict(self.metadata)
        }
        
        # Save based on file extension
        with open(filepath, 'w') as f:
            if filepath.suffix in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False)
            elif filepath.suffix == '.json':
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self.config.copy()

    def clear_config(self):
        """Clear all configuration values."""
        self.config.clear()
        self.metadata.updated_at = datetime.now().isoformat()

    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration with new values."""
        self.config.update(new_config)
        self.metadata.updated_at = datetime.now().isoformat()

    def get_environment_config(self, env: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific environment."""
        env = env or self.default_env
        return self.config.get('environments', {}).get(env, {})

    def set_environment_config(self, env: str, config: Dict[str, Any]):
        """Set configuration for a specific environment."""
        if 'environments' not in self.config:
            self.config['environments'] = {}
        self.config['environments'][env] = config
        self.metadata.updated_at = datetime.now().isoformat()

    def get_secret(self, key: str, default: Any = None) -> Any:
        """Get a secret value from environment variables."""
        return os.getenv(key, default)

    def set_secret(self, key: str, value: str):
        """Set a secret value in environment variables."""
        os.environ[key] = value

    def validate_config(self) -> List[str]:
        """Validate the current configuration."""
        errors = []
        
        # Add validation rules here
        # Example:
        # if 'api_key' not in self.config:
        #     errors.append("Missing required configuration: api_key")
        
        return errors

    def get_config_diff(self, other_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get the difference between current and another configuration."""
        diff = {
            'added': {},
            'removed': {},
            'modified': {}
        }
        
        # Find added and modified keys
        for key, value in other_config.items():
            if key not in self.config:
                diff['added'][key] = value
            elif self.config[key] != value:
                diff['modified'][key] = {
                    'old': self.config[key],
                    'new': value
                }
        
        # Find removed keys
        for key in self.config:
            if key not in other_config:
                diff['removed'][key] = self.config[key]
        
        return diff

    def merge_config(self, other_config: Dict[str, Any], overwrite: bool = False):
        """Merge another configuration with the current one."""
        if overwrite:
            self.config = other_config
        else:
            self.update_config(other_config)
        self.metadata.updated_at = datetime.now().isoformat() 