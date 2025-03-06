"""
Environment configuration for LangGraph.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel

class EnvironmentConfig(BaseModel):
    """Configuration for the current environment."""
    name: str
    config_dir: Path
    data_dir: Path
    log_dir: Path
    cache_dir: Path
    model_dir: Path
    vector_store_dir: Path

class ConfigManager:
    """Manages configuration for different environments."""
    
    def __init__(self, environment: str = "development"):
        """Initialize the configuration manager.
        
        Args:
            environment: Environment name (development, production, etc.)
        """
        self.environment = environment
        self.env_config = self._load_environment_config()
        self.config = self._load_config()
    
    def _load_environment_config(self) -> EnvironmentConfig:
        """Load environment-specific configuration."""
        base_dir = Path(__file__).parent.parent.parent
        
        return EnvironmentConfig(
            name=self.environment,
            config_dir=base_dir / "config",
            data_dir=base_dir / "storage",
            log_dir=base_dir / "logs",
            cache_dir=base_dir / "storage" / "cache",
            model_dir=base_dir / "storage" / "models",
            vector_store_dir=base_dir / "storage" / "vector_store"
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files."""
        config = {}
        
        # Load default config
        default_config = self._load_yaml(self.env_config.config_dir / "default.yaml")
        if default_config:
            config.update(default_config)
        
        # Load environment-specific config
        env_config = self._load_yaml(
            self.env_config.config_dir / f"{self.environment}.yaml"
        )
        if env_config:
            config.update(env_config)
        
        return config
    
    def _load_yaml(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load YAML configuration file."""
        try:
            if path.exists():
                with open(path, "r") as f:
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file {path}: {e}")
        return None
    
    def get_model_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific model provider."""
        return self.config.get("models", {}).get(provider, {})
    
    def get_storage_config(self, storage_type: str, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific storage provider."""
        return self.config.get("storage", {}).get(storage_type, {}).get(provider, {})
    
    def get_workflow_config(self) -> Dict[str, Any]:
        """Get workflow configuration."""
        return self.config.get("workflow", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get("logging", {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.config.get("monitoring", {})
    
    def get_core_config(self) -> Dict[str, Any]:
        """Get core configuration."""
        return self.config.get("core", {})

# Global config manager instance
config_manager = ConfigManager()

__all__ = ['EnvironmentConfig', 'ConfigManager', 'config_manager']
