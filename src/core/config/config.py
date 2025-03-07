"""
Configuration utilities for the LangGraph project.
This module provides configuration management capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging
import json
import os
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class Config(BaseModel):
    """Configuration model."""
    name: str
    value: Any
    description: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class ConfigManager:
    """Manager for handling configurations in the LangGraph system."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Optional directory for storing configurations
        """
        self.configs: Dict[str, Config] = {}
        self.config_dir = config_dir or os.path.join(os.getcwd(), "config")
        self._ensure_config_dir()
    
    def _ensure_config_dir(self) -> None:
        """Ensure configuration directory exists."""
        os.makedirs(self.config_dir, exist_ok=True)
    
    def set_config(
        self,
        name: str,
        value: Any,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Config:
        """
        Set a configuration value.
        
        Args:
            name: Configuration name
            value: Configuration value
            description: Optional description
            metadata: Optional metadata
            
        Returns:
            Config: Created configuration
        """
        try:
            # Create config
            config = Config(
                name=name,
                value=value,
                description=description,
                metadata=metadata
            )
            
            # Store config
            self.configs[name] = config
            
            # Log config
            logger.info(
                f"Configuration set: {name}",
                extra={
                    "config_name": name,
                    "config_value": value,
                    "config_description": description,
                    "config_metadata": metadata
                }
            )
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to set configuration: {e}")
            raise
    
    def get_config(self, name: str) -> Optional[Config]:
        """
        Get a configuration value.
        
        Args:
            name: Configuration name
            
        Returns:
            Optional[Config]: Configuration if found
        """
        return self.configs.get(name)
    
    def delete_config(self, name: str) -> None:
        """
        Delete a configuration value.
        
        Args:
            name: Configuration name
        """
        if name in self.configs:
            del self.configs[name]
            logger.info(f"Configuration deleted: {name}")
    
    def list_configs(self) -> Dict[str, Config]:
        """
        List all configurations.
        
        Returns:
            Dict[str, Config]: Dictionary of configurations
        """
        return self.configs
    
    def save_configs(self, filename: Optional[str] = None) -> None:
        """
        Save configurations to file.
        
        Args:
            filename: Optional filename to save to
        """
        try:
            if not filename:
                filename = os.path.join(self.config_dir, "configs.json")
            
            # Convert configs to dict
            config_dict = {
                name: {
                    "value": config.value,
                    "description": config.description,
                    "timestamp": config.timestamp.isoformat(),
                    "metadata": config.metadata
                }
                for name, config in self.configs.items()
            }
            
            # Save to file
            with open(filename, "w") as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configurations saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save configurations: {e}")
            raise
    
    def load_configs(self, filename: Optional[str] = None) -> None:
        """
        Load configurations from file.
        
        Args:
            filename: Optional filename to load from
        """
        try:
            if not filename:
                filename = os.path.join(self.config_dir, "configs.json")
            
            if not os.path.exists(filename):
                logger.warning(f"Configuration file not found: {filename}")
                return
            
            # Load from file
            with open(filename, "r") as f:
                config_dict = json.load(f)
            
            # Convert to configs
            self.configs = {
                name: Config(
                    name=name,
                    value=config["value"],
                    description=config.get("description"),
                    timestamp=datetime.fromisoformat(config["timestamp"]),
                    metadata=config.get("metadata")
                )
                for name, config in config_dict.items()
            }
            
            logger.info(f"Configurations loaded from {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise
    
    def reset(self) -> None:
        """Reset configuration manager."""
        self.configs = {}
        logger.info("Configuration manager reset")

__all__ = ['ConfigManager', 'Config'] 