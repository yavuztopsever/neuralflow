"""
Configuration management utilities for the LangGraph application.
This module provides functionality for managing application configuration settings.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from utils.common.text import TextProcessor
from utils.error.handlers import ErrorHandler

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages application configuration settings."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.text_processor = TextProcessor()
        self._config_cache = {}
        self._last_load = {}
        self._initialize_config()
    
    def _initialize_config(self):
        """Initialize configuration settings."""
        try:
            # Load default configuration
            self._load_default_config()
            
            # Load environment-specific configuration
            self._load_environment_config()
            
            # Load user configuration if exists
            self._load_user_config()
            
            logger.info("Initialized configuration settings")
        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            raise
    
    def _load_default_config(self):
        """Load default configuration settings."""
        try:
            default_config = {
                'app': {
                    'name': 'LangGraph',
                    'version': '1.0.0',
                    'debug': False,
                    'log_level': 'INFO'
                },
                'storage': {
                    'documents_dir': 'src/storage/documents',
                    'notes_dir': 'src/storage/notes',
                    'state_dir': 'src/storage/state',
                    'vector_db_dir': 'src/storage/vector'
                },
                'processing': {
                    'chunk_size': 1000,
                    'chunk_overlap': 100,
                    'max_file_size': 10 * 1024 * 1024  # 10MB
                },
                'search': {
                    'top_k': 5,
                    'min_score': 0.5,
                    'max_results': 100
                },
                'cache': {
                    'enabled': True,
                    'ttl': 3600,
                    'max_size': 1000
                },
                'models': {
                    'embedder': 'sentence-transformers/all-MiniLM-L6-v2',
                    'embedding_dimension': 384,
                    'model_dir': 'models'
                }
            }
            self._config_cache['default'] = default_config
            logger.info("Loaded default configuration")
        except Exception as e:
            logger.error(f"Failed to load default configuration: {e}")
            raise
    
    def _load_environment_config(self):
        """Load environment-specific configuration."""
        try:
            env = os.getenv('APP_ENV', 'development')
            env_config_path = self.config_dir / f"config.{env}.json"
            
            if env_config_path.exists():
                with open(env_config_path, 'r', encoding='utf-8') as f:
                    env_config = json.load(f)
                self._config_cache['environment'] = env_config
                logger.info(f"Loaded {env} environment configuration")
        except Exception as e:
            logger.error(f"Failed to load environment configuration: {e}")
            raise
    
    def _load_user_config(self):
        """Load user-specific configuration."""
        try:
            user_config_path = self.config_dir / "config.user.json"
            if user_config_path.exists():
                with open(user_config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                self._config_cache['user'] = user_config
                logger.info("Loaded user configuration")
        except Exception as e:
            logger.error(f"Failed to load user configuration: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key (can be nested using dots)
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        try:
            # Try to get from cache first
            if key in self._config_cache:
                return self._config_cache[key]
            
            # Split key into parts
            parts = key.split('.')
            value = self._config_cache['default']
            
            # Traverse nested configuration
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, default)
                else:
                    return default
            
            # Cache the result
            self._config_cache[key] = value
            return value
        except Exception as e:
            logger.error(f"Failed to get configuration value for {key}: {e}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: Configuration key (can be nested using dots)
            value: Value to set
        """
        try:
            # Split key into parts
            parts = key.split('.')
            config = self._config_cache['default']
            
            # Create nested structure if needed
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            
            # Set the value
            config[parts[-1]] = value
            
            # Update cache
            self._config_cache[key] = value
            self._last_load[key] = datetime.now()
            
            logger.info(f"Set configuration value for {key}")
        except Exception as e:
            logger.error(f"Failed to set configuration value for {key}: {e}")
            raise
    
    def save(self, config_type: str = 'user') -> None:
        """Save configuration to file.
        
        Args:
            config_type: Type of configuration to save ('user' or 'environment')
        """
        try:
            if config_type not in ['user', 'environment']:
                raise ValueError(f"Invalid configuration type: {config_type}")
            
            config_path = self.config_dir / f"config.{config_type}.json"
            config_data = self._config_cache.get(config_type, {})
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {config_type} configuration")
        except Exception as e:
            logger.error(f"Failed to save {config_type} configuration: {e}")
            raise
    
    def reload(self, config_type: Optional[str] = None) -> None:
        """Reload configuration from files.
        
        Args:
            config_type: Type of configuration to reload (if None, reloads all)
        """
        try:
            if config_type is None:
                self._initialize_config()
            elif config_type == 'environment':
                self._load_environment_config()
            elif config_type == 'user':
                self._load_user_config()
            else:
                raise ValueError(f"Invalid configuration type: {config_type}")
            
            logger.info(f"Reloaded {config_type or 'all'} configuration")
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            raise
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration settings.
        
        Returns:
            Dictionary containing all configuration settings
        """
        try:
            # Merge configurations in order of precedence
            config = self._config_cache['default'].copy()
            
            if 'environment' in self._config_cache:
                self._deep_update(config, self._config_cache['environment'])
            
            if 'user' in self._config_cache:
                self._deep_update(config, self._config_cache['user'])
            
            return config
        except Exception as e:
            logger.error(f"Failed to get all configuration: {e}")
            return {}
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update a dictionary with another dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value 