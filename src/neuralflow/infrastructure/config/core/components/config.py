import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

class Config:
    def __init__(self, env: str = None):
        # Load environment variables from .env file
        load_dotenv()
        
        self.env = env or os.getenv("ENV", "development")
        self.config = self._load_config()
        self._apply_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files in a hierarchical manner."""
        config_dir = Path(__file__).parent
        user_config_dir = Path.home() / '.langgraph'
        user_config_dir.mkdir(exist_ok=True)
        
        # Load configurations in order of precedence
        config = {}
        
        # 1. Load default config (lowest priority)
        with open(config_dir / "default.yaml", "r") as f:
            config.update(yaml.safe_load(f))
        
        # 2. Load environment-specific config
        env_file = config_dir / f"{self.env}.yaml"
        if env_file.exists():
            with open(env_file, "r") as f:
                config.update(yaml.safe_load(f))
        
        # 3. Load user-level config (highest priority)
        user_config_file = user_config_dir / "config.yaml"
        if user_config_file.exists():
            with open(user_config_file, "r") as f:
                user_config = yaml.safe_load(f)
                config = self._deep_update(config, user_config)
        
        # 4. Load environment variables (highest priority)
        env_config = self._load_env_vars()
        config = self._deep_update(config, env_config)
        
        # 5. Apply default values for required settings
        config = self._apply_defaults(config)
        
        return config

    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict

    def _load_env_vars(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        # Load all environment variables, not just LANGGRAPH_ prefixed ones
        for key, value in os.environ.items():
            # Convert SECTION_KEY to section.key
            parts = key.lower().split('_')
            if len(parts) > 1:
                section = parts[0]
                key = '_'.join(parts[1:])
                if section not in env_config:
                    env_config[section] = {}
                env_config[section][key] = value
            else:
                env_config[parts[0]] = value
        return env_config

    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for required settings."""
        defaults = {
            'paths': {
                'data_dir': './storage/data',
                'models_dir': './storage/models',
                'logs_dir': './logs'
            },
            'models': {
                'llm': {
                    'default': 'deepseek-1.5b',
                    'temperature': 0.7,
                    'max_tokens': 1024,
                    'top_p': 0.95,
                    'top_k': 40,
                    'context_window': 4096,
                    'n_threads': 6,
                    'n_batch': 256,
                    'n_gpu_layers': 1
                }
            },
            'memory': {
                'monitor_enabled': True,
                'high_threshold': 40,
                'critical_threshold': 55,
                'max_loaded_models': 1,
                'model_priority': 'deepseek-1.5b'
            },
            'performance': {
                'thread_pool_enabled': True,
                'max_threads': 6,
                'db_pool_size': 2,
                'db_max_overflow': 1,
                'db_pool_recycle': 900
            }
        }
        return self._deep_update(defaults, config)

    def _apply_config(self):
        """Apply configuration to instance attributes."""
        for key, value in self.config.items():
            setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a default fallback."""
        return getattr(self, key, default)

    def update(self, config_dict: Dict[str, Any]):
        """Update configuration with new values."""
        self.config = self._deep_update(self.config, config_dict)
        self._apply_config()

    def save_user_config(self, config_dict: Dict[str, Any]):
        """Save user-level configuration."""
        user_config_dir = Path.home() / '.langgraph'
        user_config_dir.mkdir(exist_ok=True)
        user_config_file = user_config_dir / "config.yaml"
        
        with open(user_config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        # Reload configuration
        self.config = self._load_config()
        self._apply_config()

    def validate(self) -> bool:
        """Validate the configuration."""
        required_settings = [
            'paths.data_dir',
            'paths.models_dir',
            'paths.logs_dir',
            'models.llm.default'
        ]
        
        for setting in required_settings:
            parts = setting.split('.')
            value = self.config
            for part in parts:
                if not isinstance(value, dict) or part not in value:
                    return False
                value = value[part]
            if value is None:
                return False
        
        return True 