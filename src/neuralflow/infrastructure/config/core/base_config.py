"""
Base infrastructure configuration for NeuralFlow.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: Dict[str, Any] = Field(default_factory=dict)

class InfrastructureConfig(BaseModel):
    """Infrastructure configuration."""
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache_dir: str = "cache"
    temp_dir: str = "temp"
    max_workers: int = 4
    debug_mode: bool = False
