"""
Main application configuration for NeuralFlow.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from .infrastructure.config.core.base_config import InfrastructureConfig
from .frontend.config import FrontendConfig
from .tools.base import ToolConfig

class ApplicationConfig(BaseModel):
    """Application configuration."""
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    frontend: FrontendConfig = Field(default_factory=FrontendConfig)
    tools: Dict[str, ToolConfig] = Field(default_factory=dict)
    debug: bool = False
