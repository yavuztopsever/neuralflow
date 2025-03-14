"""
Frontend configuration for NeuralFlow.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = Field(default_factory=list)
    api_prefix: str = "/api/v1"

class UIConfig(BaseModel):
    """UI configuration."""
    theme: str = "light"
    language: str = "en"
    custom_styles: Dict[str, Any] = Field(default_factory=dict)

class FrontendConfig(BaseModel):
    """Frontend configuration."""
    api: APIConfig = Field(default_factory=APIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
