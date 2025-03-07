"""
Configuration for different GGUF models.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path

@dataclass
class GGUFModelConfig:
    """Configuration for a GGUF model."""
    name: str
    path: str
    description: str
    context_window: int = 4096
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    n_threads: int = 4
    n_batch: int = 512
    n_gpu_layers: int = -1
    memory_optimized: bool = True

# Available GGUF models with M4-specific optimizations
AVAILABLE_MODELS: Dict[str, GGUFModelConfig] = {
    "deepseek-8b": GGUFModelConfig(
        name="deepseek-8b",
        path="/Volumes/HomeX/yavuztopsever/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        description="DeepSeek R1 Distill Llama 8B (Q4_K_M)",
        context_window=4096,  # Reduced for memory efficiency
        max_tokens=1024,      # Reduced for memory efficiency
        n_threads=6,          # Optimized for M4
        n_batch=256,          # Reduced batch size for memory efficiency
        n_gpu_layers=1,       # Use minimal GPU layers
        memory_optimized=True
    ),
    "deepseek-1.5b": GGUFModelConfig(
        name="deepseek-1.5b",
        path="/Volumes/HomeX/yavuztopsever/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        description="DeepSeek R1 Distill Qwen 1.5B (Q4_K_M)",
        context_window=4096,
        max_tokens=2048,      # Can use more tokens for smaller model
        n_threads=6,          # Optimized for M4
        n_batch=512,          # Can use larger batch size for smaller model
        n_gpu_layers=1,
        memory_optimized=True
    ),
    "deepseek-7b": GGUFModelConfig(
        name="deepseek-7b",
        path="/Volumes/HomeX/yavuztopsever/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        description="DeepSeek R1 Distill Qwen 7B (Q4_K_M)",
        context_window=4096,  # Reduced for memory efficiency
        max_tokens=1024,      # Reduced for memory efficiency
        n_threads=6,          # Optimized for M4
        n_batch=256,          # Reduced batch size for memory efficiency
        n_gpu_layers=1,       # Use minimal GPU layers
        memory_optimized=True
    ),
    "qwen-coder-7b": GGUFModelConfig(
        name="qwen-coder-7b",
        path="/Volumes/HomeX/yavuztopsever/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        description="Qwen 2.5 Coder 7B Instruct (Q4_K_M)",
        context_window=4096,  # Reduced for memory efficiency
        max_tokens=1024,      # Reduced for memory efficiency
        temperature=0.2,      # Lower temperature for code generation
        n_threads=6,          # Optimized for M4
        n_batch=256,          # Reduced batch size for memory efficiency
        n_gpu_layers=1,       # Use minimal GPU layers
        memory_optimized=True
    )
}

def get_model_config(model_name: str) -> Optional[GGUFModelConfig]:
    """Get configuration for a specific model."""
    return AVAILABLE_MODELS.get(model_name)

def list_available_models() -> Dict[str, str]:
    """List all available models with their descriptions."""
    return {name: config.description for name, config in AVAILABLE_MODELS.items()}

def validate_model_path(model_path: str) -> bool:
    """Validate if a model file exists at the given path."""
    return Path(model_path).exists() 