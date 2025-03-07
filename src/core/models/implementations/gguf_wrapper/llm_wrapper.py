"""
Local LLM wrapper for GGUF models using LlamaCpp
"""

import os
import logging
import psutil
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from datetime import datetime

# Import LangChain wrappers for LlamaCpp
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from langchain.prompts.chat import ChatPromptTemplate

from core.models.base_model import BaseNamedModel
from core.models.implementations.llm.base import BaseLLM
from config.models import GGUFModelConfig, get_model_config

logger = logging.getLogger(__name__)

# Default models directory
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gguf_llm")

class ModelContextManager:
    """Context manager for GGUF models to ensure proper cleanup."""
    
    def __init__(self, model_obj):
        self.model_obj = model_obj
        
    def __enter__(self):
        return self.model_obj
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clear all references to the model
        if hasattr(self.model_obj, 'llm'):
            # For LangChain models
            if hasattr(self.model_obj.llm, '_llm'):
                setattr(self.model_obj.llm, '_llm', None)
            # Set to None to help garbage collection
            setattr(self.model_obj, 'llm', None)
        
        # Force garbage collection
        import gc
        gc.collect()
        return False  # Don't suppress exceptions

class LocalLLM(BaseNamedModel):
    """Wrapper for local GGUF language models with memory optimization."""
    
    def __init__(
        self,
        name: str,
        model_path: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        top_k: int = 40,
        context_window: int = 4096,
        verbose: bool = False,
        n_gpu_layers: int = 1,
        n_threads: Optional[int] = None,
        n_batch: int = 512,
        description: Optional[str] = None,
        memory_optimized: bool = True
    ):
        """
        Initialize a local LLM using a GGUF model with memory optimizations.
        """
        super().__init__(name=name, description=description)
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.context_window = context_window
        self.verbose = verbose
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.memory_optimized = memory_optimized
        self.llm = None
        self.backend = None
        self.last_used: Optional[datetime] = None
        
        # Check available memory before initialization
        if self.memory_optimized:
            self._check_memory_availability()
        
        # Initialize the model
        self._initialize_model()

    def _check_memory_availability(self):
        """Check if there's enough memory available for the model."""
        try:
            available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # Convert to GB
            model_size = os.path.getsize(self.model_path) / (1024 * 1024 * 1024)  # Convert to GB
            
            # For 16GB RAM, we want to keep at least 4GB free
            if available_memory < 4:
                logger.warning(f"Low memory available: {available_memory:.2f}GB. Model size: {model_size:.2f}GB")
                # Reduce context window and batch size for memory efficiency
                self.context_window = min(self.context_window, 2048)
                self.n_batch = min(self.n_batch, 128)
                self.n_threads = min(self.n_threads or 6, 4)
        except Exception as e:
            logger.warning(f"Could not check memory availability: {e}")

    def _initialize_model(self):
        """Initialize the LLM based on the available libraries."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        try:
            self._try_initialize_langchain()
        except Exception as e:
            logger.warning(f"LangChain initialization failed: {e}. Unable to initialize model.")

    def _try_initialize_langchain(self):
        # Set up callback manager for streaming output if verbose
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]) if self.verbose else None

        # Initialize the model with memory optimizations
        self.llm = LlamaCpp(
            model_path=self.model_path,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            n_ctx=self.context_window,
            callback_manager=callback_manager,
            n_gpu_layers=self.n_gpu_layers,
            n_threads=self.n_threads or min(6, os.cpu_count() or 6),  # Optimized for M4
            n_batch=self.n_batch,
            verbose=False,
            f16_kv=True,    # Use 16-bit key/value memory for attention
            use_mlock=False, # Don't lock memory
            streaming=False, # Non-streaming mode saves some memory
            use_mmap=True,   # Use memory mapping for better memory management
            use_mlock=False, # Don't lock memory
            n_gqa=8,        # Optimize for M4
            rms_norm_eps=1e-5,  # Optimize for M4
            rope_freq_base=10000,  # Optimize for M4
            rope_freq_scale=1.0,   # Optimize for M4
        )
        self.backend = "langchain"
        logger.info(f"Initialized LLM with LangChain backend: {self.model_path}")

    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt for the model."""
        if prompt.startswith("<s>[INST]"):
            return prompt
        return f"<s>[INST] {prompt} [/INST]"

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response to the given prompt."""
        # Check memory before generation
        if self.memory_optimized:
            self._check_memory_availability()
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        
        formatted_prompt = self._format_prompt(prompt)
        
        if self.backend == "langchain":
            response = self.llm.invoke(
                formatted_prompt,
                temperature=temperature,
                max_tokens=max_tokens, 
                top_p=top_p,
                top_k=top_k
            )
            self.last_used = datetime.now()
            return response

    def __call__(self, prompt: str, **kwargs) -> str:
        """Allow calling the model directly."""
        return self.generate(prompt, **kwargs)

class GGUFModelWrapper(LocalLLM, BaseLLM):
    """Wrapper for GGUF models that implements the BaseLLM interface."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_mode = False
        self.last_interaction = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response to the given prompt."""
        if self.test_mode:
            return "Test response"
        response = super().generate(prompt, **kwargs)
        self.last_interaction = (prompt, response)
        return response
    
    def set_test_mode(self, enabled: bool = True):
        """Enable or disable test mode."""
        self.test_mode = enabled
    
    def get_last_interaction(self) -> tuple:
        """Get the last prompt and response."""
        return self.last_interaction or ("", "")

def model_context(model_obj):
    """Create a context manager for safe model usage that cleans up memory."""
    return ModelContextManager(model_obj)

def get_llm(
    model_name: str,
    **kwargs
) -> GGUFModelWrapper:
    """
    Get a local LLM instance optimized for memory-constrained environments.
    
    Args:
        model_name: Name of the model to load (must be one of the available models)
        **kwargs: Additional parameters to override model configuration
        
    Returns:
        GGUFModelWrapper instance
    """
    # Get model configuration
    model_config = get_model_config(model_name)
    if not model_config:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(get_model_config.keys())}")
    
    # Create kwargs with model configuration
    llm_kwargs = {
        'name': model_config.name,
        'model_path': model_config.path,
        'description': model_config.description,
        'context_window': model_config.context_window,
        'max_tokens': model_config.max_tokens,
        'temperature': model_config.temperature,
        'top_p': model_config.top_p,
        'top_k': model_config.top_k,
        'n_threads': model_config.n_threads,
        'n_batch': model_config.n_batch,
        'n_gpu_layers': model_config.n_gpu_layers,
        'memory_optimized': model_config.memory_optimized
    }
    
    # Override with any provided kwargs
    llm_kwargs.update(kwargs)
    
    return GGUFModelWrapper(**llm_kwargs)