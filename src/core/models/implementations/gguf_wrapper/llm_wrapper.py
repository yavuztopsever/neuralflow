"""
Local LLM wrapper for GGUF models using LlamaCpp
"""

import os
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

# Import LangChain wrappers for LlamaCpp
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from langchain.prompts.chat import ChatPromptTemplate

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

class LocalLLM:
    """Wrapper for local GGUF language models with memory optimization."""
    
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        top_k: int = 40,
        context_window: int = 4096,
        verbose: bool = False,
        n_gpu_layers: int = 1,  # Default to 1 for memory efficiency
        n_threads: Optional[int] = None,  # None means auto-detect
        n_batch: int = 512  # Batch size for inference
    ):
        """
        Initialize a local LLM using a GGUF model with memory optimizations.
        
        Args:
            model_path: Path to the GGUF model file
            temperature: Sampling temperature (higher = more creative, lower = more deterministic)
            max_tokens: Maximum number of tokens to generate per response
            top_p: Top-p sampling parameter (nucleus sampling)
            top_k: Top-k sampling parameter
            context_window: Size of the context window
            verbose: Whether to print verbose logs
            n_gpu_layers: Number of layers to offload to GPU (1 for memory efficiency on Mac M4)
            n_threads: Number of CPU threads to use (None = auto-detect)
            n_batch: Batch size for inference (512 is a good default for memory efficiency)
        """
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
        self.llm = None
        self.backend = None
        
        # Initialize the model
        self._initialize_model()
    
    def __del__(self):
        """Cleanup when object is deleted"""
        try:
            # Clear reference to the underlying model
            self.llm = None
            # Force garbage collection
            import gc
            gc.collect()
        except:
            pass
    
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
            n_threads=self.n_threads or min(4, os.cpu_count() or 4),  # Limit threads
            n_batch=self.n_batch,  # Use batch size parameter
            verbose=False,  # Less verbose for memory efficiency
            f16_kv=True,    # Use 16-bit key/value memory for attention
            use_mlock=False, # Don't lock memory
            streaming=False, # Non-streaming mode saves some memory
        )
        self.backend = "langchain"
        logger.info(f"Initialized LLM with LangChain backend: {self.model_path}")
    
    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt for the model."""
        # Avoid duplicate leading tokens
        if prompt.startswith("<s>[INST]"):
            return prompt
        return f"<s>[INST] {prompt} [/INST]"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response to the given prompt."""
        # Override instance parameters with any provided kwargs
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        
        # Format the prompt for the model
        formatted_prompt = self._format_prompt(prompt)
        
        if self.backend == "langchain":
            # Generate with LangChain
            response = self.llm.invoke(
                formatted_prompt,
                temperature=temperature,
                max_tokens=max_tokens, 
                top_p=top_p,
                top_k=top_k
            )
            return response
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Allow calling the model directly."""
        return self.generate(prompt, **kwargs)


def model_context(model_obj):
    """Create a context manager for safe model usage that cleans up memory.
    
    Usage:
        with model_context(get_llm()) as model:
            result = model.generate("Your prompt")
    """
    return ModelContextManager(model_obj)

def get_llm(
    model_name: Optional[str] = None, 
    model_path: Optional[str] = None,
    **kwargs
) -> LocalLLM:
    """
    Get a local LLM instance optimized for memory-constrained environments.
    
    Args:
        model_name: Name of the model file (without path)
        model_path: Full path to the model file (overrides model_name)
        **kwargs: Additional parameters for LocalLLM initialization
        
    Returns:
        LocalLLM instance
    """
    # Try to import config to get memory-optimized settings
    try:
        from config.config import Config
        # Apply memory optimization defaults, which can be overridden by kwargs
        gguf_kwargs = {
            'n_gpu_layers': kwargs.pop('n_gpu_layers', getattr(Config, 'GGUF_N_GPU_LAYERS', 1)),
            'n_threads': kwargs.pop('n_threads', getattr(Config, 'GGUF_N_THREADS', min(4, os.cpu_count() or 4))),
            'context_window': kwargs.pop('context_window', getattr(Config, 'GGUF_CONTEXT_WINDOW', 4096)),
            'max_tokens': kwargs.pop('max_tokens', getattr(Config, 'GGUF_MAX_TOKENS', 512))
        }
        # Merge with provided kwargs
        kwargs.update(gguf_kwargs)
        logger.info(f"Using memory-optimized settings: GPU layers={gguf_kwargs['n_gpu_layers']}, " +
                    f"Threads={gguf_kwargs['n_threads']}, " +
                    f"Context window={gguf_kwargs['context_window']}")
    except ImportError:
        logger.warning("Could not import Config for memory optimizations, using defaults")
    
    # If model_path is provided, use it directly
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        return LocalLLM(model_path=model_path, **kwargs)
    
    # If only model_name is provided, look for it in the default directory
    if model_name:
        default_path = os.path.join(DEFAULT_MODELS_DIR, model_name)
        if os.path.exists(default_path):
            return LocalLLM(model_path=default_path, **kwargs)
    
    # If neither is specified, or the file wasn't found,
    # try to find any .gguf model in the default directory
    if not model_name and not model_path:
        # Try to find the smallest usable GGUF model first
        best_model = None
        best_size = float('inf')
        
        for file in os.listdir(DEFAULT_MODELS_DIR):
            if file.endswith(".gguf"):
                file_path = os.path.join(DEFAULT_MODELS_DIR, file)
                file_size = os.path.getsize(file_path)
                
                # Prefer smaller models for memory efficiency
                if file_size < best_size:
                    best_model = file_path
                    best_size = file_size
        
        if best_model:
            logger.info(f"Using found model (smallest size): {best_model}")
            return LocalLLM(model_path=best_model, **kwargs)
                
        # Fallback to any GGUF file if we couldn't determine sizes
        for file in os.listdir(DEFAULT_MODELS_DIR):
            if file.endswith(".gguf"):
                logger.info(f"Using found model: {file}")
                return LocalLLM(model_path=os.path.join(DEFAULT_MODELS_DIR, file), **kwargs)
    
    # If nothing found, raise error
    raise FileNotFoundError(
        f"No GGUF model found. Please place a .gguf file in {DEFAULT_MODELS_DIR} or specify a valid model_path."
    )

class GGUFModelWrapper(LocalLLM):
    """Wrapper class for GGUF models that provides additional functionality for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_mode = False
        self.last_prompt = None
        self.last_response = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response with test mode support."""
        self.last_prompt = prompt
        response = super().generate(prompt, **kwargs)
        self.last_response = response
        return response
    
    def set_test_mode(self, enabled: bool = True):
        """Enable or disable test mode."""
        self.test_mode = enabled
    
    def get_last_interaction(self) -> tuple:
        """Get the last prompt and response for testing."""
        return self.last_prompt, self.last_response