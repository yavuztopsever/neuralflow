# GGUF LLM Wrapper Module
"""This module provides wrappers for GGUF-based local language models."""

import logging
import os

# Import the LLM implementation
try:
    from .llm_wrapper import LocalLLM, get_llm, model_context
    LLM_IMPLEMENTATION = "real"
    logging.info("Using real GGUF LLM implementation")
except ImportError:
    # Provide clear error message if llama-cpp-python is not available
    logging.error("GGUF LLM implementation requires llama-cpp-python. Install with: pip install llama-cpp-python")
    
    try:
        # Try to use mock LLM implementation instead
        from .mock_llm import MockLLM
        
        def get_llm(*args, **kwargs):
            """Return a mock LLM implementation"""
            return MockLLM(*args, **kwargs)
            
        # Alias for compatibility
        LocalLLM = MockLLM
        
        def model_context(model_obj):
            """Simple pass-through for the mock implementation"""
            return model_obj
            
        LLM_IMPLEMENTATION = "mock"
        logging.info("Using mock LLM implementation")
    except ImportError:
        # Define fallback classes for graceful failure
        class LocalLLM:
            def __init__(self, *args, **kwargs):
                raise ImportError("LocalLLM requires llama-cpp-python. Install with: pip install llama-cpp-python")
            
            def generate(self, *args, **kwargs):
                raise ImportError("LocalLLM requires llama-cpp-python. Install with: pip install llama-cpp-python")
        
        def get_llm(*args, **kwargs):
            raise ImportError("GGUF models require llama-cpp-python. Install with: pip install llama-cpp-python")
        
        def model_context(*args, **kwargs):
            raise ImportError("GGUF models require llama-cpp-python. Install with: pip install llama-cpp-python")
        
        LLM_IMPLEMENTATION = "unavailable"