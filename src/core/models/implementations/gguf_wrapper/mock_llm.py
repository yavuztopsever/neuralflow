"""
Mock LLM implementation for testing and development.
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

from core.models.base_model import BaseNamedModel
from core.models.implementations.llm.base import BaseLLM

class MockLLM(BaseNamedModel, BaseLLM):
    """Mock LLM for testing and development without requiring a real model."""
    
    def __init__(
        self, 
        name: str,
        model_path: str = "mock_model.gguf",
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        top_k: int = 40,
        context_window: int = 4096,
        verbose: bool = False,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        n_batch: int = 8,
        description: Optional[str] = None
    ):
        """Initialize a mock LLM that returns predetermined responses."""
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
        self.last_used: Optional[datetime] = None
        
        # Predefined response patterns
        self.responses = {
            "hello": "Hello! How can I help you today?",
            "help": "I'm here to assist you with any questions or tasks you might have.",
            "default": "I've processed your request and have a response for you."
        }
        
        # More detailed responses for specific topics
        self.topic_responses = {
            "watermelon": """
            Watermelons are large, sweet fruits with green rinds and red flesh filled with black seeds (though seedless varieties exist). 
            They belong to the Cucurbitaceae family and are approximately 92% water, making them excellent for hydration.
            
            Nutritionally, watermelons are rich in vitamins A and C, and contain lycopene, an antioxidant that gives them their red color. 
            They originated in Africa and are now grown worldwide, especially in warm climates.
            
            Watermelons are popular in summer and can be eaten fresh, in salads, or as juice. Some cultures also cook with watermelon rinds.
            """,
            
            "langgraph": """
            LangGraph is a framework for building stateful, multi-actor applications using Large Language Models (LLMs). 
            It extends LangChain, providing tools for creating complex, stateful workflows with multiple components that can interact.
            
            Key features:
            - State management for conversations and workflows
            - Checkpointing for resuming interactions
            - Conditional routing between components
            - Memory integration for tracking context
            
            LangGraph is particularly useful for building applications that need to maintain context across multiple interactions,
            such as conversational agents, decision-making systems, and tools that require persistent reasoning over time.
            """,
            
            "python": """
            Python is a high-level, interpreted programming language known for its readability and versatility. Created by Guido van Rossum and first released in 1991, it emphasizes code readability with its notable use of significant whitespace.
            
            Key features of Python include:
            - Simple, easy-to-learn syntax that emphasizes readability
            - Dynamic typing and memory management
            - Support for multiple programming paradigms (object-oriented, imperative, functional, procedural)
            - Comprehensive standard library and rich ecosystem of third-party packages
            - Cross-platform compatibility
            
            Python is widely used in web development, data analysis, artificial intelligence, scientific computing, automation, and many other fields. Its popularity continues to grow, particularly in data science and machine learning applications.
            """
        }
        
        # Log initialization
        logging.info(f"Initialized Mock LLM '{name}' in place of: {model_path}")
        
    async def generate(
        self, 
        prompt: str, 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a response from the mock LLM."""
        # Use provided parameters or defaults
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k
        
        # Determine which response to use based on prompt
        prompt_lower = prompt.lower()
        
        # Check for specific topics first
        for topic, response in self.topic_responses.items():
            if topic in prompt_lower:
                self.last_used = datetime.now()
                return response.strip()
        
        # Then check for general patterns
        if "hello" in prompt_lower or "hi" in prompt_lower.split():
            self.last_used = datetime.now()
            return self.responses["hello"]
        elif "help" in prompt_lower or "assist" in prompt_lower or "what can you" in prompt_lower:
            self.last_used = datetime.now()
            return """
            I can help you with a variety of tasks:
            
            1. Answering questions about various topics
            2. Explaining concepts and providing information
            3. Discussing ideas and offering perspectives
            4. Helping with problem-solving
            5. Providing summaries and analyses
            
            Feel free to ask me about specific subjects, and I'll do my best to assist you!
            """.strip()
        elif "error" in prompt_lower or "fail" in prompt_lower:
            raise Exception("Mock error generated as requested in prompt")
        else:
            # Create response based on prompt content
            self.last_used = datetime.now()
            if len(prompt) > 100:
                extract = prompt[50:100] + "..."
                return f"I understand you're asking about '{extract}'. Here's what I know: This appears to be a complex query that would require specialized knowledge. If you could provide more specific details about what you're looking for, I'd be happy to try to help further."
            else:
                return f"Regarding your query about '{prompt[:30]}{'...' if len(prompt) > 30 else ''}', I can provide the following information: This seems to be a specific question that I'd need more context on. Could you elaborate on what specifically you'd like to know about this topic?"
    
    async def embed(self, text: str) -> list[float]:
        """Generate mock embeddings for input text."""
        # Return a mock embedding vector of appropriate size
        return [0.1] * 1536  # Standard embedding size
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Allow calling the model directly."""
        return asyncio.run(self.generate(prompt, **kwargs))