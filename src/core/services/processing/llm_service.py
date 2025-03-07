"""
LLM service for the LangGraph project.
This service provides language model management capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

class LLMService:
    """Service for managing language models in the LangGraph system."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.llms: Dict[str, BaseLLM] = {}
        self.embeddings: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
    
    def create_llm(
        self,
        name: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> BaseLLM:
        """
        Create a new language model.
        
        Args:
            name: LLM name
            model_name: Model name (default: gpt-3.5-turbo)
            temperature: Temperature for sampling (default: 0.7)
            max_tokens: Maximum tokens to generate (default: None)
            **kwargs: Additional arguments for the model
            
        Returns:
            BaseLLM: Created language model
        """
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        self.llms[name] = llm
        return llm
    
    def get_llm(self, name: str) -> Optional[BaseLLM]:
        """
        Get a language model by name.
        
        Args:
            name: LLM name
            
        Returns:
            Optional[BaseLLM]: Language model if found, None otherwise
        """
        return self.llms.get(name)
    
    def create_embeddings(
        self,
        name: str,
        model_name: str = "text-embedding-ada-002",
        **kwargs: Any
    ) -> Any:
        """
        Create new embeddings.
        
        Args:
            name: Embeddings name
            model_name: Model name (default: text-embedding-ada-002)
            **kwargs: Additional arguments for the embeddings
            
        Returns:
            Any: Created embeddings
        """
        embeddings = OpenAIEmbeddings(
            model=model_name,
            **kwargs
        )
        self.embeddings[name] = embeddings
        return embeddings
    
    def get_embeddings(self, name: str) -> Optional[Any]:
        """
        Get embeddings by name.
        
        Args:
            name: Embeddings name
            
        Returns:
            Optional[Any]: Embeddings if found, None otherwise
        """
        return self.embeddings.get(name)
    
    def generate(
        self,
        name: str,
        prompt: str,
        **kwargs: Any
    ) -> str:
        """
        Generate text using a language model.
        
        Args:
            name: LLM name
            prompt: Input prompt
            **kwargs: Additional arguments for generation
            
        Returns:
            str: Generated text
        """
        llm = self.get_llm(name)
        if not llm:
            raise ValueError(f"LLM not found: {name}")
        
        try:
            response = llm(prompt)
            
            # Record in history
            self.history.append({
                'timestamp': datetime.now(),
                'llm': name,
                'action': 'generate',
                'prompt': prompt,
                'response': response
            })
            
            return response
            
        except Exception as e:
            # Record error in history
            self.history.append({
                'timestamp': datetime.now(),
                'llm': name,
                'action': 'generate',
                'prompt': prompt,
                'error': str(e)
            })
            raise
    
    def embed_text(
        self,
        name: str,
        text: Union[str, List[str]],
        **kwargs: Any
    ) -> Union[List[float], List[List[float]]]:
        """
        Embed text using embeddings.
        
        Args:
            name: Embeddings name
            text: Text to embed
            **kwargs: Additional arguments for embedding
            
        Returns:
            Union[List[float], List[List[float]]]: Embeddings
        """
        embeddings = self.get_embeddings(name)
        if not embeddings:
            raise ValueError(f"Embeddings not found: {name}")
        
        try:
            result = embeddings.embed_documents(text if isinstance(text, list) else [text])
            
            # Record in history
            self.history.append({
                'timestamp': datetime.now(),
                'embeddings': name,
                'action': 'embed_text',
                'text': text,
                'result': result
            })
            
            return result[0] if not isinstance(text, list) else result
            
        except Exception as e:
            # Record error in history
            self.history.append({
                'timestamp': datetime.now(),
                'embeddings': name,
                'action': 'embed_text',
                'text': text,
                'error': str(e)
            })
            raise
    
    def get_history(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the LLM usage history.
        
        Args:
            name: Optional LLM/embeddings name to filter history
            
        Returns:
            List[Dict[str, Any]]: LLM usage history
        """
        if name:
            return [entry for entry in self.history if entry.get('llm') == name or entry.get('embeddings') == name]
        return self.history
    
    def clear_history(self, name: Optional[str] = None) -> None:
        """
        Clear the LLM usage history.
        
        Args:
            name: Optional LLM/embeddings name to clear history for
        """
        if name:
            self.history = [entry for entry in self.history if entry.get('llm') != name and entry.get('embeddings') != name]
        else:
            self.history = []
    
    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset LLMs and embeddings.
        
        Args:
            name: Optional LLM/embeddings name to reset
        """
        if name:
            self.llms.pop(name, None)
            self.embeddings.pop(name, None)
        else:
            self.llms = {}
            self.embeddings = {}
            self.history = []

__all__ = ['LLMService'] 