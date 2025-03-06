"""
Text processing utilities for the LangGraph application.
This module provides functions for text manipulation, formatting, and validation.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import os
from datetime import datetime
import re
from urllib.parse import urlparse

# Initialize logger
logger = logging.getLogger(__name__)

class TextProcessor:
    """Handles text processing operations."""
    
    @staticmethod
    def format_datetime(dt: Optional[datetime] = None) -> str:
        """Format a datetime object into a standardized string.
        
        Args:
            dt: Datetime object to format (defaults to current time)
            
        Returns:
            Formatted timestamp string
        """
        if dt is None:
            dt = datetime.now()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def split_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            if end > text_length:
                end = text_length
                
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        return chunks
    
    @staticmethod
    def get_nested_dict_value(dictionary: Dict[str, Any], *keys: str, default: Any = None) -> Any:
        """Safely get a value from a nested dictionary.
        
        Args:
            dictionary: Dictionary to search in
            *keys: Keys to traverse
            default: Default value if key doesn't exist
            
        Returns:
            Value found or default value
        """
        current = dictionary
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key, default)
            else:
                return default
        return current

class EnvironmentValidator:
    """Handles environment variable validation."""
    
    @staticmethod
    def validate_environment_variables(required_vars: List[str]) -> Dict[str, str]:
        """Validate that required environment variables are set.
        
        Args:
            required_vars: List of required environment variable names
            
        Returns:
            Dict containing the values of the environment variables
        """
        env_vars = {}
        missing_vars = []
        
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                env_vars[var] = value
            else:
                missing_vars.append(var)
                
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        return env_vars 

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def split_text(text: str, max_length: int, overlap: int = 0) -> List[str]:
    """Split text into chunks with optional overlap."""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Find the last space before max_length
        last_space = text.rfind(" ", start, end)
        if last_space > start:
            end = last_space
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


def is_url(text: str) -> bool:
    """Check if text is a valid URL."""
    try:
        result = urlparse(text)
        return all([result.scheme, result.netloc])
    except:
        return False


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*"
    return re.findall(url_pattern, text)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*"
    return re.sub(url_pattern, "", text)


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Count the number of tokens in text."""
    # This is a simple implementation. In practice, you would use
    # the appropriate tokenizer for the specific model.
    return len(text.split())


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def remove_extra_newlines(text: str) -> str:
    """Remove extra newlines from text."""
    # Replace multiple newlines with a single newline
    text = re.sub(r"\n\s*\n", "\n", text)
    # Remove leading/trailing newlines
    text = text.strip()
    return text 