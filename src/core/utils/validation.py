"""
Validation utilities for the LangGraph project.
These utilities provide input/output validation and configuration validation
integrated with LangChain capabilities.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import re
from datetime import datetime

from langchain.schema import Document, BaseMessage
from langchain.vectorstores import VectorStore
from langchain.embeddings import OpenAIEmbeddings

def validate_input(input_data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate input data against a schema.
    
    Args:
        input_data: The input data to validate
        schema: The schema to validate against
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        for key, value in schema.items():
            if key not in input_data:
                return False
            if not isinstance(input_data[key], value):
                return False
        return True
    except Exception:
        return False

def validate_output(output_data: Any, expected_type: type) -> bool:
    """
    Validate output data against expected type.
    
    Args:
        output_data: The output data to validate
        expected_type: The expected type
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        return isinstance(output_data, expected_type)
    except Exception:
        return False

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration data.
    
    Args:
        config: The configuration to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    required_fields = {
        "api_key": str,
        "model_name": str,
        "temperature": (int, float),
        "max_tokens": int
    }
    
    try:
        for field, field_type in required_fields.items():
            if field not in config:
                return False
            if not isinstance(config[field], field_type):
                return False
        return True
    except Exception:
        return False

def validate_file_path(file_path: Union[str, Path], check_exists: bool = True) -> bool:
    """
    Validate file path.
    
    Args:
        file_path: The file path to validate
        check_exists: Whether to check if file exists
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        path = Path(file_path)
        if check_exists and not path.exists():
            return False
        return True
    except Exception:
        return False

def validate_document(doc: Document) -> bool:
    """
    Validate LangChain Document.
    
    Args:
        doc: The document to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        if not isinstance(doc, Document):
            return False
        if not doc.page_content:
            return False
        return True
    except Exception:
        return False

def validate_vector_store(vector_store: VectorStore) -> bool:
    """
    Validate LangChain VectorStore.
    
    Args:
        vector_store: The vector store to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        if not isinstance(vector_store, VectorStore):
            return False
        return True
    except Exception:
        return False

def validate_embeddings(embeddings: OpenAIEmbeddings) -> bool:
    """
    Validate OpenAI embeddings.
    
    Args:
        embeddings: The embeddings to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        if not isinstance(embeddings, OpenAIEmbeddings):
            return False
        return True
    except Exception:
        return False

def validate_messages(messages: List[BaseMessage]) -> bool:
    """
    Validate list of LangChain messages.
    
    Args:
        messages: The messages to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        if not isinstance(messages, list):
            return False
        for msg in messages:
            if not isinstance(msg, BaseMessage):
                return False
        return True
    except Exception:
        return False

def validate_json(data: str) -> bool:
    """
    Validate JSON string.
    
    Args:
        data: The JSON string to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        json.loads(data)
        return True
    except json.JSONDecodeError:
        return False

def validate_timestamp(timestamp: str) -> bool:
    """
    Validate ISO format timestamp.
    
    Args:
        timestamp: The timestamp to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        datetime.fromisoformat(timestamp)
        return True
    except ValueError:
        return False

def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: The URL to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(url))

__all__ = [
    'validate_input',
    'validate_output',
    'validate_config',
    'validate_file_path',
    'validate_document',
    'validate_vector_store',
    'validate_embeddings',
    'validate_messages',
    'validate_json',
    'validate_timestamp',
    'validate_url'
] 