"""
Logging utilities for the LangGraph project.
These utilities provide logging capabilities integrated with LangChain.
"""

from typing import Any, Dict, Optional, List, Union
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage, LLMResult
from langchain.schema.document import Document
from langchain.vectorstores import VectorStore

class LangGraphLogger:
    """Logger class for the LangGraph project."""
    
    def __init__(
        self,
        name: str = "langgraph",
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        log_format: Optional[str] = None
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_level: Logging level
            log_file: Optional log file path
            log_format: Optional log format string
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Create formatters
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        
        # Create handlers
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Add handlers to logger
        for handler in handlers:
            self.logger.addHandler(handler)
        
        # Initialize metrics
        self.metrics: Dict[str, Any] = {
            'start_time': datetime.now(),
            'message_count': 0,
            'error_count': 0,
            'warning_count': 0,
            'info_count': 0
        }
    
    def log_message(self, message: str, level: int = logging.INFO) -> None:
        """
        Log a message with the specified level.
        
        Args:
            message: The message to log
            level: Logging level
        """
        self.logger.log(level, message)
        
        # Update metrics
        if level == logging.ERROR:
            self.metrics['error_count'] += 1
        elif level == logging.WARNING:
            self.metrics['warning_count'] += 1
        else:
            self.metrics['info_count'] += 1
        self.metrics['message_count'] += 1
    
    def log_dict(self, data: Dict[str, Any], level: int = logging.INFO) -> None:
        """
        Log a dictionary as JSON.
        
        Args:
            data: The dictionary to log
            level: Logging level
        """
        self.log_message(json.dumps(data, indent=2), level)
    
    def log_langchain_message(self, message: BaseMessage, level: int = logging.INFO) -> None:
        """
        Log a LangChain message.
        
        Args:
            message: The LangChain message to log
            level: Logging level
        """
        self.log_dict({
            'type': type(message).__name__,
            'content': message.content,
            'metadata': message.additional_kwargs
        }, level)
    
    def log_langchain_result(self, result: LLMResult, level: int = logging.INFO) -> None:
        """
        Log a LangChain LLM result.
        
        Args:
            result: The LLM result to log
            level: Logging level
        """
        self.log_dict({
            'type': 'LLMResult',
            'generations': [
                {
                    'text': gen[0].text,
                    'logprobs': gen[0].logprobs
                }
                for gen in result.generations
            ]
        }, level)
    
    def log_document(self, document: Document, level: int = logging.INFO) -> None:
        """
        Log a LangChain document.
        
        Args:
            document: The document to log
            level: Logging level
        """
        self.log_dict({
            'type': 'Document',
            'content': document.page_content,
            'metadata': document.metadata
        }, level)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get logging metrics.
        
        Returns:
            Dict[str, Any]: Logging metrics
        """
        self.metrics['duration'] = (datetime.now() - self.metrics['start_time']).total_seconds()
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset logging metrics."""
        self.metrics = {
            'start_time': datetime.now(),
            'message_count': 0,
            'error_count': 0,
            'warning_count': 0,
            'info_count': 0
        }

class LangChainCallbackHandler(BaseCallbackHandler):
    """Callback handler for LangChain that integrates with LangGraphLogger."""
    
    def __init__(self, logger: LangGraphLogger):
        """
        Initialize the callback handler.
        
        Args:
            logger: LangGraphLogger instance
        """
        self.logger = logger
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Handle LLM start."""
        self.logger.log_dict({
            'event': 'llm_start',
            'model': serialized.get('name', 'unknown'),
            'prompts': prompts
        })
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Handle LLM end."""
        self.logger.log_langchain_result(response)
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Handle LLM error."""
        self.logger.log_message(f"LLM Error: {str(error)}", logging.ERROR)
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Handle chain start."""
        self.logger.log_dict({
            'event': 'chain_start',
            'chain': serialized.get('name', 'unknown'),
            'inputs': inputs
        })
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Handle chain end."""
        self.logger.log_dict({
            'event': 'chain_end',
            'outputs': outputs
        })
    
    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Handle chain error."""
        self.logger.log_message(f"Chain Error: {str(error)}", logging.ERROR)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Handle tool start."""
        self.logger.log_dict({
            'event': 'tool_start',
            'tool': serialized.get('name', 'unknown'),
            'input': input_str
        })
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Handle tool end."""
        self.logger.log_dict({
            'event': 'tool_end',
            'output': output
        })
    
    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Handle tool error."""
        self.logger.log_message(f"Tool Error: {str(error)}", logging.ERROR)

__all__ = [
    'LangGraphLogger',
    'LangChainCallbackHandler'
] 