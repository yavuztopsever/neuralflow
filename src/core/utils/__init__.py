"""
Core utilities for the LangGraph project.
These utilities provide various capabilities integrated with LangChain.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import logging
from datetime import datetime

from langchain.schema import Document, BaseMessage
from langchain.vectorstores import VectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)

# Import utility modules
from .validation import (
    validate_input,
    validate_output,
    validate_config,
    validate_file_path
)

from .text_processing import (
    clean_text,
    extract_entities,
    summarize_text,
    chunk_text,
    analyze_sentiment,
    extract_keywords,
    translate_text,
    extract_phrases
)

from .error_handling import (
    LangGraphError,
    ValidationError,
    ProcessingError,
    StorageError,
    SearchError,
    GraphError,
    ErrorHandler,
    error_handler,
    validate_langchain_input,
    validate_langchain_output,
    validate_vector_store,
    validate_embeddings,
    validate_document,
    validate_messages,
    validate_llm_result
)

from .logging_manager import (
    LangGraphLogger,
    LangChainCallbackHandler
)

from .document_processor import DocumentProcessor

from .storage_manager import StorageManager

class LangChainUtils:
    """Utility class for LangChain integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.logger = get_logger(__name__)
    
    def process_documents(self, file_path: Union[str, Path], doc_type: str = "auto") -> List[Document]:
        """Process documents using appropriate LangChain loader."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Select appropriate loader based on file type
            if doc_type == "auto":
                if file_path.suffix.lower() == ".pdf":
                    loader = PDFLoader(str(file_path))
                elif file_path.suffix.lower() == ".md":
                    loader = UnstructuredMarkdownLoader(str(file_path))
                elif file_path.is_dir():
                    loader = DirectoryLoader(str(file_path))
                else:
                    loader = TextLoader(str(file_path))
            else:
                loader_class = {
                    "pdf": PDFLoader,
                    "markdown": UnstructuredMarkdownLoader,
                    "directory": DirectoryLoader,
                    "text": TextLoader
                }.get(doc_type.lower(), TextLoader)
                loader = loader_class(str(file_path))
            
            # Load and process documents
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def create_vector_store(self, documents: List[Document]) -> VectorStore:
        """Create a vector store from documents."""
        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)
            
            return VectorStore.from_texts(
                texts=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
        except Exception as e:
            self.logger.error(f"Error creating vector store: {e}")
            raise
    
    def save_vector_store(self, vector_store: VectorStore, path: Union[str, Path]):
        """Save vector store to disk."""
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(str(path))
        except Exception as e:
            self.logger.error(f"Error saving vector store: {e}")
            raise
    
    def load_vector_store(self, path: Union[str, Path]) -> VectorStore:
        """Load vector store from disk."""
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Vector store not found: {path}")
            return VectorStore.load_local(str(path), self.embeddings)
        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
            raise
    
    def process_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Process messages and extract relevant information."""
        try:
            processed_messages = []
            for msg in messages:
                processed_msg = {
                    "type": msg.type,
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                }
                if hasattr(msg, "additional_kwargs"):
                    processed_msg.update(msg.additional_kwargs)
                processed_messages.append(processed_msg)
            return processed_messages
        except Exception as e:
            self.logger.error(f"Error processing messages: {e}")
            raise

__all__ = [
    # Text processing
    'clean_text',
    'extract_entities',
    'summarize_text',
    'chunk_text',
    'analyze_sentiment',
    'extract_keywords',
    'translate_text',
    'extract_phrases',
    
    # Error handling
    'LangGraphError',
    'ValidationError',
    'ProcessingError',
    'StorageError',
    'SearchError',
    'GraphError',
    'ErrorHandler',
    'error_handler',
    'validate_langchain_input',
    'validate_langchain_output',
    'validate_vector_store',
    'validate_embeddings',
    'validate_document',
    'validate_messages',
    'validate_llm_result',
    
    # Logging
    'LangGraphLogger',
    'LangChainCallbackHandler',
    
    # Document processing
    'DocumentProcessor',
    
    # Storage
    'StorageManager',
    
    # Core utility functions
    'validate_input',
    'validate_output',
    'validate_config',
    'validate_file_path',
    'handle_error',
    'log_error',
    'setup_logging',
    'get_logger',
    'process_document',
    'extract_metadata',
    'save_to_disk',
    'load_from_disk',
    
    # Classes
    'CustomError',
    'LogManager',
    'LangChainUtils'
] 