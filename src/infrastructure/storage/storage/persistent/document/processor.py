"""
Document processing utilities for the LangGraph application.
This module provides functionality for loading, preprocessing, and chunking documents.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from utils.common.text import TextProcessor
from utils.error.handlers import ErrorHandler
from config.config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, preprocessing, and chunking operations."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the document processor.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.text_processor = TextProcessor()
    
    def load_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a document from a file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document content and metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file type is not supported
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            content = self._read_file_content(file_path)
            metadata = self._extract_metadata(file_path)
            return {
                'content': content,
                'metadata': metadata,
                'id': str(file_path.absolute()),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def process_document(self, document: Dict[str, Any], 
                        chunk_size: Optional[int] = None,
                        overlap: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process a document by splitting it into chunks.
        
        Args:
            document: Document dictionary containing content and metadata
            chunk_size: Size of each chunk (defaults to config value)
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of processed document chunks
        """
        chunk_size = chunk_size or self.config.CHUNK_SIZE
        overlap = overlap or self.config.CHUNK_OVERLAP
        
        try:
            chunks = self.text_processor.split_text(
                document['content'],
                chunk_size=chunk_size,
                overlap=overlap
            )
            
            return [
                {
                    'content': chunk,
                    'metadata': {
                        **document['metadata'],
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    },
                    'id': f"{document['id']}_chunk_{i}",
                    'timestamp': datetime.now().isoformat()
                }
                for i, chunk in enumerate(chunks)
            ]
        except Exception as e:
            logger.error(f"Error processing document {document.get('id', 'unknown')}: {e}")
            raise
    
    def process_directory(self, directory: Union[str, Path],
                         file_pattern: str = "*.*",
                         recursive: bool = True) -> List[Dict[str, Any]]:
        """Process all documents in a directory.
        
        Args:
            directory: Path to the directory
            file_pattern: Pattern to match files
            recursive: Whether to process subdirectories
            
        Returns:
            List of processed documents
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        try:
            documents = []
            pattern = "**/" + file_pattern if recursive else file_pattern
            
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    try:
                        doc = self.load_document(file_path)
                        chunks = self.process_document(doc)
                        documents.extend(chunks)
                    except Exception as e:
                        logger.warning(f"Failed to process {file_path}: {e}")
                        continue
                        
            return documents
        except Exception as e:
            logger.error(f"Error processing directory {directory}: {e}")
            raise
    
    def _read_file_content(self, file_path: Path) -> str:
        """Read content from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
            
        Raises:
            ValueError: If the file type is not supported
        """
        suffix = file_path.suffix.lower()
        try:
            if suffix in ['.txt', '.md', '.rst']:
                return file_path.read_text(encoding='utf-8')
            elif suffix in ['.pdf']:
                return self._read_pdf(file_path)
            elif suffix in ['.doc', '.docx']:
                return self._read_docx(file_path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata
        """
        try:
            stats = file_path.stat()
            return {
                'filename': file_path.name,
                'file_type': file_path.suffix.lower(),
                'file_size': stats.st_size,
                'created': datetime.fromtimestamp(stats.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                'path': str(file_path.relative_to(self.config.DOCUMENTS_DIR))
            }
        except Exception as e:
            logger.warning(f"Error extracting metadata from {file_path}: {e}")
            return {}
    
    def _read_pdf(self, file_path: Path) -> str:
        """Read content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            PDF content as string
        """
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return '\n'.join(page.extract_text() for page in reader.pages)
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            raise
    
    def _read_docx(self, file_path: Path) -> str:
        """Read content from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            DOCX content as string
        """
        try:
            from docx import Document
            doc = Document(file_path)
            return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            raise 