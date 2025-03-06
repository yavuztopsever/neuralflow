"""
Document processing utilities for the LangGraph project.
These utilities provide document processing capabilities integrated with LangChain.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import yaml
import csv
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PDFLoader,
    CSVLoader,
    JSONLoader,
    YAMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredURLLoader
)
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import VectorStore, FAISS

class DocumentProcessor:
    """Processor for handling various document types in the LangGraph project."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embeddings: Optional[Embeddings] = None
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embeddings: Optional embeddings model
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embeddings = embeddings
    
    def load_document(self, file_path: Union[str, Path], **kwargs: Any) -> List[Document]:
        """
        Load a document from a file.
        
        Args:
            file_path: Path to the document
            **kwargs: Additional arguments for the loader
            
        Returns:
            List[Document]: List of documents
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        # Select appropriate loader based on file type
        if suffix == '.txt':
            loader = TextLoader(str(file_path), **kwargs)
        elif suffix == '.pdf':
            loader = PDFLoader(str(file_path), **kwargs)
        elif suffix == '.csv':
            loader = CSVLoader(str(file_path), **kwargs)
        elif suffix == '.json':
            loader = JSONLoader(str(file_path), **kwargs)
        elif suffix == '.yaml' or suffix == '.yml':
            loader = YAMLLoader(str(file_path), **kwargs)
        elif suffix == '.md':
            loader = UnstructuredMarkdownLoader(str(file_path), **kwargs)
        elif suffix == '.doc' or suffix == '.docx':
            loader = UnstructuredWordDocumentLoader(str(file_path), **kwargs)
        elif suffix == '.xls' or suffix == '.xlsx':
            loader = UnstructuredExcelLoader(str(file_path), **kwargs)
        elif suffix == '.ppt' or suffix == '.pptx':
            loader = UnstructuredPowerPointLoader(str(file_path), **kwargs)
        elif suffix == '.html' or suffix == '.htm':
            loader = UnstructuredHTMLLoader(str(file_path), **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        return loader.load()
    
    def load_url(self, url: str, **kwargs: Any) -> List[Document]:
        """
        Load a document from a URL.
        
        Args:
            url: URL of the document
            **kwargs: Additional arguments for the loader
            
        Returns:
            List[Document]: List of documents
        """
        loader = UnstructuredURLLoader(url, **kwargs)
        return loader.load()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List[Document]: List of document chunks
        """
        return self.text_splitter.split_documents(documents)
    
    def create_vector_store(
        self,
        documents: List[Document],
        vector_store_type: str = "faiss"
    ) -> VectorStore:
        """
        Create a vector store from documents.
        
        Args:
            documents: List of documents
            vector_store_type: Type of vector store to create
            
        Returns:
            VectorStore: Vector store containing document embeddings
        """
        if not self.embeddings:
            raise ValueError("Embeddings model not initialized")
        
        if vector_store_type.lower() == "faiss":
            return FAISS.from_documents(documents, self.embeddings)
        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")
    
    def add_metadata(
        self,
        documents: List[Document],
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Add metadata to documents.
        
        Args:
            documents: List of documents
            metadata: Metadata to add
            
        Returns:
            List[Document]: Documents with added metadata
        """
        for doc in documents:
            doc.metadata.update(metadata)
        return documents
    
    def filter_documents(
        self,
        documents: List[Document],
        filter_func: callable
    ) -> List[Document]:
        """
        Filter documents based on a function.
        
        Args:
            documents: List of documents
            filter_func: Function to filter documents
            
        Returns:
            List[Document]: Filtered documents
        """
        return [doc for doc in documents if filter_func(doc)]
    
    def merge_documents(
        self,
        documents: List[Document],
        merge_func: Optional[callable] = None
    ) -> Document:
        """
        Merge multiple documents into one.
        
        Args:
            documents: List of documents
            merge_func: Optional function to merge document content
            
        Returns:
            Document: Merged document
        """
        if not documents:
            raise ValueError("No documents to merge")
        
        if merge_func:
            content = merge_func([doc.page_content for doc in documents])
        else:
            content = "\n\n".join(doc.page_content for doc in documents)
        
        metadata = {}
        for doc in documents:
            metadata.update(doc.metadata)
        
        return Document(page_content=content, metadata=metadata)
    
    def save_documents(
        self,
        documents: List[Document],
        output_dir: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Save documents to files.
        
        Args:
            documents: List of documents
            output_dir: Output directory
            format: Output format
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, doc in enumerate(documents):
            output_path = output_dir / f"document_{i}.{format}"
            
            if format == "json":
                with open(output_path, "w") as f:
                    json.dump({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }, f, indent=2)
            elif format == "yaml":
                with open(output_path, "w") as f:
                    yaml.dump({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def load_documents_from_dir(
        self,
        input_dir: Union[str, Path],
        format: str = "json"
    ) -> List[Document]:
        """
        Load documents from a directory.
        
        Args:
            input_dir: Input directory
            format: Input format
            
        Returns:
            List[Document]: List of documents
        """
        input_dir = Path(input_dir)
        documents = []
        
        for file_path in input_dir.glob(f"*.{format}"):
            if format == "json":
                with open(file_path) as f:
                    data = json.load(f)
                    documents.append(Document(
                        page_content=data["content"],
                        metadata=data["metadata"]
                    ))
            elif format == "yaml":
                with open(file_path) as f:
                    data = yaml.safe_load(f)
                    documents.append(Document(
                        page_content=data["content"],
                        metadata=data["metadata"]
                    ))
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        return documents

__all__ = ['DocumentProcessor'] 