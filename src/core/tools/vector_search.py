from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import time

@dataclass
class SearchResult:
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class VectorSearch:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_texts(
            texts=[""],
            embedding=self.embeddings,
            metadatas=[{}]
        )
        
        # Initialize retrievers
        self.time_weighted_retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=self.vector_store,
            decay_rate=0.01,
            k=5
        )
        
        # Initialize self-query retriever
        metadata_field_info = [
            AttributeInfo(
                name="timestamp",
                description="Timestamp of the document",
                type="float",
            ),
            AttributeInfo(
                name="source",
                description="Source of the document",
                type="string",
            ),
            AttributeInfo(
                name="category",
                description="Category of the document",
                type="string",
            ),
        ]
        
        self.self_query_retriever = SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vector_store,
            document_contents="Document content",
            metadata_field_info=metadata_field_info,
        )

    def add_document(self, text: str, vector: Optional[List[float]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Add a document to the vector store with optional pre-computed embedding."""
        if vector is None:
            vector = self.embeddings.embed_query(text)
        elif len(vector) != self.dimension:
            raise ValueError(f"Vector dimension must be {self.dimension}")
            
        metadata = metadata or {}
        metadata["timestamp"] = time.time()
        
        self.vector_store.add_texts(
            texts=[text],
            metadatas=[metadata],
            embeddings=[vector]
        )

    def search(self, query: str, k: int = 5, search_type: str = "similarity") -> List[SearchResult]:
        """Search for similar documents using different search strategies."""
        if not self.vector_store:
            return []

        if search_type == "similarity":
            results = self.vector_store.similarity_search_with_score(query, k=k)
        elif search_type == "time_weighted":
            results = self.time_weighted_retriever.get_relevant_documents(query)
        elif search_type == "self_query":
            results = self.self_query_retriever.get_relevant_documents(query)
        else:
            raise ValueError(f"Invalid search type: {search_type}")

        return [
            SearchResult(
                text=doc.page_content,
                score=score if isinstance(score, float) else 1.0,
                metadata=doc.metadata,
                timestamp=doc.metadata.get("timestamp")
            )
            for doc, score in results
        ]

    def hybrid_search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Perform hybrid search combining different search strategies."""
        results = []
        
        # Get results from different retrievers
        similarity_results = self.search(query, k=k, search_type="similarity")
        time_weighted_results = self.search(query, k=k, search_type="time_weighted")
        self_query_results = self.search(query, k=k, search_type="self_query")
        
        # Combine and deduplicate results
        seen_texts = set()
        for result in similarity_results + time_weighted_results + self_query_results:
            if result.text not in seen_texts:
                seen_texts.add(result.text)
                results.append(result)
        
        # Sort by score and take top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def delete_document(self, index: int):
        """Delete a document by index."""
        if 0 <= index < len(self.vector_store.docstore.docs):
            self.vector_store.delete([index])

    def update_document(self, index: int, text: str, vector: Optional[List[float]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Update a document by index."""
        if 0 <= index < len(self.vector_store.docstore.docs):
            if vector is None:
                vector = self.embeddings.embed_query(text)
            elif len(vector) != self.dimension:
                raise ValueError(f"Vector dimension must be {self.dimension}")
                
            metadata = metadata or {}
            metadata["timestamp"] = time.time()
            
            self.vector_store.update_document(
                index,
                Document(
                    page_content=text,
                    metadata=metadata
                ),
                embedding=vector
            )

    def clear(self):
        """Clear all documents."""
        self.vector_store = FAISS.from_texts(
            texts=[""],
            embedding=self.embeddings,
            metadatas=[{}]
        )

    def count(self) -> int:
        """Get the number of documents."""
        return len(self.vector_store.docstore.docs)

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Get a document by its ID."""
        return self.vector_store.docstore.search(doc_id)

    def get_similar_documents(self, doc_id: str, k: int = 5) -> List[Document]:
        """Get similar documents to a given document ID."""
        doc = self.get_document_by_id(doc_id)
        if doc:
            return self.vector_store.similarity_search(doc.page_content, k=k)
        return [] 