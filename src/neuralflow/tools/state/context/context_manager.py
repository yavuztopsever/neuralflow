from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from ..utils.vector_store import VectorStore
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ContextItem:
    content: str
    relevance_score: float
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]

class ContextManager:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.context_window = 5  # Number of most relevant contexts to keep
        self.min_relevance_threshold = 0.7
        
    def retrieve_context(self, query: str, max_items: int = 5) -> List[ContextItem]:
        """Retrieve relevant context based on semantic similarity."""
        try:
            # Get semantically similar items from vector store
            similar_items = self.vector_store.similarity_search(
                query,
                k=max_items
            )
            
            # Convert to ContextItems and filter by relevance
            contexts = []
            for item in similar_items:
                if item.relevance_score >= self.min_relevance_threshold:
                    contexts.append(ContextItem(
                        content=item.content,
                        relevance_score=item.relevance_score,
                        timestamp=item.timestamp,
                        source=item.source,
                        metadata=item.metadata
                    ))
            
            # Sort by relevance score
            contexts.sort(key=lambda x: x.relevance_score, reverse=True)
            return contexts[:self.context_window]
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def prioritize_context(self, contexts: List[ContextItem], 
                         current_task: str) -> List[ContextItem]:
        """Prioritize contexts based on task relevance and recency."""
        if not contexts:
            return []
            
        # Calculate task-specific relevance scores
        task_embeddings = self.vector_store.get_embeddings(current_task)
        for context in contexts:
            context_embeddings = self.vector_store.get_embeddings(context.content)
            # Combine semantic similarity with recency
            recency_factor = self._calculate_recency_factor(context.timestamp)
            context.relevance_score = (
                0.7 * np.dot(task_embeddings, context_embeddings) +
                0.3 * recency_factor
            )
        
        # Sort by updated relevance scores
        contexts.sort(key=lambda x: x.relevance_score, reverse=True)
        return contexts
    
    def _calculate_recency_factor(self, timestamp: datetime) -> float:
        """Calculate a recency factor based on timestamp."""
        now = datetime.now()
        age_hours = (now - timestamp).total_seconds() / 3600
        return np.exp(-age_hours / 24)  # Exponential decay over 24 hours
    
    def update_context(self, new_context: str, source: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add new context to the vector store."""
        try:
            self.vector_store.add_texts(
                texts=[new_context],
                metadatas=[metadata or {}],
                sources=[source]
            )
        except Exception as e:
            logger.error(f"Error updating context: {str(e)}")
    
    def clear_old_contexts(self, max_age_hours: int = 24) -> None:
        """Remove contexts older than specified age."""
        try:
            self.vector_store.remove_old_entries(max_age_hours)
        except Exception as e:
            logger.error(f"Error clearing old contexts: {str(e)}") 