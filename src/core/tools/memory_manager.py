from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
import json
import os
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import BaseMessage
from langchain.vectorstores import VectorStore
from langchain.embeddings import OpenAIEmbeddings

@dataclass
class MemoryItem:
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None

class MemoryManager:
    def __init__(self, max_items: int = 1000, vector_store: Optional[VectorStore] = None):
        self.max_items = max_items
        self.short_term = []
        self.mid_term = []
        self.long_term = []
        self.memory_types = ["short_term", "mid_term", "long_term"]
        self.vector_store = vector_store or VectorStore()
        self.embeddings = OpenAIEmbeddings()
        
        # LangChain memory components
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.summary_memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="summary"
        )

    def add_to_memory(self, content: str, memory_type: str = "short_term", metadata: Optional[Dict[str, Any]] = None):
        """Add an item to memory with vector embedding."""
        if memory_type not in self.memory_types:
            raise ValueError(f"Invalid memory type. Must be one of {self.memory_types}")

        # Generate embedding for the content
        embedding = self.embeddings.embed_query(content)
        
        item = MemoryItem(
            content=content,
            timestamp=time.time(),
            metadata=metadata,
            embedding=embedding
        )

        memory_list = getattr(self, memory_type)
        memory_list.append(item)

        # Add to vector store
        self.vector_store.add_texts(
            texts=[content],
            metadatas=[metadata or {}],
            embeddings=[embedding]
        )

        # Maintain size limits
        if len(memory_list) > self.max_items:
            memory_list.pop(0)

    def get_from_memory(self, memory_type: str = "short_term", limit: int = 10, query: Optional[str] = None) -> List[MemoryItem]:
        """Get items from memory with optional semantic search."""
        if memory_type not in self.memory_types:
            raise ValueError(f"Invalid memory type. Must be one of {self.memory_types}")

        if query:
            # Perform semantic search
            results = self.vector_store.similarity_search_with_score(query, k=limit)
            return [
                MemoryItem(
                    content=doc.page_content,
                    timestamp=time.time(),
                    metadata=doc.metadata
                )
                for doc, score in results
            ]
        
        memory_list = getattr(self, memory_type)
        return memory_list[-limit:]

    def update_conversation_memory(self, messages: List[BaseMessage]):
        """Update LangChain conversation memory."""
        for message in messages:
            self.conversation_memory.save_context(
                {"input": message.content},
                {"output": ""}
            )
            self.summary_memory.save_context(
                {"input": message.content},
                {"output": ""}
            )

    def get_conversation_summary(self) -> str:
        """Get conversation summary from LangChain memory."""
        return self.summary_memory.load_memory_variables({})["summary"]

    def clear_memory(self, memory_type: Optional[str] = None):
        """Clear memory of specified type or all memory if type is None."""
        if memory_type is None:
            for mt in self.memory_types:
                setattr(self, mt, [])
            self.vector_store.delete_all()
            self.conversation_memory.clear()
            self.summary_memory.clear()
        elif memory_type in self.memory_types:
            setattr(self, memory_type, [])
        else:
            raise ValueError(f"Invalid memory type. Must be one of {self.memory_types}")

    def save_conversation(self, conversation: List[Dict[str, str]], filepath: str):
        """Save conversation to file with embeddings."""
        conversation_data = []
        for msg in conversation:
            embedding = self.embeddings.embed_query(msg["content"])
            conversation_data.append({
                "content": msg["content"],
                "role": msg["role"],
                "timestamp": time.time(),
                "embedding": embedding
            })
        
        with open(filepath, 'w') as f:
            json.dump(conversation_data, f)

    def get_conversation_history(self, filepath: str) -> List[Dict[str, str]]:
        """Load conversation from file with embeddings."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Reconstruct vector store
                for item in data:
                    self.vector_store.add_texts(
                        texts=[item["content"]],
                        metadatas=[{"role": item["role"]}],
                        embeddings=[item["embedding"]]
                    )
                return [{"content": item["content"], "role": item["role"]} for item in data]
        return []

    def consolidate_memory(self):
        """Move items from short-term to mid-term memory with vector store update."""
        if self.short_term:
            self.mid_term.extend(self.short_term)
            self.short_term.clear()
            if len(self.mid_term) > self.max_items:
                self.mid_term = self.mid_term[-self.max_items:]

    def archive_memory(self):
        """Move items from mid-term to long-term memory with vector store update."""
        if self.mid_term:
            self.long_term.extend(self.mid_term)
            self.mid_term.clear()
            if len(self.long_term) > self.max_items:
                self.long_term = self.long_term[-self.max_items:] 