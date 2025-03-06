import os
import sys
import json
import asyncio
import zlib
from collections import deque
from langgraph.graph import StateGraph
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func, Index, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from redis import Redis, ConnectionPool, ConnectionError
from abc import ABC, abstractmethod
from typing import TypedDict, Dict, Any, List, Optional, Union
from transformers import pipeline
import time
from config.config import Config
from sqlalchemy.exc import SQLAlchemyError
from redis.exceptions import RedisError
import backoff
import threading
from ratelimit import limits, sleep_and_retry
import logging
import gc
import psutil
from datetime import datetime, timedelta
import shutil
from tools.vector_search import VectorSearch
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker as async_sessionmaker

# Initialize logger
logger = logging.getLogger(__name__)

# Try to import aiofiles - add proper error handling
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False
    logging.warning("aiofiles not installed. Using synchronous file I/O.")

# Define a schema for interactions using TypedDict
class InteractionSchema(TypedDict):
    type: str
    user_query: str
    response: str
    document_path: str
    search_results: str  # Add search_results field

Base = declarative_base()

class Interaction(Base):
    __tablename__ = 'interactions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String)
    user_query = Column(String)
    response = Column(String)
    document_path = Column(String)
    timestamp = Column(DateTime, default=func.now())
    __table_args__ = (
        Index('idx_type', 'type'),
        Index('idx_user_query', 'user_query'),
    )

class ShortTermMemory(Base):
    """SQLAlchemy model for short-term memory."""
    __tablename__ = "short_term_memory"
    
    id = Column(Integer, primary_key=True)
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    extra_data = Column(JSON)

class MidTermMemory(Base):
    """SQLAlchemy model for mid-term memory."""
    __tablename__ = "mid_term_memory"
    
    id = Column(Integer, primary_key=True)
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    extra_data = Column(JSON)

class LongTermMemory(Base):
    """SQLAlchemy model for long-term memory."""
    __tablename__ = "long_term_memory"
    
    id = Column(Integer, primary_key=True)
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    extra_data = Column(JSON)

class Conversation(Base):
    """SQLAlchemy model for conversation history."""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True)
    messages = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    extra_data = Column(JSON)

class MemoryBackend(ABC):
    @abstractmethod
    def save_interaction(self, interaction):
        pass

    @abstractmethod
    def get_interactions(self, query=None, interaction_type=None, limit=10):
        pass

    @abstractmethod
    def get_interactions_by_time_range(self, start_time, end_time, limit=10):
        pass

    @abstractmethod
    def get_interactions_by_timestamp(self, timestamp, limit=10):
        pass

class SQLAlchemyBackend(MemoryBackend):
    def __init__(self, db_url):
        # Configure engine with memory-efficient settings
        engine_args = {
            'pool_pre_ping': True,
            'pool_recycle': 3600,  # Connection recycle time in seconds
            'poolclass': QueuePool,
            'pool_size': 5,  # Smaller pool size (default is usually 5)
            'max_overflow': 5,  # Smaller overflow (default is usually 10)
            'echo': False,  # Disable SQL echoing
            'echo_pool': False  # Disable connection pool logging
        }
        
        self.engine = create_engine(db_url, **engine_args)
        Base.metadata.create_all(self.engine)
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        
        # Add periodic cleanup
        self._last_cleanup_time = time.time()
        self._cleanup_interval = 300  # 5 minutes
        
    def _maybe_cleanup(self):
        """Perform periodic cleanup to prevent memory leaks."""
        current_time = time.time()
        if current_time - self._last_cleanup_time > self._cleanup_interval:
            self._last_cleanup_time = current_time
            # Dispose connections in the pool
            self.engine.dispose()
            # Clear session registry
            self.Session.remove()
            print("SQLAlchemyBackend: Connection pool cleaned up")

    @backoff.on_exception(backoff.expo, SQLAlchemyError, max_tries=3)
    def save_interaction(self, interaction):
        # Periodic cleanup check
        self._maybe_cleanup()
        
        session = self.Session()
        try:
            # Only keep essential fields to reduce memory usage
            essential_fields = {k: v for k, v in interaction.items() 
                               if k in ['type', 'user_query', 'response', 'document_path', 'timestamp']}
            
            # Truncate large strings to prevent excessive memory usage
            for field in ['user_query', 'response']:
                if field in essential_fields and essential_fields[field] and isinstance(essential_fields[field], str):
                    if len(essential_fields[field]) > 10000:  # Limit to 10KB
                        essential_fields[field] = essential_fields[field][:10000] + "... [truncated]"
            
            new_interaction = Interaction(**essential_fields)
            session.add(new_interaction)
            session.commit()
        except SQLAlchemyError as e:
            print(f"Error saving interaction: {e}")
            session.rollback()
            # Don't raise, just log the error to prevent pipeline interruption
        except Exception as e:
            print(f"Unexpected error saving interaction: {e}")
            session.rollback()
        finally:
            session.close()

    @backoff.on_exception(backoff.expo, SQLAlchemyError, max_tries=3)
    def get_interactions(self, query=None, interaction_type=None, limit=10):
        # Use a smaller limit to reduce memory impact
        if limit > 20:
            limit = 20
            
        # Periodic cleanup check
        self._maybe_cleanup()
        
        session = self.Session()
        try:
            query_obj = session.query(Interaction)
            if query:
                # Use a more efficient LIKE query
                query_obj = query_obj.filter(Interaction.user_query.like(f"%{query}%"))
            if interaction_type:
                query_obj = query_obj.filter(Interaction.type == interaction_type)
                
            # Get only needed columns and limit results
            results = query_obj.order_by(Interaction.timestamp.desc()).limit(limit).all()
            
            # Convert to dict outside session
            return [self._interaction_to_dict(interaction) for interaction in results]
        except SQLAlchemyError as e:
            print(f"Error retrieving interactions: {e}")
            session.rollback()
            # Return empty list instead of raising
            return []
        except Exception as e:
            print(f"Unexpected error retrieving interactions: {e}")
            return []
        finally:
            session.close()

    def get_interactions_by_time_range(self, start_time, end_time, limit=10):
        return self._get_interactions_by_filter(
            Interaction.timestamp.between(start_time, end_time), limit
        )

    def get_interactions_by_timestamp(self, timestamp, limit=10):
        return self._get_interactions_by_filter(
            Interaction.timestamp == timestamp, limit
        )

    def _get_interactions_by_filter(self, filter_condition, limit):
        session = self.Session()
        try:
            query_obj = session.query(Interaction).filter(filter_condition).order_by(Interaction.timestamp.desc()).limit(limit)
            results = query_obj.all()
        except Exception as e:
            print(f"Error retrieving interactions: {e}")
            return []
        finally:
            session.close()
        return [self._interaction_to_dict(interaction) for interaction in results]

    def _interaction_to_dict(self, interaction):
        return {
            "id": interaction.id,
            "type": interaction.type,
            "user_query": interaction.user_query,
            "response": interaction.response,
            "document_path": interaction.document_path,
            "timestamp": interaction.timestamp
        }

def initialize_database(db_url=None):
    """Initialize the SQLAlchemy database with proper fallback.
    
    Args:
        db_url: Optional database URL. If not provided, will use Config.SQLALCHEMY_DB_URL.
        
    Returns:
        SQLAlchemyBackend instance
    """
    if db_url is None:
        # Use the default URL from Config
        db_url = getattr(Config, 'SQLALCHEMY_DB_URL', 'sqlite:///memory.db')
        
    # Handle Path objects if necessary
    if hasattr(db_url, 'as_posix'):
        db_url = str(db_url)
        
    # Initialize the SQLAlchemy backend
    try:
        sqlalchemy_backend = SQLAlchemyBackend(db_url=db_url)
        return sqlalchemy_backend
    except Exception as e:
        print(f"Error initializing database: {e}. Using in-memory SQLite.")
        # Fallback to in-memory SQLite
        return SQLAlchemyBackend(db_url='sqlite:///:memory:')

def initialize_memory_manager(state_graph, memory_backend, redis_url, short_term_limit, mid_term_limit, session_expiration_threshold):
    # Create a MemoryManager instance with the provided dependencies
    memory_manager = MemoryManager(
        state_graph=state_graph,
        memory_backend=memory_backend,
        redis_url=redis_url,
        short_term_limit=short_term_limit,
        mid_term_limit=mid_term_limit,
        session_expiration_threshold=session_expiration_threshold,
        redis_pool=ConnectionPool.from_url(redis_url)
    )
    return memory_manager

# Define rate limit constants
CALLS = 10
PERIOD = 60

class MemoryManager:
    """Manages different types of memory storage and retrieval."""
    
    def __init__(self, config: Any):
        """Initialize the memory manager with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize required components."""
        try:
            # Create database engine
            self.engine = create_async_engine(
                self.config.memory_db_url,
                echo=self.config.debug
            )
            
            # Create session factory
            self.async_session = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Initialize vector search
            self.vector_search = VectorSearch(config=self.config)
        except Exception as e:
            self.logger.error(f"Error initializing memory manager components: {str(e)}")
            raise
            
    async def add_to_memory(
        self,
        content: str,
        memory_type: str = "short_term"
    ) -> bool:
        """Add content to the specified memory type."""
        try:
            async with self.async_session() as session:
                if memory_type == "short_term":
                    memory = ShortTermMemory(content=content)
                elif memory_type == "mid_term":
                    memory = MidTermMemory(content=content)
                elif memory_type == "long_term":
                    memory = LongTermMemory(content=content)
                else:
                    raise ValueError(f"Invalid memory type: {memory_type}")
                    
                session.add(memory)
                await session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error adding to memory: {str(e)}")
            return False
            
    async def get_from_memory(
        self,
        memory_type: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve content from the specified memory type."""
        try:
            async with self.async_session() as session:
                if memory_type == "short_term":
                    query = session.query(ShortTermMemory)
                elif memory_type == "mid_term":
                    query = session.query(MidTermMemory)
                elif memory_type == "long_term":
                    query = session.query(LongTermMemory)
                else:
                    raise ValueError(f"Invalid memory type: {memory_type}")
                    
                memories = await query.order_by(
                    getattr(query.model, "timestamp").desc()
                ).limit(limit).all()
                
                return [
                    {
                        "content": memory.content,
                        "timestamp": memory.timestamp,
                        "metadata": getattr(memory, "extra_data", None)
                    }
                    for memory in memories
                ]
        except Exception as e:
            self.logger.error(f"Error getting from memory: {str(e)}")
            return []
            
    async def promote_to_mid_term(
        self,
        memory_id: int
    ) -> bool:
        """Promote a short-term memory to mid-term memory."""
        try:
            async with self.async_session() as session:
                # Get short-term memory
                stm = await session.query(ShortTermMemory).filter(
                    ShortTermMemory.id == memory_id
                ).first()
                
                if not stm:
                    return False
                    
                # Create mid-term memory
                mtm = MidTermMemory(
                    content=stm.content,
                    extra_data=stm.extra_data
                )
                
                # Delete short-term memory
                await session.delete(stm)
                session.add(mtm)
                await session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error promoting to mid-term memory: {str(e)}")
            return False
            
    async def clear_memory(
        self,
        memory_type: Optional[str] = None
    ) -> bool:
        """Clear memory of the specified type or all memory if no type specified."""
        try:
            async with self.async_session() as session:
                if memory_type == "short_term":
                    await session.query(ShortTermMemory).delete()
                elif memory_type == "mid_term":
                    await session.query(MidTermMemory).delete()
                elif memory_type == "long_term":
                    await session.query(LongTermMemory).delete()
                else:
                    await session.query(ShortTermMemory).delete()
                    await session.query(MidTermMemory).delete()
                    await session.query(LongTermMemory).delete()
                    
                await session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error clearing memory: {str(e)}")
            return False
            
    async def save_conversation_history(
        self,
        messages: List[Dict[str, str]]
    ) -> bool:
        """Save conversation history to memory."""
        try:
            async with self.async_session() as session:
                conversation = Conversation(messages=messages)
                session.add(conversation)
                await session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error saving conversation history: {str(e)}")
            return False
            
    async def get_conversation_history(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve conversation history from memory."""
        try:
            async with self.async_session() as session:
                conversations = await session.query(Conversation).order_by(
                    Conversation.timestamp.desc()
                ).limit(limit).all()
                
                return [
                    {
                        "messages": conversation.messages,
                        "timestamp": conversation.timestamp,
                        "metadata": conversation.extra_data
                    }
                    for conversation in conversations
                ]
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {str(e)}")
            return []
            
    async def prune_old_memories(
        self,
        age_days: Optional[int] = None
    ) -> bool:
        """Prune memories older than the specified age."""
        try:
            age = age_days or self.config.memory_prune_age_days
            cutoff_date = datetime.utcnow() - timedelta(days=age)
            
            async with self.async_session() as session:
                # Prune short-term memory
                await session.query(ShortTermMemory).filter(
                    ShortTermMemory.timestamp < cutoff_date
                ).delete()
                
                # Prune mid-term memory
                await session.query(MidTermMemory).filter(
                    MidTermMemory.timestamp < cutoff_date
                ).delete()
                
                # Prune long-term memory
                await session.query(LongTermMemory).filter(
                    LongTermMemory.timestamp < cutoff_date
                ).delete()
                
                await session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error pruning old memories: {str(e)}")
            return False
            
    async def backup_memory(self) -> bool:
        """Back up the memory database."""
        try:
            # Create backup directory if it doesn't exist
            backup_dir = self.config.memory_backup_dir
            if not backup_dir.exists():
                backup_dir.mkdir(parents=True)
                
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"memory_backup_{timestamp}.db"
            
            # Copy database file
            shutil.copy2(self.config.memory_db_url.replace("sqlite+aiosqlite:///", ""), backup_path)
            return True
        except Exception as e:
            self.logger.error(f"Error backing up memory: {str(e)}")
            return False

def create_memory_manager(config=None):
    """Factory function to create a MemoryManager instance with the provided configuration.
    
    Args:
        config: Optional config object. If not provided, the global Config will be used.
        
    Returns:
        A new MemoryManager instance
    """
    # Use provided config or default to global Config
    if config is None:
        config = Config
        
    # Initialize the SQLAlchemy database
    sqlalchemy_backend = initialize_database(db_url=config.SQLALCHEMY_DB_URL)
    
    # Create a LangGraph state graph
    graph = StateGraph(state_schema=InteractionSchema)
    graph.state = {}  # Initialize the state attribute
    
    # Create a MemoryManager instance with SQLAlchemy backend
    memory_manager = initialize_memory_manager(
        state_graph=graph,
        memory_backend=sqlalchemy_backend,
        redis_url=config.REDIS_URL,
        short_term_limit=config.SHORT_TERM_LIMIT,
        mid_term_limit=config.MID_TERM_LIMIT,
        session_expiration_threshold=config.SESSION_EXPIRATION_THRESHOLD
    )
    
    # Add the MemoryManager as a node to the graph
    if hasattr(graph, 'add_node'):
        graph.add_node("memory_manager", memory_manager.save_interaction)
    
    return memory_manager

class StateGraph:
    def __init__(self, state_schema):
        self.state_schema = state_schema
        self.state = None  # Initialize the state attribute
        # ...existing code...
