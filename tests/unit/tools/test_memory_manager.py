"""
Unit tests for the memory manager component.
"""
import pytest
import os
import json
import sqlite3
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
import asyncio
from typing import Dict, Any, List
import tempfile

from tools.memory_manager import MemoryManager


class TestMemoryManagerUnit:
    """Unit tests for the MemoryManager class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_add_to_memory_short_term(
        self, 
        test_config,
        temp_db_path
    ):
        """Test adding an interaction to short-term memory."""
        # Override the memory path in config
        test_config.MEMORY_STORE_PATH = temp_db_path
        
        # Create memory manager
        memory_manager = MemoryManager(test_config)
        
        # Create test interaction
        interaction = {
            "query": "What is LangGraph?",
            "response": "LangGraph is a library for building stateful workflows with LLMs."
        }
        
        # Add to memory
        await memory_manager.add_to_memory(interaction)
        
        # Verify it was added by querying directly
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM short_term_memory")
        rows = cursor.fetchall()
        conn.close()
        
        # Assertions
        assert len(rows) == 1
        assert "LangGraph" in json.loads(rows[0][1])

    @pytest.mark.asyncio
    async def test_get_from_memory(
        self, 
        test_config,
        temp_db_path
    ):
        """Test retrieving interactions from memory."""
        # Override the memory path in config
        test_config.MEMORY_STORE_PATH = temp_db_path
        
        # Create memory manager
        memory_manager = MemoryManager(test_config)
        
        # Directly insert test data into the database
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS short_term_memory (id INTEGER PRIMARY KEY, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        cursor.execute("CREATE TABLE IF NOT EXISTS mid_term_memory (id INTEGER PRIMARY KEY, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        
        # Insert test interactions
        short_term_interaction = json.dumps({
            "query": "What is LangGraph?",
            "response": "LangGraph is a library for building stateful workflows."
        })
        mid_term_interaction = json.dumps({
            "query": "How does LangGraph compare to LangChain?",
            "response": "LangGraph is built on LangChain and provides graph-based workflows."
        })
        
        cursor.execute("INSERT INTO short_term_memory (content) VALUES (?)", (short_term_interaction,))
        cursor.execute("INSERT INTO mid_term_memory (content) VALUES (?)", (mid_term_interaction,))
        conn.commit()
        conn.close()
        
        # Retrieve from memory
        memory = await memory_manager.get_from_memory()
        
        # Assertions
        assert "short_term" in memory
        assert "mid_term" in memory
        assert len(memory["short_term"]) == 1
        assert len(memory["mid_term"]) == 1
        assert "LangGraph is a library" in memory["short_term"][0]["response"]
        assert "built on LangChain" in memory["mid_term"][0]["response"]

    @pytest.mark.asyncio
    async def test_promote_to_mid_term(
        self, 
        test_config,
        temp_db_path
    ):
        """Test promoting interactions from short-term to mid-term memory."""
        # Override the memory path in config
        test_config.MEMORY_STORE_PATH = temp_db_path
        
        # Create memory manager
        memory_manager = MemoryManager(test_config)
        
        # Directly insert test data into the database
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS short_term_memory (id INTEGER PRIMARY KEY, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        cursor.execute("CREATE TABLE IF NOT EXISTS mid_term_memory (id INTEGER PRIMARY KEY, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        
        # Insert test interaction
        interaction = json.dumps({
            "query": "What is LangGraph?",
            "response": "LangGraph is a library for building stateful workflows."
        })
        
        cursor.execute("INSERT INTO short_term_memory (content) VALUES (?)", (interaction,))
        conn.commit()
        
        # Get the ID of the inserted interaction
        cursor.execute("SELECT id FROM short_term_memory")
        interaction_id = cursor.fetchone()[0]
        conn.close()
        
        # Promote to mid-term
        await memory_manager.promote_to_mid_term(interaction_id)
        
        # Verify it was moved
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM short_term_memory WHERE id=?", (interaction_id,))
        short_term_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM mid_term_memory")
        mid_term_count = cursor.fetchone()[0]
        conn.close()
        
        # Assertions
        assert short_term_count == 0  # Should be removed from short-term
        assert mid_term_count == 1    # Should be added to mid-term

    @pytest.mark.asyncio
    async def test_clear_memory(
        self, 
        test_config,
        temp_db_path
    ):
        """Test clearing memory."""
        # Override the memory path in config
        test_config.MEMORY_STORE_PATH = temp_db_path
        
        # Create memory manager
        memory_manager = MemoryManager(test_config)
        
        # Directly insert test data into the database
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS short_term_memory (id INTEGER PRIMARY KEY, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        
        # Insert test interactions
        for i in range(3):
            interaction = json.dumps({
                "query": f"Question {i}",
                "response": f"Answer {i}"
            })
            cursor.execute("INSERT INTO short_term_memory (content) VALUES (?)", (interaction,))
        conn.commit()
        conn.close()
        
        # Clear memory
        await memory_manager.clear_memory("short_term")
        
        # Verify it was cleared
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM short_term_memory")
        count = cursor.fetchone()[0]
        conn.close()
        
        # Assertions
        assert count == 0

    @pytest.mark.asyncio
    async def test_save_conversation_history(
        self, 
        test_config,
        temp_db_path,
        sample_conversation_history
    ):
        """Test saving conversation history."""
        # Override the memory path in config
        test_config.MEMORY_STORE_PATH = temp_db_path
        
        # Create memory manager
        memory_manager = MemoryManager(test_config)
        
        # Save conversation history
        conversation_id = "test-conversation-123"
        await memory_manager.save_conversation(
            conversation_id=conversation_id,
            history=sample_conversation_history
        )
        
        # Verify it was saved
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE conversation_id=?", (conversation_id,))
        row = cursor.fetchone()
        conn.close()
        
        # Assertions
        assert row is not None
        saved_history = json.loads(row[2])  # Assuming column 2 is the history
        assert len(saved_history) == len(sample_conversation_history)
        assert saved_history[0]["content"] == "What is LangGraph?"

    @pytest.mark.asyncio
    async def test_get_conversation_history(
        self, 
        test_config,
        temp_db_path,
        sample_conversation_history
    ):
        """Test retrieving conversation history."""
        # Override the memory path in config
        test_config.MEMORY_STORE_PATH = temp_db_path
        
        # Create memory manager
        memory_manager = MemoryManager(test_config)
        
        # Directly insert test conversation
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, conversation_id TEXT, history TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        
        conversation_id = "test-conversation-123"
        cursor.execute(
            "INSERT INTO conversations (conversation_id, history) VALUES (?, ?)",
            (conversation_id, json.dumps(sample_conversation_history))
        )
        conn.commit()
        conn.close()
        
        # Retrieve conversation history
        history = await memory_manager.get_conversation_history(conversation_id)
        
        # Assertions
        assert history is not None
        assert len(history) == len(sample_conversation_history)
        assert history[0]["content"] == "What is LangGraph?"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_memory_search(
        self, 
        test_config,
        temp_db_path,
        mock_model_manager
    ):
        """Test semantic search in memory."""
        # Override the memory path in config
        test_config.MEMORY_STORE_PATH = temp_db_path
        
        # Create memory manager with mock model manager
        memory_manager = MemoryManager(
            config=test_config,
            model_manager=mock_model_manager
        )
        
        # Add test data to memory
        interactions = [
            {
                "query": "What is LangGraph?",
                "response": "LangGraph is a library for building stateful workflows."
            },
            {
                "query": "How do I create a workflow?",
                "response": "You can create a workflow by defining nodes and edges."
            },
            {
                "query": "What's the weather like today?",
                "response": "I don't have real-time weather information."
            }
        ]
        
        # Add interactions to memory
        for interaction in interactions:
            await memory_manager.add_to_memory(interaction)
        
        # Mock the embeddings function
        mock_model_manager.get_embeddings.return_value = [0.1, 0.2, 0.3, 0.4]
        
        # Search memory
        results = await memory_manager.search_memory("workflow creation")
        
        # Assertions
        assert results is not None
        assert len(results) > 0
        
        # Verify model manager was called
        mock_model_manager.get_embeddings.assert_called_once()

    @pytest.mark.asyncio
    async def test_prune_old_memories(
        self, 
        test_config,
        temp_db_path
    ):
        """Test pruning old memories."""
        # Override the memory path in config
        test_config.MEMORY_STORE_PATH = temp_db_path
        
        # Create memory manager
        memory_manager = MemoryManager(test_config)
        
        # Directly insert test data with old timestamps
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS short_term_memory (id INTEGER PRIMARY KEY, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        
        # Insert test interactions with manual timestamps
        for i in range(10):
            interaction = json.dumps({
                "query": f"Question {i}",
                "response": f"Answer {i}"
            })
            # Insert with progressively older timestamps
            cursor.execute(
                "INSERT INTO short_term_memory (content, timestamp) VALUES (?, datetime('now', '-' || ? || ' days'))",
                (interaction, i)
            )
        conn.commit()
        conn.close()
        
        # Set retention period to 5 days
        retention_days = 5
        
        # Prune old memories
        await memory_manager.prune_old_memories("short_term", retention_days)
        
        # Verify older entries were removed
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM short_term_memory")
        count = cursor.fetchone()[0]
        conn.close()
        
        # Assertions - only entries newer than 5 days should remain
        assert count <= 6  # 0 to 5 days old

    @pytest.mark.asyncio
    async def test_backup_memory(
        self, 
        test_config,
        temp_db_path,
        temp_test_dir
    ):
        """Test backing up memory to a file."""
        # Override the memory path in config
        test_config.MEMORY_STORE_PATH = temp_db_path
        
        # Create memory manager
        memory_manager = MemoryManager(test_config)
        
        # Add test data to memory
        interaction = {
            "query": "What is LangGraph?",
            "response": "LangGraph is a library for building stateful workflows."
        }
        await memory_manager.add_to_memory(interaction)
        
        # Create backup path
        backup_path = os.path.join(temp_test_dir, "memory_backup.db")
        
        # Backup memory
        await memory_manager.backup_memory(backup_path)
        
        # Verify backup was created
        assert os.path.exists(backup_path)
        
        # Verify backup contains the data
        conn = sqlite3.connect(backup_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM short_term_memory")
        rows = cursor.fetchall()
        conn.close()
        
        # Assertions
        assert len(rows) == 1
        assert "LangGraph" in json.loads(rows[0][1])