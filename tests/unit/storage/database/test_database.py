"""
Unit tests for database functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.storage.database import Database

class TestDatabase:
    """Test suite for database functionality."""
    
    @pytest.fixture
    def database(self):
        """Create a database for testing."""
        return Database()
    
    @pytest.mark.asyncio
    async def test_database_operations(self, database):
        """Test database operations."""
        db_data = {
            "collection": "test_collection",
            "document": {
                "id": "test_doc",
                "data": "test_data"
            }
        }
        
        await database.store(db_data)
        
        retrieved_doc = await database.retrieve(
            "test_collection",
            "test_doc"
        )
        assert retrieved_doc is not None
        assert retrieved_doc["data"] == "test_data"
        
        update_data = {
            "collection": "test_collection",
            "document": {
                "id": "test_doc",
                "data": "updated_data"
            }
        }
        
        await database.update(update_data)
        updated_doc = await database.retrieve(
            "test_collection",
            "test_doc"
        )
        assert updated_doc["data"] == "updated_data"
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, database):
        """Test database error handling."""
        with pytest.raises(ValueError):
            await database.store(None)
        
        with pytest.raises(KeyError):
            await database.retrieve("non_existent_collection", "non_existent_doc")
    
    @pytest.mark.asyncio
    async def test_database_metrics(self, database):
        """Test database metrics collection."""
        db_metrics = await database.collect_metrics()
        assert db_metrics is not None
        assert isinstance(db_metrics, dict)
        assert "total_documents" in db_metrics
        assert "storage_size" in db_metrics
    
    @pytest.mark.asyncio
    async def test_database_optimization(self, database):
        """Test database optimization operations."""
        db_params = {
            "index_fields": ["id", "data"],
            "max_connections": 10,
            "pool_size": 5
        }
        
        optimized_database = await database.optimize(db_params)
        assert optimized_database is not None
        assert optimized_database.is_optimized 