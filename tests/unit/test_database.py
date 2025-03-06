"""
Unit tests for database system functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.storage.database import Database
from src.storage.database.connection import DatabaseConnection
from src.storage.database.query import QueryBuilder

class TestDatabase:
    """Test suite for database system functionality."""
    
    @pytest.fixture
    def database(self):
        """Create a database for testing."""
        return Database()
    
    @pytest.fixture
    def connection(self):
        """Create a database connection for testing."""
        return DatabaseConnection()
    
    @pytest.fixture
    def query_builder(self):
        """Create a query builder for testing."""
        return QueryBuilder()
    
    @pytest.mark.asyncio
    async def test_database_operations(self, database):
        """Test database operations."""
        # Test storing data
        db_data = {
            "collection": "test_collection",
            "document": {
                "id": "test_doc",
                "data": "test_data"
            }
        }
        
        await database.store(db_data)
        
        # Test retrieving data
        retrieved_doc = await database.retrieve(
            "test_collection",
            "test_doc"
        )
        assert retrieved_doc is not None
        assert retrieved_doc["data"] == "test_data"
        
        # Test updating data
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
        
        # Test deleting data
        await database.delete("test_collection", "test_doc")
        with pytest.raises(KeyError):
            await database.retrieve("test_collection", "test_doc")
    
    @pytest.mark.asyncio
    async def test_database_connection_operations(self, connection):
        """Test database connection operations."""
        # Test connection establishment
        await connection.connect()
        assert connection.is_connected
        
        # Test connection pooling
        pool = await connection.get_pool()
        assert pool is not None
        assert pool.size > 0
        
        # Test connection timeout
        await connection.set_timeout(5)
        assert connection.timeout == 5
        
        # Test connection cleanup
        await connection.cleanup()
        assert not connection.is_connected
    
    @pytest.mark.asyncio
    async def test_query_builder_operations(self, query_builder):
        """Test query builder operations."""
        # Test building select query
        select_query = query_builder.select("test_collection").where(
            {"field": "value"}
        ).build()
        assert select_query is not None
        assert isinstance(select_query, str)
        assert "SELECT" in select_query
        
        # Test building insert query
        insert_query = query_builder.insert("test_collection").values(
            {"field": "value"}
        ).build()
        assert insert_query is not None
        assert isinstance(insert_query, str)
        assert "INSERT" in insert_query
        
        # Test building update query
        update_query = query_builder.update("test_collection").set(
            {"field": "new_value"}
        ).where({"id": "test_doc"}).build()
        assert update_query is not None
        assert isinstance(update_query, str)
        assert "UPDATE" in update_query
        
        # Test building delete query
        delete_query = query_builder.delete("test_collection").where(
            {"id": "test_doc"}
        ).build()
        assert delete_query is not None
        assert isinstance(delete_query, str)
        assert "DELETE" in delete_query
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, database, connection, query_builder):
        """Test database error handling."""
        # Test database errors
        with pytest.raises(ValueError):
            await database.store(None)
        
        # Test connection errors
        with pytest.raises(ValueError):
            await connection.connect(None)
        
        # Test query builder errors
        with pytest.raises(ValueError):
            query_builder.select(None)
    
    @pytest.mark.asyncio
    async def test_database_metrics(self, database, connection, query_builder):
        """Test database metrics collection."""
        # Test database metrics
        db_metrics = await database.collect_metrics()
        assert db_metrics is not None
        assert isinstance(db_metrics, dict)
        assert "total_documents" in db_metrics
        assert "storage_size" in db_metrics
        
        # Test connection metrics
        conn_metrics = await connection.collect_metrics()
        assert conn_metrics is not None
        assert isinstance(conn_metrics, dict)
        assert "active_connections" in conn_metrics
        assert "pool_size" in conn_metrics
        
        # Test query metrics
        query_metrics = await query_builder.collect_metrics()
        assert query_metrics is not None
        assert isinstance(query_metrics, dict)
        assert "total_queries" in query_metrics
        assert "average_build_time" in query_metrics
    
    @pytest.mark.asyncio
    async def test_database_optimization(self, database, connection, query_builder):
        """Test database optimization operations."""
        # Test database optimization
        db_params = {
            "index_fields": ["id", "data"],
            "max_connections": 10,
            "pool_size": 5
        }
        
        optimized_db = await database.optimize(db_params)
        assert optimized_db is not None
        assert optimized_db.is_optimized
        
        # Test connection optimization
        conn_params = {
            "max_connections": 20,
            "timeout": 10,
            "retry_count": 3
        }
        
        optimized_conn = await connection.optimize(conn_params)
        assert optimized_conn is not None
        assert optimized_conn.is_optimized
        
        # Test query builder optimization
        query_params = {
            "cache_size": 1000,
            "max_query_length": 10000,
            "timeout": 5
        }
        
        optimized_query = await query_builder.optimize(query_params)
        assert optimized_query is not None
        assert optimized_query.is_optimized
    
    @pytest.mark.asyncio
    async def test_database_transactions(self, database):
        """Test database transaction operations."""
        # Test transaction start
        transaction = await database.start_transaction()
        assert transaction is not None
        assert transaction.is_active
        
        # Test transaction commit
        await transaction.commit()
        assert not transaction.is_active
        
        # Test transaction rollback
        transaction = await database.start_transaction()
        await transaction.rollback()
        assert not transaction.is_active
        
        # Test transaction timeout
        transaction = await database.start_transaction()
        await transaction.set_timeout(5)
        assert transaction.timeout == 5 