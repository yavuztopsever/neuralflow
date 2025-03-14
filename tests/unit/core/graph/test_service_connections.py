"""
Unit tests for the service connections components.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from pydantic import BaseModel

from src.core.graph.service_connections import (
    ServiceConnection,
    HTTPConnection,
    WebSocketConnection,
    DatabaseConnection,
    MessageQueueConnection
)

@pytest.fixture
def mock_http_config():
    """Create mock HTTP connection configuration."""
    return {
        "url": "http://test.com",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "timeout": 30
    }

@pytest.fixture
def mock_websocket_config():
    """Create mock WebSocket connection configuration."""
    return {
        "url": "ws://test.com",
        "protocols": ["v1"],
        "ping_interval": 20
    }

@pytest.fixture
def mock_database_config():
    """Create mock database connection configuration."""
    return {
        "type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "user": "test_user",
        "password": "test_password"
    }

@pytest.fixture
def mock_message_queue_config():
    """Create mock message queue connection configuration."""
    return {
        "type": "rabbitmq",
        "host": "localhost",
        "port": 5672,
        "queue": "test_queue",
        "exchange": "test_exchange"
    }

def test_service_connection_initialization():
    """Test initialization of base ServiceConnection."""
    connection = ServiceConnection(connection_id="test_connection")
    assert connection.connection_id == "test_connection"
    assert connection.is_connected is False
    assert connection.last_error is None

def test_http_connection_initialization(mock_http_config):
    """Test initialization of HTTPConnection."""
    connection = HTTPConnection(
        connection_id="http_connection",
        config=mock_http_config
    )
    assert connection.connection_id == "http_connection"
    assert connection.url == mock_http_config["url"]
    assert connection.method == mock_http_config["method"]
    assert connection.headers == mock_http_config["headers"]
    assert connection.timeout == mock_http_config["timeout"]

def test_websocket_connection_initialization(mock_websocket_config):
    """Test initialization of WebSocketConnection."""
    connection = WebSocketConnection(
        connection_id="ws_connection",
        config=mock_websocket_config
    )
    assert connection.connection_id == "ws_connection"
    assert connection.url == mock_websocket_config["url"]
    assert connection.protocols == mock_websocket_config["protocols"]
    assert connection.ping_interval == mock_websocket_config["ping_interval"]

def test_database_connection_initialization(mock_database_config):
    """Test initialization of DatabaseConnection."""
    connection = DatabaseConnection(
        connection_id="db_connection",
        config=mock_database_config
    )
    assert connection.connection_id == "db_connection"
    assert connection.db_type == mock_database_config["type"]
    assert connection.host == mock_database_config["host"]
    assert connection.port == mock_database_config["port"]
    assert connection.database == mock_database_config["database"]
    assert connection.user == mock_database_config["user"]
    assert connection.password == mock_database_config["password"]

def test_message_queue_connection_initialization(mock_message_queue_config):
    """Test initialization of MessageQueueConnection."""
    connection = MessageQueueConnection(
        connection_id="mq_connection",
        config=mock_message_queue_config
    )
    assert connection.connection_id == "mq_connection"
    assert connection.queue_type == mock_message_queue_config["type"]
    assert connection.host == mock_message_queue_config["host"]
    assert connection.port == mock_message_queue_config["port"]
    assert connection.queue == mock_message_queue_config["queue"]
    assert connection.exchange == mock_message_queue_config["exchange"]

@pytest.mark.asyncio
async def test_http_connection_operations(mock_http_config):
    """Test HTTP connection operations."""
    connection = HTTPConnection(
        connection_id="http_connection",
        config=mock_http_config
    )
    
    # Mock requests
    with patch('aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.json = AsyncMock(return_value={"result": "success"})
        
        # Test connect
        await connection.connect()
        assert connection.is_connected is True
        
        # Test send
        response = await connection.send({"data": "test"})
        assert response == {"result": "success"}
        
        # Test disconnect
        await connection.disconnect()
        assert connection.is_connected is False

@pytest.mark.asyncio
async def test_websocket_connection_operations(mock_websocket_config):
    """Test WebSocket connection operations."""
    connection = WebSocketConnection(
        connection_id="ws_connection",
        config=mock_websocket_config
    )
    
    # Mock websockets
    with patch('websockets.connect') as mock_connect:
        mock_ws = AsyncMock()
        mock_connect.return_value.__aenter__.return_value = mock_ws
        
        # Test connect
        await connection.connect()
        assert connection.is_connected is True
        
        # Test send
        await connection.send({"data": "test"})
        mock_ws.send.assert_called_once()
        
        # Test receive
        mock_ws.recv.return_value = '{"result": "success"}'
        response = await connection.receive()
        assert response == {"result": "success"}
        
        # Test disconnect
        await connection.disconnect()
        assert connection.is_connected is False

@pytest.mark.asyncio
async def test_database_connection_operations(mock_database_config):
    """Test database connection operations."""
    connection = DatabaseConnection(
        connection_id="db_connection",
        config=mock_database_config
    )
    
    # Mock database connection
    with patch('asyncpg.create_pool') as mock_pool:
        mock_pool.return_value = AsyncMock()
        
        # Test connect
        await connection.connect()
        assert connection.is_connected is True
        
        # Test execute
        await connection.execute("SELECT * FROM test")
        mock_pool.return_value.execute.assert_called_once()
        
        # Test disconnect
        await connection.disconnect()
        assert connection.is_connected is False

@pytest.mark.asyncio
async def test_message_queue_connection_operations(mock_message_queue_config):
    """Test message queue connection operations."""
    connection = MessageQueueConnection(
        connection_id="mq_connection",
        config=mock_message_queue_config
    )
    
    # Mock message queue connection
    with patch('aio_pika.connect_robust') as mock_connect:
        mock_channel = AsyncMock()
        mock_connect.return_value.channel.return_value = mock_channel
        
        # Test connect
        await connection.connect()
        assert connection.is_connected is True
        
        # Test publish
        await connection.publish({"data": "test"})
        mock_channel.default_exchange.publish.assert_called_once()
        
        # Test consume
        mock_channel.consume.return_value = AsyncMock()
        await connection.consume(lambda msg: None)
        mock_channel.consume.assert_called_once()
        
        # Test disconnect
        await connection.disconnect()
        assert connection.is_connected is False

def test_connection_error_handling():
    """Test connection error handling."""
    connection = ServiceConnection(connection_id="test_connection")
    
    # Test error setting
    connection.set_error("Test error")
    assert connection.last_error == "Test error"
    
    # Test error clearing
    connection.clear_error()
    assert connection.last_error is None

def test_connection_validation():
    """Test connection validation."""
    connection = ServiceConnection(connection_id="test_connection")
    
    # Test required configuration
    connection.required_config = ["url"]
    assert not connection.validate_config({})
    
    assert connection.validate_config({"url": "http://test.com"})

def test_connection_serialization():
    """Test connection serialization."""
    connection = ServiceConnection(connection_id="test_connection")
    connection.set_error("Test error")
    
    serialized = connection.serialize()
    
    assert serialized["connection_id"] == "test_connection"
    assert serialized["is_connected"] is False
    assert serialized["last_error"] == "Test error"

def test_connection_deserialization():
    """Test connection deserialization."""
    serialized_data = {
        "connection_id": "test_connection",
        "is_connected": True,
        "last_error": "Test error"
    }
    
    connection = ServiceConnection.deserialize(serialized_data)
    
    assert connection.connection_id == "test_connection"
    assert connection.is_connected is True
    assert connection.last_error == "Test error" 