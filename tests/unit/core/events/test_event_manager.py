"""
Unit tests for event manager functionality.
"""
import pytest
from datetime import datetime
from typing import Dict, Any, List, Callable

from src.core.events import EventManager, Event, EventType, EventHandler

class TestEventManager:
    """Test suite for event manager functionality."""
    
    @pytest.fixture
    def event_manager(self):
        """Create an event manager for testing."""
        return EventManager(
            max_events=100,
            event_ttl=3600  # 1 hour
        )
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample event for testing."""
        return Event(
            id="test_event",
            type=EventType.WORKFLOW_STARTED,
            data={
                "workflow_id": "test_workflow",
                "timestamp": datetime.now().isoformat()
            },
            metadata={
                "user_id": "test_user",
                "priority": "high"
            }
        )
    
    @pytest.fixture
    def sample_handler(self):
        """Create a sample event handler for testing."""
        async def handler(event: Event) -> None:
            pass
        return EventHandler(
            id="test_handler",
            type=EventType.WORKFLOW_STARTED,
            handler=handler,
            priority=1
        )
    
    @pytest.mark.asyncio
    async def test_event_manager_initialization(self, event_manager):
        """Test event manager initialization."""
        assert event_manager.max_events == 100
        assert event_manager.event_ttl == 3600
        assert event_manager._events == []
        assert event_manager._handlers == {}
        assert event_manager._last_cleanup is not None
    
    @pytest.mark.asyncio
    async def test_event_management(self, event_manager, sample_event):
        """Test event management operations."""
        # Emit event
        await event_manager.emit_event(sample_event)
        assert len(event_manager._events) == 1
        stored_event = event_manager._events[0]
        assert stored_event.type == sample_event.type
        assert stored_event.data == sample_event.data
        
        # Get event
        retrieved_event = await event_manager.get_event(sample_event.id)
        assert retrieved_event is not None
        assert retrieved_event.id == sample_event.id
        
        # Update event
        updated_event = Event(
            id=sample_event.id,
            type=sample_event.type,
            data={
                "workflow_id": "test_workflow",
                "timestamp": sample_event.data["timestamp"],
                "status": "completed"
            },
            metadata=sample_event.metadata
        )
        await event_manager.update_event(updated_event)
        retrieved_event = await event_manager.get_event(sample_event.id)
        assert retrieved_event.data["status"] == "completed"
        
        # Delete event
        await event_manager.delete_event(sample_event.id)
        assert len(event_manager._events) == 0
    
    @pytest.mark.asyncio
    async def test_event_limits(self, event_manager):
        """Test event limit enforcement."""
        # Emit maximum allowed events
        events = []
        for i in range(event_manager.max_events):
            event = Event(
                id=f"test_event_{i}",
                type=EventType.WORKFLOW_STARTED,
                data={"workflow_id": f"workflow_{i}"},
                metadata={}
            )
            await event_manager.emit_event(event)
            events.append(event)
        
        # Try to emit one more event
        extra_event = Event(
            id="extra_event",
            type=EventType.WORKFLOW_STARTED,
            data={"workflow_id": "extra_workflow"},
            metadata={}
        )
        with pytest.raises(ValueError):
            await event_manager.emit_event(extra_event)
        
        # Delete one event and emit a new one
        await event_manager.delete_event(events[0].id)
        new_event = Event(
            id="new_event",
            type=EventType.WORKFLOW_STARTED,
            data={"workflow_id": "new_workflow"},
            metadata={}
        )
        await event_manager.emit_event(new_event)
        assert len(event_manager._events) == event_manager.max_events
    
    @pytest.mark.asyncio
    async def test_event_expiration(self, event_manager, sample_event):
        """Test event expiration handling."""
        # Emit event with short TTL
        event_manager.event_ttl = 1  # 1 second
        await event_manager.emit_event(sample_event)
        
        # Wait for event to expire
        await asyncio.sleep(2)
        
        # Verify event is expired
        retrieved_event = await event_manager.get_event(sample_event.id)
        assert retrieved_event is None
    
    @pytest.mark.asyncio
    async def test_event_cleanup(self, event_manager, sample_event):
        """Test event cleanup functionality."""
        await event_manager.emit_event(sample_event)
        
        # Force cleanup
        await event_manager._cleanup_expired_events()
        
        # Verify event still exists (not expired)
        assert len(event_manager._events) == 1
        
        # Expire event
        event_manager._events[0].timestamp = (
            datetime.now() - timedelta(seconds=event_manager.event_ttl + 1)
        )
        
        # Run cleanup
        await event_manager._cleanup_expired_events()
        
        # Verify event is removed
        assert len(event_manager._events) == 0
    
    @pytest.mark.asyncio
    async def test_event_metrics(self, event_manager, sample_event):
        """Test event metrics collection."""
        await event_manager.emit_event(sample_event)
        
        # Get metrics
        metrics = await event_manager.get_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "total_events" in metrics
        assert "active_events" in metrics
        assert "event_types" in metrics
    
    @pytest.mark.asyncio
    async def test_event_error_handling(self, event_manager):
        """Test event error handling."""
        # Test invalid event
        with pytest.raises(ValueError):
            await event_manager.emit_event(None)
        
        # Test invalid event ID
        with pytest.raises(ValueError):
            await event_manager.get_event(None)
        
        # Test getting non-existent event
        event = await event_manager.get_event("non_existent")
        assert event is None
        
        # Test updating non-existent event
        with pytest.raises(KeyError):
            await event_manager.update_event(Event(
                id="non_existent",
                type=EventType.WORKFLOW_STARTED,
                data={},
                metadata={}
            ))
        
        # Test deleting non-existent event
        with pytest.raises(KeyError):
            await event_manager.delete_event("non_existent")
    
    @pytest.mark.asyncio
    async def test_event_handler_management(self, event_manager, sample_handler):
        """Test event handler management."""
        # Register handler
        await event_manager.register_handler(sample_handler)
        assert sample_handler.id in event_manager._handlers
        stored_handler = event_manager._handlers[sample_handler.id]
        assert stored_handler.type == sample_handler.type
        assert stored_handler.priority == sample_handler.priority
        
        # Get handler
        retrieved_handler = await event_manager.get_handler(sample_handler.id)
        assert retrieved_handler is not None
        assert retrieved_handler.id == sample_handler.id
        
        # Update handler
        updated_handler = EventHandler(
            id=sample_handler.id,
            type=sample_handler.type,
            handler=sample_handler.handler,
            priority=2
        )
        await event_manager.update_handler(updated_handler)
        retrieved_handler = await event_manager.get_handler(sample_handler.id)
        assert retrieved_handler.priority == 2
        
        # Unregister handler
        await event_manager.unregister_handler(sample_handler.id)
        assert sample_handler.id not in event_manager._handlers
    
    @pytest.mark.asyncio
    async def test_event_handling(self, event_manager, sample_event):
        """Test event handling functionality."""
        # Create a handler that records events
        handled_events = []
        async def handler(event: Event) -> None:
            handled_events.append(event)
        
        handler_obj = EventHandler(
            id="test_handler",
            type=EventType.WORKFLOW_STARTED,
            handler=handler,
            priority=1
        )
        
        # Register handler
        await event_manager.register_handler(handler_obj)
        
        # Emit event
        await event_manager.emit_event(sample_event)
        
        # Verify handler was called
        assert len(handled_events) == 1
        assert handled_events[0].id == sample_event.id
    
    @pytest.mark.asyncio
    async def test_event_type_validation(self, event_manager):
        """Test event type validation."""
        # Test invalid event type
        with pytest.raises(ValueError):
            await event_manager.emit_event(Event(
                id="test_event",
                type="invalid_type",  # type: ignore
                data={},
                metadata={}
            ))
        
        # Test event type filtering
        workflow_event = Event(
            id="workflow_event",
            type=EventType.WORKFLOW_STARTED,
            data={},
            metadata={}
        )
        await event_manager.emit_event(workflow_event)
        
        session_event = Event(
            id="session_event",
            type=EventType.SESSION_STARTED,
            data={},
            metadata={}
        )
        await event_manager.emit_event(session_event)
        
        # Get events by type
        workflow_events = await event_manager.get_events_by_type(EventType.WORKFLOW_STARTED)
        assert len(workflow_events) == 1
        assert workflow_events[0].id == "workflow_event"
        
        session_events = await event_manager.get_events_by_type(EventType.SESSION_STARTED)
        assert len(session_events) == 1
        assert session_events[0].id == "session_event" 