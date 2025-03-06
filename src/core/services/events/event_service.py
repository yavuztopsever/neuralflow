"""
Event management utilities for the LangGraph application.
This module provides functionality for managing workflow events and notifications.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from datetime import datetime
from utils.common.text import TextProcessor
from utils.error.handlers import ErrorHandler
from config.manager import ConfigManager
from utils.logging.manager import LogManager

logger = logging.getLogger(__name__)

class WorkflowEvent:
    """Represents a workflow event."""
    
    def __init__(self, event_id: str,
                 event_type: str,
                 workflow_id: str,
                 data: Dict[str, Any],
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a workflow event.
        
        Args:
            event_id: Unique identifier for the event
            event_type: Type of event
            workflow_id: ID of the workflow
            data: Event data
            metadata: Optional metadata for the event
        """
        self.id = event_id
        self.type = event_type
        self.workflow_id = workflow_id
        self.data = data
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.handled = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary.
        
        Returns:
            Dictionary representation of the event
        """
        return {
            'id': self.id,
            'type': self.type,
            'workflow_id': self.workflow_id,
            'data': self.data,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'handled': self.handled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowEvent':
        """Create event from dictionary.
        
        Args:
            data: Dictionary representation of the event
            
        Returns:
            WorkflowEvent instance
        """
        event = cls(
            event_id=data['id'],
            event_type=data['type'],
            workflow_id=data['workflow_id'],
            data=data['data'],
            metadata=data.get('metadata')
        )
        event.timestamp = data['timestamp']
        event.handled = data['handled']
        return event

class EventHandler:
    """Handles workflow events."""
    
    def __init__(self, handler_id: str,
                 event_types: List[str],
                 handler: Callable,
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize an event handler.
        
        Args:
            handler_id: Unique identifier for the handler
            event_types: List of event types to handle
            handler: Function to handle events
            metadata: Optional metadata for the handler
        """
        self.id = handler_id
        self.event_types = event_types
        self.handler = handler
        self.metadata = metadata or {}
        self.created = datetime.now().isoformat()
        self.modified = self.created
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type.
        
        Args:
            event_type: Event type to check
            
        Returns:
            True if handler can handle event type
        """
        return event_type in self.event_types
    
    def handle(self, event: WorkflowEvent) -> bool:
        """Handle an event.
        
        Args:
            event: Event to handle
            
        Returns:
            True if event was handled successfully
        """
        try:
            if not self.can_handle(event.type):
                return False
            
            self.handler(event)
            event.handled = True
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to handle event {event.id} with handler {self.id}: {e}")
            return False

class EventManager:
    """Manages workflow events and handlers."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the event manager.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.log_manager = LogManager(self.config)
        self.text_processor = TextProcessor()
        self._events = []
        self._handlers = {}
        self._initialize_handlers()
    
    def _initialize_handlers(self):
        """Initialize default event handlers."""
        try:
            # Create default handlers
            self.add_handler(
                'workflow_started',
                ['workflow_started'],
                self._handle_workflow_started
            )
            self.add_handler(
                'workflow_completed',
                ['workflow_completed'],
                self._handle_workflow_completed
            )
            self.add_handler(
                'workflow_failed',
                ['workflow_failed'],
                self._handle_workflow_failed
            )
            
            logger.info("Initialized event handlers")
        except Exception as e:
            logger.error(f"Failed to initialize event handlers: {e}")
            raise
    
    def add_handler(self, handler_id: str,
                   event_types: List[str],
                   handler: Callable,
                   metadata: Optional[Dict[str, Any]] = None) -> EventHandler:
        """Add an event handler.
        
        Args:
            handler_id: Unique identifier for the handler
            event_types: List of event types to handle
            handler: Function to handle events
            metadata: Optional metadata for the handler
            
        Returns:
            Created event handler
            
        Raises:
            ValueError: If handler_id already exists
        """
        try:
            if handler_id in self._handlers:
                raise ValueError(f"Handler {handler_id} already exists")
            
            event_handler = EventHandler(handler_id, event_types, handler, metadata)
            self._handlers[handler_id] = event_handler
            logger.info(f"Added event handler {handler_id}")
            return event_handler
        except Exception as e:
            logger.error(f"Failed to add event handler {handler_id}: {e}")
            raise
    
    def get_handler(self, handler_id: str) -> Optional[EventHandler]:
        """Get a handler by ID.
        
        Args:
            handler_id: Handler ID
            
        Returns:
            EventHandler instance or None if not found
        """
        return self._handlers.get(handler_id)
    
    def remove_handler(self, handler_id: str) -> bool:
        """Remove an event handler.
        
        Args:
            handler_id: Handler ID
            
        Returns:
            True if handler was removed, False otherwise
        """
        try:
            if handler_id not in self._handlers:
                return False
            
            del self._handlers[handler_id]
            logger.info(f"Removed event handler {handler_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove event handler {handler_id}: {e}")
            return False
    
    def emit_event(self, event_type: str,
                  workflow_id: str,
                  data: Dict[str, Any],
                  metadata: Optional[Dict[str, Any]] = None) -> WorkflowEvent:
        """Emit a new event.
        
        Args:
            event_type: Type of event
            workflow_id: ID of the workflow
            data: Event data
            metadata: Optional metadata
            
        Returns:
            Created event
        """
        try:
            # Generate event ID
            event_id = f"{event_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create event
            event = WorkflowEvent(event_id, event_type, workflow_id, data, metadata)
            self._events.append(event)
            
            # Handle event
            self._handle_event(event)
            
            logger.info(f"Emitted event {event_id}")
            return event
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")
            raise
    
    def _handle_event(self, event: WorkflowEvent) -> None:
        """Handle an event with all applicable handlers.
        
        Args:
            event: Event to handle
        """
        try:
            for handler in self._handlers.values():
                if handler.can_handle(event.type):
                    handler.handle(event)
        except Exception as e:
            logger.error(f"Failed to handle event {event.id}: {e}")
    
    def get_events(self, workflow_id: Optional[str] = None,
                  event_type: Optional[str] = None,
                  handled: Optional[bool] = None) -> List[WorkflowEvent]:
        """Get events matching criteria.
        
        Args:
            workflow_id: Optional workflow ID to filter by
            event_type: Optional event type to filter by
            handled: Optional handled status to filter by
            
        Returns:
            List of matching events
        """
        events = self._events
        
        if workflow_id is not None:
            events = [e for e in events if e.workflow_id == workflow_id]
        
        if event_type is not None:
            events = [e for e in events if e.type == event_type]
        
        if handled is not None:
            events = [e for e in events if e.handled == handled]
        
        return events
    
    def get_event_stats(self) -> Dict[str, Any]:
        """Get statistics about events.
        
        Returns:
            Dictionary containing event statistics
        """
        try:
            return {
                'total_events': len(self._events),
                'handlers': len(self._handlers),
                'event_types': {
                    event_type: len([e for e in self._events if e.type == event_type])
                    for event_type in set(e.type for e in self._events)
                },
                'handled_events': len([e for e in self._events if e.handled]),
                'unhandled_events': len([e for e in self._events if not e.handled])
            }
        except Exception as e:
            logger.error(f"Failed to get event stats: {e}")
            return {}
    
    def _handle_workflow_started(self, event: WorkflowEvent) -> None:
        """Handle workflow started event.
        
        Args:
            event: Event to handle
        """
        try:
            logger.info(f"Workflow {event.workflow_id} started")
        except Exception as e:
            logger.error(f"Failed to handle workflow started event: {e}")
    
    def _handle_workflow_completed(self, event: WorkflowEvent) -> None:
        """Handle workflow completed event.
        
        Args:
            event: Event to handle
        """
        try:
            logger.info(f"Workflow {event.workflow_id} completed")
        except Exception as e:
            logger.error(f"Failed to handle workflow completed event: {e}")
    
    def _handle_workflow_failed(self, event: WorkflowEvent) -> None:
        """Handle workflow failed event.
        
        Args:
            event: Event to handle
        """
        try:
            logger.error(f"Workflow {event.workflow_id} failed: {event.data.get('error')}")
        except Exception as e:
            logger.error(f"Failed to handle workflow failed event: {e}") 