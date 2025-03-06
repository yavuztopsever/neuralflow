"""
Base provider interfaces for event providers.
This module provides base classes for event provider implementations.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')

class EventType(Enum):
    """Event types."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"
    TRACE = "trace"

@dataclass
class Event:
    """Event with metadata."""
    
    id: str
    type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary.
        
        Returns:
            Dictionary representation of the event
        """
        return {
            'id': self.id,
            'type': self.type.value,
            'source': self.source,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }

class EventConfig:
    """Configuration for event providers."""
    
    def __init__(self,
                 max_events: Optional[int] = None,
                 retention_days: Optional[int] = None,
                 **kwargs):
        """Initialize the configuration.
        
        Args:
            max_events: Maximum number of events to store
            retention_days: Number of days to retain events
            **kwargs: Additional configuration parameters
        """
        self.max_events = max_events
        self.retention_days = retention_days
        self.extra_params = kwargs

class BaseEventProvider(ABC):
    """Base class for event providers."""
    
    def __init__(self, provider_id: str,
                 config: EventConfig,
                 **kwargs):
        """Initialize the provider.
        
        Args:
            provider_id: Unique identifier for the provider
            config: Event provider configuration
            **kwargs: Additional initialization parameters
        """
        self.id = provider_id
        self.config = config
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self._kwargs = kwargs
        self._handlers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._events: List[Event] = []
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up provider resources."""
        pass
    
    def emit(self,
            event_type: EventType,
            source: str,
            data: Dict[str, Any],
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Emit an event.
        
        Args:
            event_type: Type of event
            source: Event source
            data: Event data
            metadata: Optional event metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            event = Event(
                id=str(uuid.uuid4()),
                type=event_type,
                source=source,
                data=data,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            # Store event
            self._events.append(event)
            
            # Check max events
            if (self.config.max_events is not None and
                len(self._events) > self.config.max_events):
                self._events = self._events[-self.config.max_events:]
            
            # Notify handlers
            if event_type in self._handlers:
                for handler in self._handlers[event_type]:
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(f"Failed to handle event {event.id}: {e}")
            
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to emit event in provider {self.id}: {e}")
            return False
    
    def subscribe(self,
                 event_type: EventType,
                 handler: Callable[[Event], None]) -> bool:
        """Subscribe to events.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Event handler function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to subscribe to events in provider {self.id}: {e}")
            return False
    
    def unsubscribe(self,
                   event_type: EventType,
                   handler: Callable[[Event], None]) -> bool:
        """Unsubscribe from events.
        
        Args:
            event_type: Type of event to unsubscribe from
            handler: Event handler function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if event_type in self._handlers:
                if handler in self._handlers[event_type]:
                    self._handlers[event_type].remove(handler)
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to unsubscribe from events in provider {self.id}: {e}")
            return False
    
    def get_events(self,
                  event_type: Optional[EventType] = None,
                  source: Optional[str] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None) -> List[Event]:
        """Get events matching criteria.
        
        Args:
            event_type: Optional event type to filter by
            source: Optional source to filter by
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            
        Returns:
            List of matching events
        """
        try:
            events = self._events
            
            if event_type:
                events = [e for e in events if e.type == event_type]
            
            if source:
                events = [e for e in events if e.source == source]
            
            if start_time:
                events = [e for e in events if e.timestamp >= start_time]
            
            if end_time:
                events = [e for e in events if e.timestamp <= end_time]
            
            return events
        except Exception as e:
            logger.error(f"Failed to get events from provider {self.id}: {e}")
            return []
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information.
        
        Returns:
            Dictionary containing provider information
        """
        return {
            'id': self.id,
            'type': type(self).__name__,
            'created': self.created,
            'modified': self.modified,
            'config': {
                'max_events': self.config.max_events,
                'retention_days': self.config.retention_days,
                'extra_params': self.config.extra_params
            },
            'stats': {
                'total_events': len(self._events),
                'handlers': {
                    et.value: len(handlers)
                    for et, handlers in self._handlers.items()
                }
            }
        }
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics.
        
        Returns:
            Dictionary containing provider statistics
        """
        try:
            return {
                'total_events': len(self._events),
                'event_types': {
                    et.value: len([e for e in self._events if e.type == et])
                    for et in EventType
                },
                'sources': {
                    source: len([e for e in self._events if e.source == source])
                    for source in set(e.source for e in self._events)
                }
            }
        except Exception as e:
            logger.error(f"Failed to get provider stats: {e}")
            return {} 