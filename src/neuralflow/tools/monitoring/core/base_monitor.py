"""
Base monitoring system for NeuralFlow.
Provides unified monitoring capabilities including metrics, events, and alerts.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Union, List, TypeVar, Generic, Callable
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel

from ..storage.base import BaseStorage, StorageConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')

class MonitoringConfig:
    """Configuration for monitoring systems."""
    
    def __init__(self,
                 monitor_id: str,
                 monitor_type: str,
                 storage_config: Optional[StorageConfig] = None,
                 alert_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """Initialize monitoring configuration.
        
        Args:
            monitor_id: Unique identifier for the monitor
            monitor_type: Type of monitor (metric, event, alert)
            storage_config: Optional storage configuration
            alert_config: Optional alert configuration
            **kwargs: Additional configuration parameters
        """
        self.id = monitor_id
        self.type = monitor_type
        self.storage_config = storage_config
        self.alert_config = alert_config or {}
        self.parameters = kwargs
        self.metadata = {}

class BaseMetric(BaseModel):
    """Base metric model."""
    id: str
    name: str
    value: Any
    type: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class BaseEvent(BaseModel):
    """Base event model."""
    id: str
    name: str
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class BaseAlert(BaseModel):
    """Base alert model."""
    id: str
    name: str
    type: str
    level: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class BaseMonitor(ABC, Generic[T]):
    """Base class for all monitoring implementations."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize the monitoring system.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self._handlers: Dict[str, List[Callable]] = {
            'metric': [],
            'event': [],
            'alert': []
        }
        self._storage = self._initialize_storage()
        self._alert_queue = asyncio.Queue()
        self._processing = False
    
    def _initialize_storage(self) -> Optional[BaseStorage]:
        """Initialize storage if configured."""
        if self.config.storage_config:
            return BaseStorage(self.config.storage_config)
        return None
    
    async def start(self) -> None:
        """Start monitoring."""
        self._processing = True
        asyncio.create_task(self._process_alerts())
    
    async def stop(self) -> None:
        """Stop monitoring."""
        self._processing = False
        if self._storage:
            await self._storage.cleanup()
    
    def add_handler(self, event_type: str, handler: Callable) -> None:
        """Add an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        if event_type in self._handlers:
            self._handlers[event_type].append(handler)
    
    def remove_handler(self, event_type: str, handler: Callable) -> None:
        """Remove an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
    
    async def record_metric(self, metric: BaseMetric) -> bool:
        """Record a metric.
        
        Args:
            metric: Metric to record
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store metric
            if self._storage:
                await self._storage.store(metric)
            
            # Call handlers
            for handler in self._handlers['metric']:
                await handler(metric)
            
            return True
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
            return False
    
    async def record_event(self, event: BaseEvent) -> bool:
        """Record an event.
        
        Args:
            event: Event to record
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store event
            if self._storage:
                await self._storage.store(event)
            
            # Call handlers
            for handler in self._handlers['event']:
                await handler(event)
            
            return True
        except Exception as e:
            logger.error(f"Failed to record event: {e}")
            return False
    
    async def trigger_alert(self, alert: BaseAlert) -> bool:
        """Trigger an alert.
        
        Args:
            alert: Alert to trigger
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add to queue
            await self._alert_queue.put(alert)
            
            # Store alert
            if self._storage:
                await self._storage.store(alert)
            
            return True
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")
            return False
    
    async def _process_alerts(self) -> None:
        """Process alerts from queue."""
        while self._processing:
            try:
                # Get alert from queue
                alert = await self._alert_queue.get()
                
                # Call handlers
                for handler in self._handlers['alert']:
                    try:
                        await handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler failed: {e}")
                
                self._alert_queue.task_done()
            except Exception as e:
                logger.error(f"Failed to process alert: {e}")
            
            await asyncio.sleep(0.1)
    
    def get_monitor_info(self) -> Dict[str, Any]:
        """Get monitor information.
        
        Returns:
            Dictionary containing monitor information
        """
        return {
            'id': self.config.id,
            'type': self.config.type,
            'created': self.created,
            'modified': self.modified,
            'parameters': self.config.parameters,
            'metadata': self.config.metadata,
            'handlers': {
                event_type: len(handlers)
                for event_type, handlers in self._handlers.items()
            },
            'storage': self._storage.get_storage_info() if self._storage else None
        }
    
    def get_monitor_stats(self) -> Dict[str, Any]:
        """Get monitor statistics.
        
        Returns:
            Dictionary containing monitor statistics
        """
        try:
            stats = {
                'handlers': {
                    event_type: len(handlers)
                    for event_type, handlers in self._handlers.items()
                },
                'alert_queue_size': self._alert_queue.qsize()
            }
            
            if self._storage:
                stats['storage'] = self._storage.get_storage_stats()
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get monitor stats: {e}")
            return {} 