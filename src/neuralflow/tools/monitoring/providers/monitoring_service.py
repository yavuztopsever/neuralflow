"""
Monitoring service for collecting metrics and events.
"""

import os
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

from ....services.base import BaseService
from ....storage.manager import StorageManager
from ....models.metric import Metric
from ....models.event import Event

logger = logging.getLogger(__name__)

class MonitoringService(BaseService[Union[Metric, Event]]):
    """Service for unified monitoring including metrics and events."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the monitoring service.
        
        Args:
            storage_dir: Optional directory for storing monitoring data
        """
        super().__init__()
        
        # Initialize storage
        self.storage_dir = Path(storage_dir or os.path.join(os.getcwd(), "storage", "monitoring"))
        self.storage_manager = StorageManager(self.storage_dir)
        
        # Get storage providers
        self.metric_store = self.storage_manager.get_provider("metric")
        self.event_store = self.storage_manager.get_provider("event")
        
        if not all([self.metric_store, self.event_store]):
            raise RuntimeError("Failed to initialize required storage providers")
    
    def record_metric(self, metric: Metric) -> bool:
        """Record a metric.
        
        Args:
            metric: Metric to record
            
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.metric_store.store(metric)
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
            return False
    
    def record_event(self, event: Event) -> bool:
        """Record an event.
        
        Args:
            event: Event to record
            
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.event_store.store(event)
        except Exception as e:
            logger.error(f"Failed to record event: {e}")
            return False
    
    def get_metrics(self, metric_type: Optional[str] = None) -> List[Metric]:
        """Get recorded metrics.
        
        Args:
            metric_type: Optional metric type to filter by
            
        Returns:
            List of metrics
        """
        try:
            metrics = self.metric_store.list()
            if metric_type:
                metrics = [m for m in metrics if m.type == metric_type]
            return metrics
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []
    
    def get_events(self, event_type: Optional[str] = None) -> List[Event]:
        """Get recorded events.
        
        Args:
            event_type: Optional event type to filter by
            
        Returns:
            List of events
        """
        try:
            events = self.event_store.list()
            if event_type:
                events = [e for e in events if e.type == event_type]
            return events
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []
    
    def get_monitoring_info(self) -> Dict[str, Any]:
        """Get monitoring information.
        
        Returns:
            Dictionary containing monitoring information
        """
        return {
            'storage': self.storage_manager.get_storage_info(),
            'metrics': {
                'total': len(self.get_metrics()),
                'types': {
                    metric_type: len(self.get_metrics(metric_type))
                    for metric_type in set(m.type for m in self.get_metrics())
                }
            },
            'events': {
                'total': len(self.get_events()),
                'types': {
                    event_type: len(self.get_events(event_type))
                    for event_type in set(e.type for e in self.get_events())
                }
            }
        }
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics.
        
        Returns:
            Dictionary containing monitoring statistics
        """
        return {
            'storage': self.storage_manager.get_storage_stats(),
            'metrics': {
                'total': len(self.get_metrics()),
                'latest': next(iter(self.get_metrics()), None),
                'types': {
                    metric_type: len(self.get_metrics(metric_type))
                    for metric_type in set(m.type for m in self.get_metrics())
                }
            },
            'events': {
                'total': len(self.get_events()),
                'latest': next(iter(self.get_events()), None),
                'types': {
                    event_type: len(self.get_events(event_type))
                    for event_type in set(e.type for e in self.get_events())
                }
            }
        }
    
    def cleanup(self) -> None:
        """Clean up monitoring resources."""
        self.storage_manager.cleanup()
    
    def __del__(self):
        """Clean up resources on deletion."""
        self.cleanup()

__all__ = ['MonitoringService', 'Metric', 'Event'] 