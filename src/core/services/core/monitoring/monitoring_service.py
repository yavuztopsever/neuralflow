"""
Monitoring service for the LangGraph project.
This service provides unified monitoring capabilities including metrics and events.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from ...services.base_service import BaseService
import json
import os
from pathlib import Path

class Metric(BaseModel):
    """Metric model."""
    name: str
    value: Any
    type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class Event(BaseModel):
    """Event model."""
    name: str
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class MonitoringService(BaseService[Union[Metric, Event]]):
    """Service for unified monitoring including metrics and events."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the monitoring service.
        
        Args:
            storage_dir: Optional directory for storing monitoring data
        """
        super().__init__()
        self.metrics: Dict[str, Metric] = {}
        self.events: List[Event] = []
        self.storage_dir = Path(storage_dir or os.path.join(os.getcwd(), "storage", "monitoring"))
        self._ensure_storage()
        self._load_data()
    
    def _ensure_storage(self) -> None:
        """Ensure storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "metrics").mkdir(exist_ok=True)
        (self.storage_dir / "events").mkdir(exist_ok=True)
    
    def _load_data(self) -> None:
        """Load monitoring data from storage."""
        try:
            # Load metrics
            metrics_dir = self.storage_dir / "metrics"
            for metric_file in metrics_dir.glob("*.json"):
                try:
                    with open(metric_file, "r") as f:
                        data = json.load(f)
                        metric = Metric(**data)
                        self.metrics[metric.name] = metric
                except Exception as e:
                    self.logger.error(f"Failed to load metric from {metric_file}: {e}")
            
            # Load events
            events_dir = self.storage_dir / "events"
            for event_file in events_dir.glob("*.json"):
                try:
                    with open(event_file, "r") as f:
                        data = json.load(f)
                        event = Event(**data)
                        self.events.append(event)
                except Exception as e:
                    self.logger.error(f"Failed to load event from {event_file}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load monitoring data: {e}")
    
    def record_metric(
        self,
        name: str,
        value: Any,
        metric_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Metric:
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            metadata: Optional metadata
            
        Returns:
            Metric: Created metric
        """
        try:
            metric = Metric(
                name=name,
                value=value,
                type=metric_type,
                metadata=metadata
            )
            
            # Store metric
            self.metrics[name] = metric
            
            # Save to storage
            self._save_metric(metric)
            
            # Record in history
            self.record_history(
                "record_metric",
                details={
                    "name": name,
                    "value": value,
                    "type": metric_type
                },
                metadata=metadata
            )
            
            return metric
            
        except Exception as e:
            self.logger.error(f"Failed to record metric: {e}")
            raise
    
    def record_event(
        self,
        name: str,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """
        Record an event.
        
        Args:
            name: Event name
            event_type: Type of event
            data: Event data
            metadata: Optional metadata
            
        Returns:
            Event: Created event
        """
        try:
            event = Event(
                name=name,
                type=event_type,
                data=data,
                metadata=metadata
            )
            
            # Store event
            self.events.append(event)
            
            # Save to storage
            self._save_event(event)
            
            # Record in history
            self.record_history(
                "record_event",
                details={
                    "name": name,
                    "type": event_type,
                    "data": data
                },
                metadata=metadata
            )
            
            return event
            
        except Exception as e:
            self.logger.error(f"Failed to record event: {e}")
            raise
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """
        Get a metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            Optional[Metric]: Metric if found
        """
        return self.metrics.get(name)
    
    def get_metrics(
        self,
        metric_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Metric]:
        """
        Get metrics.
        
        Args:
            metric_type: Optional metric type to filter by
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            
        Returns:
            List[Metric]: List of metrics
        """
        metrics = list(self.metrics.values())
        
        if metric_type:
            metrics = [m for m in metrics if m.type == metric_type]
        
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
        
        return metrics
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Event]:
        """
        Get events.
        
        Args:
            event_type: Optional event type to filter by
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            
        Returns:
            List[Event]: List of events
        """
        events = self.events
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return events
    
    def _save_metric(self, metric: Metric) -> None:
        """
        Save metric to storage.
        
        Args:
            metric: Metric to save
        """
        try:
            file_path = self.storage_dir / "metrics" / f"{metric.name}.json"
            with open(file_path, "w") as f:
                json.dump(metric.dict(), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save metric: {e}")
            raise
    
    def _save_event(self, event: Event) -> None:
        """
        Save event to storage.
        
        Args:
            event: Event to save
        """
        try:
            file_path = self.storage_dir / "events" / f"{event.name}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            with open(file_path, "w") as f:
                json.dump(event.dict(), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save event: {e}")
            raise
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get monitoring summary.
        
        Returns:
            Dict[str, Any]: Monitoring summary
        """
        return {
            "total_metrics": len(self.metrics),
            "metric_types": {
                metric_type: len([m for m in self.metrics.values() if m.type == metric_type])
                for metric_type in set(m.type for m in self.metrics.values())
            },
            "total_events": len(self.events),
            "event_types": {
                event_type: len([e for e in self.events if e.type == event_type])
                for event_type in set(e.type for e in self.events)
            },
            "latest_metric": next(iter(self.metrics.values())) if self.metrics else None,
            "latest_event": self.events[-1] if self.events else None
        }
    
    def reset(self) -> None:
        """Reset monitoring service."""
        super().reset()
        self.metrics = {}
        self.events = []

__all__ = ['MonitoringService', 'Metric', 'Event'] 