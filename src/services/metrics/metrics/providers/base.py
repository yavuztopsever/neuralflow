"""
Base provider interfaces for metrics providers.
This module provides base classes for metrics provider implementations.
"""

import logging
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')

class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class MetricValue:
    """Metric value with metadata."""
    
    value: float
    timestamp: datetime
    labels: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric value to dictionary.
        
        Returns:
            Dictionary representation of the metric value
        """
        return {
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels or {}
        }

@dataclass
class Metric:
    """Metric with metadata."""
    
    name: str
    type: MetricType
    description: Optional[str] = None
    unit: Optional[str] = None
    values: Optional[List[MetricValue]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary.
        
        Returns:
            Dictionary representation of the metric
        """
        return {
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'unit': self.unit,
            'values': [v.to_dict() for v in (self.values or [])]
        }

class MetricsConfig:
    """Configuration for metrics providers."""
    
    def __init__(self,
                 max_values: Optional[int] = None,
                 retention_days: Optional[int] = None,
                 **kwargs):
        """Initialize the configuration.
        
        Args:
            max_values: Maximum number of values to store per metric
            retention_days: Number of days to retain values
            **kwargs: Additional configuration parameters
        """
        self.max_values = max_values
        self.retention_days = retention_days
        self.extra_params = kwargs

class BaseMetricsProvider(ABC):
    """Base class for metrics providers."""
    
    def __init__(self, provider_id: str,
                 config: MetricsConfig,
                 **kwargs):
        """Initialize the provider.
        
        Args:
            provider_id: Unique identifier for the provider
            config: Metrics provider configuration
            **kwargs: Additional initialization parameters
        """
        self.id = provider_id
        self.config = config
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self._kwargs = kwargs
        self._metrics: Dict[str, Metric] = {}
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up provider resources."""
        pass
    
    def create_metric(self,
                     name: str,
                     metric_type: MetricType,
                     description: Optional[str] = None,
                     unit: Optional[str] = None) -> bool:
        """Create a new metric.
        
        Args:
            name: Metric name
            metric_type: Type of metric
            description: Optional metric description
            unit: Optional metric unit
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if name in self._metrics:
                return False
            
            metric = Metric(
                name=name,
                type=metric_type,
                description=description,
                unit=unit,
                values=[]
            )
            
            self._metrics[name] = metric
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to create metric {name} in provider {self.id}: {e}")
            return False
    
    def record_value(self,
                    name: str,
                    value: float,
                    labels: Optional[Dict[str, str]] = None) -> bool:
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Value to record
            labels: Optional value labels
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if name not in self._metrics:
                return False
            
            metric = self._metrics[name]
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.now(),
                labels=labels
            )
            
            metric.values = metric.values or []
            metric.values.append(metric_value)
            
            # Check max values
            if (self.config.max_values is not None and
                len(metric.values) > self.config.max_values):
                metric.values = metric.values[-self.config.max_values:]
            
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to record value for metric {name} in provider {self.id}: {e}")
            return False
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            Metric or None if not found
        """
        return self._metrics.get(name)
    
    def get_metric_values(self,
                         name: str,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         labels: Optional[Dict[str, str]] = None) -> List[MetricValue]:
        """Get metric values matching criteria.
        
        Args:
            name: Metric name
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            labels: Optional labels to filter by
            
        Returns:
            List of matching metric values
        """
        try:
            metric = self._metrics.get(name)
            if not metric or not metric.values:
                return []
            
            values = metric.values
            
            if start_time:
                values = [v for v in values if v.timestamp >= start_time]
            
            if end_time:
                values = [v for v in values if v.timestamp <= end_time]
            
            if labels:
                values = [
                    v for v in values
                    if all(v.labels.get(k) == v for k, v in labels.items())
                ]
            
            return values
        except Exception as e:
            logger.error(f"Failed to get values for metric {name} from provider {self.id}: {e}")
            return []
    
    def delete_metric(self, name: str) -> bool:
        """Delete a metric.
        
        Args:
            name: Metric name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if name not in self._metrics:
                return False
            
            del self._metrics[name]
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to delete metric {name} from provider {self.id}: {e}")
            return False
    
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
                'max_values': self.config.max_values,
                'retention_days': self.config.retention_days,
                'extra_params': self.config.extra_params
            },
            'stats': {
                'total_metrics': len(self._metrics),
                'total_values': sum(len(m.values or []) for m in self._metrics.values())
            }
        }
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics.
        
        Returns:
            Dictionary containing provider statistics
        """
        try:
            return {
                'total_metrics': len(self._metrics),
                'metric_types': {
                    mt.value: len([m for m in self._metrics.values() if m.type == mt])
                    for mt in MetricType
                },
                'values_per_metric': {
                    name: len(metric.values or [])
                    for name, metric in self._metrics.items()
                }
            }
        except Exception as e:
            logger.error(f"Failed to get provider stats: {e}")
            return {} 