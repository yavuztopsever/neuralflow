"""
Metrics management utilities specific to the LangGraph project.
These utilities handle metrics that are not covered by LangChain's built-in metrics.
"""

import json
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import statistics
from collections import defaultdict

T = TypeVar('T')

class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"
    CUSTOM = "custom"

@dataclass
class MetricValue:
    """A metric value with metadata."""
    value: Any
    timestamp: str
    metadata: Dict[str, Any]

class MetricsManager:
    """Manages metrics for the LangGraph project."""
    
    def __init__(
        self,
        metrics_dir: str = "metrics",
        retention_period: int = 30,  # days
        max_samples: int = 1000
    ):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.retention_period = retention_period
        self.max_samples = max_samples
        
        # Initialize metrics storage
        self.metrics: Dict[str, Dict[str, List[MetricValue]]] = defaultdict(lambda: defaultdict(list))
        self.lock = threading.Lock()
        
        # Load existing metrics
        self._load_metrics()

    def _load_metrics(self):
        """Load metrics from file."""
        metrics_file = self.metrics_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                for metric_name, metric_data in data.items():
                    for metric_type, values in metric_data.items():
                        self.metrics[metric_name][metric_type] = [
                            MetricValue(**v)
                            for v in values
                        ]

    def _save_metrics(self):
        """Save metrics to file."""
        metrics_file = self.metrics_dir / "metrics.json"
        data = {
            metric_name: {
                metric_type: [asdict(v) for v in values]
                for metric_type, values in metric_data.items()
            }
            for metric_name, metric_data in self.metrics.items()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)

    def record(
        self,
        name: str,
        value: Any,
        metric_type: MetricType,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric value."""
        with self.lock:
            # Create metric value
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {}
            )
            
            # Add to storage
            self.metrics[name][metric_type.value].append(metric_value)
            
            # Enforce max samples
            if len(self.metrics[name][metric_type.value]) > self.max_samples:
                self.metrics[name][metric_type.value].pop(0)
            
            # Save metrics
            self._save_metrics()

    def increment(self, name: str, value: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        """Increment a counter metric."""
        self.record(name, value, MetricType.COUNTER, metadata)

    def set_gauge(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Set a gauge metric value."""
        self.record(name, value, MetricType.GAUGE, metadata)

    def record_histogram(
        self,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a histogram metric value."""
        self.record(name, value, MetricType.HISTOGRAM, metadata)

    def record_timing(
        self,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a timing metric value."""
        self.record(name, value, MetricType.TIMING, metadata)

    def get_metric(
        self,
        name: str,
        metric_type: MetricType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricValue]:
        """Get metric values within a time range."""
        with self.lock:
            values = self.metrics[name][metric_type.value]
            
            if not start_time and not end_time:
                return values
                
            filtered_values = []
            for value in values:
                timestamp = datetime.fromisoformat(value.timestamp)
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                filtered_values.append(value)
                
            return filtered_values

    def get_metric_stats(
        self,
        name: str,
        metric_type: MetricType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get statistics for a metric."""
        values = self.get_metric(name, metric_type, start_time, end_time)
        
        if not values:
            return {
                'count': 0,
                'min': None,
                'max': None,
                'mean': None,
                'median': None,
                'stddev': None
            }
        
        numeric_values = [v.value for v in values if isinstance(v.value, (int, float))]
        
        if not numeric_values:
            return {
                'count': len(values),
                'min': None,
                'max': None,
                'mean': None,
                'median': None,
                'stddev': None
            }
        
        return {
            'count': len(values),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'mean': statistics.mean(numeric_values),
            'median': statistics.median(numeric_values),
            'stddev': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
        }

    def get_metric_summary(
        self,
        name: str,
        metric_type: MetricType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get a summary of metric values."""
        values = self.get_metric(name, metric_type, start_time, end_time)
        
        if not values:
            return {
                'count': 0,
                'latest': None,
                'earliest': None,
                'metadata': {}
            }
        
        return {
            'count': len(values),
            'latest': values[-1].value,
            'earliest': values[0].value,
            'metadata': values[-1].metadata
        }

    def cleanup(self):
        """Remove old metric values."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(days=self.retention_period)
            
            for metric_name in list(self.metrics.keys()):
                for metric_type in list(self.metrics[metric_name].keys()):
                    self.metrics[metric_name][metric_type] = [
                        v for v in self.metrics[metric_name][metric_type]
                        if datetime.fromisoformat(v.timestamp) > cutoff_time
                    ]
                
                # Remove empty metric types
                self.metrics[metric_name] = {
                    k: v for k, v in self.metrics[metric_name].items()
                    if v
                }
            
            # Remove empty metrics
            self.metrics = {
                k: v for k, v in self.metrics.items()
                if v
            }
            
            self._save_metrics()

    def timed(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Decorator for timing function execution."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args, **kwargs) -> T:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.record_timing(name, duration, metadata)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.record_timing(name, duration, metadata)
                    raise
            return wrapper
        return decorator

    def export_metrics(self, filepath: Union[str, Path]):
        """Export metrics to a file."""
        filepath = Path(filepath)
        data = {
            metric_name: {
                metric_type: [asdict(v) for v in values]
                for metric_type, values in metric_data.items()
            }
            for metric_name, metric_data in self.metrics.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_metrics(self, filepath: Union[str, Path]):
        """Import metrics from a file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Metrics file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.metrics = defaultdict(lambda: defaultdict(list))
            for metric_name, metric_data in data.items():
                for metric_type, values in metric_data.items():
                    self.metrics[metric_name][metric_type] = [
                        MetricValue(**v)
                        for v in values
                    ]
            
        self._save_metrics() 