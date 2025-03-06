"""
Metrics management utilities for the LangGraph application.
This module provides functionality for collecting and reporting workflow metrics.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
from utils.common.text import TextProcessor
from utils.error.handlers import ErrorHandler
from config.manager import ConfigManager
from utils.logging.manager import LogManager

logger = logging.getLogger(__name__)

class WorkflowMetric:
    """Represents a workflow metric."""
    
    def __init__(self, metric_id: str,
                 metric_type: str,
                 workflow_id: str,
                 value: Any,
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a workflow metric.
        
        Args:
            metric_id: Unique identifier for the metric
            metric_type: Type of metric
            workflow_id: ID of the workflow
            value: Metric value
            metadata: Optional metadata for the metric
        """
        self.id = metric_id
        self.type = metric_type
        self.workflow_id = workflow_id
        self.value = value
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary.
        
        Returns:
            Dictionary representation of the metric
        """
        return {
            'id': self.id,
            'type': self.type,
            'workflow_id': self.workflow_id,
            'value': self.value,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowMetric':
        """Create metric from dictionary.
        
        Args:
            data: Dictionary representation of the metric
            
        Returns:
            WorkflowMetric instance
        """
        return cls(
            metric_id=data['id'],
            metric_type=data['type'],
            workflow_id=data['workflow_id'],
            value=data['value'],
            metadata=data.get('metadata')
        )

class MetricsCollector:
    """Collects workflow metrics."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the metrics collector.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.log_manager = LogManager(self.config)
        self.text_processor = TextProcessor()
        self._metrics = []
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize metrics storage."""
        try:
            # Create storage directory
            storage_dir = Path(self.config.get('storage_dir', 'storage'))
            metrics_dir = storage_dir / 'metrics'
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing metrics
            self._load_metrics(metrics_dir)
            
            logger.info("Initialized metrics storage")
        except Exception as e:
            logger.error(f"Failed to initialize metrics storage: {e}")
            raise
    
    def _load_metrics(self, metrics_dir: Path) -> None:
        """Load metrics from storage.
        
        Args:
            metrics_dir: Directory containing metric files
        """
        try:
            for metric_file in metrics_dir.glob('*.json'):
                try:
                    with open(metric_file, 'r') as f:
                        data = json.load(f)
                        metric = WorkflowMetric.from_dict(data)
                        self._metrics.append(metric)
                except Exception as e:
                    logger.error(f"Failed to load metric from {metric_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
    
    def record_metric(self, metric_type: str,
                     workflow_id: str,
                     value: Any,
                     metadata: Optional[Dict[str, Any]] = None) -> WorkflowMetric:
        """Record a new metric.
        
        Args:
            metric_type: Type of metric
            workflow_id: ID of the workflow
            value: Metric value
            metadata: Optional metadata
            
        Returns:
            Created metric
        """
        try:
            # Generate metric ID
            metric_id = f"{metric_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create metric
            metric = WorkflowMetric(metric_id, metric_type, workflow_id, value, metadata)
            self._metrics.append(metric)
            
            # Save to storage
            self._save_metric(metric)
            
            logger.info(f"Recorded metric {metric_id}")
            return metric
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
            raise
    
    def get_metrics(self, workflow_id: Optional[str] = None,
                   metric_type: Optional[str] = None) -> List[WorkflowMetric]:
        """Get metrics matching criteria.
        
        Args:
            workflow_id: Optional workflow ID to filter by
            metric_type: Optional metric type to filter by
            
        Returns:
            List of matching metrics
        """
        metrics = self._metrics
        
        if workflow_id is not None:
            metrics = [m for m in metrics if m.workflow_id == workflow_id]
        
        if metric_type is not None:
            metrics = [m for m in metrics if m.type == metric_type]
        
        return metrics
    
    def _save_metric(self, metric: WorkflowMetric) -> None:
        """Save metric to storage.
        
        Args:
            metric: WorkflowMetric instance to save
        """
        try:
            storage_dir = Path(self.config.get('storage_dir', 'storage'))
            metric_file = storage_dir / 'metrics' / f"{metric.id}.json"
            
            with open(metric_file, 'w') as f:
                json.dump(metric.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metric {metric.id}: {e}")
            raise
    
    def get_metric_stats(self) -> Dict[str, Any]:
        """Get statistics about metrics.
        
        Returns:
            Dictionary containing metric statistics
        """
        try:
            return {
                'total_metrics': len(self._metrics),
                'workflows': len(set(m.workflow_id for m in self._metrics)),
                'metric_types': {
                    metric_type: len([m for m in self._metrics if m.type == metric_type])
                    for metric_type in set(m.type for m in self._metrics)
                }
            }
        except Exception as e:
            logger.error(f"Failed to get metric stats: {e}")
            return {}

class MetricsReporter:
    """Reports workflow metrics."""
    
    def __init__(self, collector: MetricsCollector,
                 config: Optional[ConfigManager] = None):
        """Initialize the metrics reporter.
        
        Args:
            collector: Metrics collector instance
            config: Configuration manager instance
        """
        self.collector = collector
        self.config = config or ConfigManager()
        self.log_manager = LogManager(self.config)
        self.text_processor = TextProcessor()
    
    def generate_report(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a metrics report.
        
        Args:
            workflow_id: Optional workflow ID to filter by
            
        Returns:
            Dictionary containing metrics report
        """
        try:
            # Get metrics
            metrics = self.collector.get_metrics(workflow_id)
            
            # Group metrics by type
            metrics_by_type = {}
            for metric in metrics:
                if metric.type not in metrics_by_type:
                    metrics_by_type[metric.type] = []
                metrics_by_type[metric.type].append(metric)
            
            # Calculate statistics
            report = {
                'timestamp': datetime.now().isoformat(),
                'workflow_id': workflow_id,
                'total_metrics': len(metrics),
                'metric_types': len(metrics_by_type),
                'metrics': {}
            }
            
            for metric_type, type_metrics in metrics_by_type.items():
                values = [m.value for m in type_metrics]
                report['metrics'][metric_type] = {
                    'count': len(values),
                    'values': values
                }
                
                # Calculate numeric statistics if applicable
                if all(isinstance(v, (int, float)) for v in values):
                    report['metrics'][metric_type].update({
                        'min': min(values),
                        'max': max(values),
                        'sum': sum(values),
                        'avg': sum(values) / len(values)
                    })
            
            return report
        except Exception as e:
            logger.error(f"Failed to generate metrics report: {e}")
            return {}
    
    def save_report(self, report: Dict[str, Any]) -> bool:
        """Save a metrics report.
        
        Args:
            report: Metrics report to save
            
        Returns:
            True if report was saved successfully
        """
        try:
            # Create reports directory
            storage_dir = Path(self.config.get('storage_dir', 'storage'))
            reports_dir = storage_dir / 'reports'
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate report filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            workflow_id = report.get('workflow_id', 'all')
            report_file = reports_dir / f"metrics_{workflow_id}_{timestamp}.json"
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved metrics report to {report_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save metrics report: {e}")
            return False
    
    def get_report_history(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get metrics report history.
        
        Args:
            workflow_id: Optional workflow ID to filter by
            
        Returns:
            List of historical reports
        """
        try:
            # Get reports directory
            storage_dir = Path(self.config.get('storage_dir', 'storage'))
            reports_dir = storage_dir / 'reports'
            
            if not reports_dir.exists():
                return []
            
            # Load reports
            reports = []
            for report_file in reports_dir.glob('metrics_*.json'):
                try:
                    with open(report_file, 'r') as f:
                        report = json.load(f)
                        if workflow_id is None or report.get('workflow_id') == workflow_id:
                            reports.append(report)
                except Exception as e:
                    logger.error(f"Failed to load report from {report_file}: {e}")
            
            # Sort by timestamp
            reports.sort(key=lambda r: r['timestamp'])
            
            return reports
        except Exception as e:
            logger.error(f"Failed to get report history: {e}")
            return [] 