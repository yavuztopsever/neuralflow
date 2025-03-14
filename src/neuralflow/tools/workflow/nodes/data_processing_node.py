"""
Data processing workflow nodes using the unified processing system.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from ..base import WorkflowNode, NodeConfig
from ...data.core.unified_data import UnifiedData, DataConfig
from ...data.pipeline.unified_pipeline import UnifiedDataPipeline, PipelineConfig
from ...data.processors.unified_processor import ProcessedDataType
from ...monitoring.core.unified_monitor import UnifiedMonitor, MonitoringConfig
from ...utils.error.base_handler import BaseErrorHandler
from ...utils.logging.base_manager import BaseLogManager

logger = logging.getLogger(__name__)

class DataProcessingNodeConfig(NodeConfig):
    """Configuration for data processing node."""
    def __init__(
        self,
        node_id: str,
        data_config: Optional[DataConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        monitor_config: Optional[MonitoringConfig] = None,
        data_types: Optional[List[ProcessedDataType]] = None,
        batch_size: int = 100,
        parallel_processing: bool = True,
        cache_enabled: bool = True
    ):
        super().__init__(node_id)
        self.data_config = data_config or DataConfig()
        self.pipeline_config = pipeline_config or PipelineConfig(
            data_config=self.data_config,
            batch_size=batch_size,
            parallel_processing=parallel_processing,
            cache_enabled=cache_enabled
        )
        self.monitor_config = monitor_config or MonitoringConfig(
            monitor_id=f"node_{node_id}",
            monitor_type="data_processing"
        )
        self.data_types = data_types or [
            ProcessedDataType.EMBEDDING,
            ProcessedDataType.FINETUNING
        ]

class UnifiedDataProcessingNode(WorkflowNode):
    """Node for unified data processing."""
    
    def __init__(
        self,
        config: DataProcessingNodeConfig,
        error_handler: Optional[BaseErrorHandler] = None,
        log_manager: Optional[BaseLogManager] = None
    ):
        """Initialize the data processing node.
        
        Args:
            config: Node configuration
            error_handler: Optional error handler
            log_manager: Optional log manager
        """
        super().__init__(config.node_id)
        self.config = config
        self.error_handler = error_handler or BaseErrorHandler()
        self.log_manager = log_manager or BaseLogManager()
        
        # Initialize components
        self.data_system = UnifiedData(self.config.data_config)
        self.pipeline = UnifiedDataPipeline(
            config=self.config.pipeline_config,
            error_handler=self.error_handler,
            log_manager=self.log_manager
        )
        self.monitor = UnifiedMonitor(self.config.monitor_config)
    
    async def process(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process input data.
        
        Args:
            input_data: Input data to process
            context: Optional processing context
            
        Returns:
            Processing results
        """
        try:
            # Record start
            await self.monitor.record_event({
                "type": "processing_start",
                "node_id": self.config.node_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Process data
            if isinstance(input_data, list):
                results = await self.pipeline.process_batch(
                    batch_data=input_data,
                    data_types=self.config.data_types
                )
            else:
                results = await self.pipeline.process_batch(
                    batch_data=[input_data],
                    data_types=self.config.data_types
                )
            
            # Record completion
            await self.monitor.record_event({
                "type": "processing_complete",
                "node_id": self.config.node_id,
                "timestamp": datetime.now().isoformat(),
                "results_count": len(results)
            })
            
            return {
                "status": "success",
                "results": results[0] if not isinstance(input_data, list) else results
            }
            
        except Exception as e:
            # Record failure
            await self.monitor.record_event({
                "type": "processing_failure",
                "node_id": self.config.node_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            self.error_handler.handle_error(
                "NODE_PROCESSING_ERROR",
                f"Node processing failed: {e}",
                details={"node_id": self.config.node_id}
            )
            raise
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.data_system.cleanup()
            self.pipeline.cleanup()
            self.monitor.cleanup()
        except Exception as e:
            self.error_handler.handle_error(
                "CLEANUP_ERROR",
                f"Failed to clean up resources: {e}",
                details={"node_id": self.config.node_id}
            ) 