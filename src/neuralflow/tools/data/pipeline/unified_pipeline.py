"""
Unified data science pipeline for NeuralFlow.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ..core.unified_data import UnifiedData, DataConfig
from ..processors.unified_processor import UnifiedDataProcessor, ProcessedDataType
from ...monitoring.core.unified_monitor import UnifiedMonitor, MonitoringConfig
from ...utils.error.base_handler import BaseErrorHandler
from ...utils.logging.base_manager import BaseLogManager

logger = logging.getLogger(__name__)

class PipelineConfig:
    """Pipeline configuration."""
    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        monitor_config: Optional[MonitoringConfig] = None,
        batch_size: int = 100,
        parallel_processing: bool = True,
        cache_enabled: bool = True
    ):
        self.data_config = data_config or DataConfig()
        self.monitor_config = monitor_config or MonitoringConfig(
            monitor_id="pipeline",
            monitor_type="data_pipeline"
        )
        self.batch_size = batch_size
        self.parallel_processing = parallel_processing
        self.cache_enabled = cache_enabled

class UnifiedDataPipeline:
    """Unified data science pipeline with integrated processing."""
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        error_handler: Optional[BaseErrorHandler] = None,
        log_manager: Optional[BaseLogManager] = None
    ):
        """Initialize the pipeline.
        
        Args:
            config: Optional pipeline configuration
            error_handler: Optional error handler
            log_manager: Optional log manager
        """
        self.config = config or PipelineConfig()
        self.error_handler = error_handler or BaseErrorHandler()
        self.log_manager = log_manager or BaseLogManager()
        
        # Initialize components
        self.data_system = UnifiedData(self.config.data_config)
        self.processor = UnifiedDataProcessor(
            error_handler=self.error_handler,
            log_manager=self.log_manager
        )
        self.monitor = UnifiedMonitor(self.config.monitor_config)
    
    async def process_batch(
        self,
        batch_data: List[Dict[str, Any]],
        data_types: Optional[List[ProcessedDataType]] = None
    ) -> List[Dict[str, Any]]:
        """Process a batch of data.
        
        Args:
            batch_data: List of data to process
            data_types: Optional list of data types to process
            
        Returns:
            List of processed data
        """
        try:
            results = []
            
            # Process each item
            for data in batch_data:
                try:
                    # Process data
                    processed = await self.processor.process_data(
                        data=data,
                        data_types=data_types
                    )
                    
                    results.append(processed)
                    
                    # Record success
                    await self.monitor.record_event({
                        "type": "processing_success",
                        "data_id": data.get("id"),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    # Record failure
                    await self.monitor.record_event({
                        "type": "processing_failure",
                        "data_id": data.get("id"),
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Handle error
                    self.error_handler.handle_error(
                        "PROCESSING_ERROR",
                        f"Failed to process data: {e}",
                        details={"data_id": data.get("id")}
                    )
            
            return results
            
        except Exception as e:
            self.error_handler.handle_error(
                "BATCH_PROCESSING_ERROR",
                f"Failed to process batch: {e}"
            )
            raise
    
    async def process_dataset(
        self,
        dataset_path: str,
        data_types: Optional[List[ProcessedDataType]] = None
    ) -> Dict[str, Any]:
        """Process an entire dataset.
        
        Args:
            dataset_path: Path to the dataset
            data_types: Optional list of data types to process
            
        Returns:
            Processing results and statistics
        """
        try:
            # Load dataset
            dataset = await self.data_system.load_dataset(dataset_path)
            
            # Process in batches
            results = []
            total_items = len(dataset)
            processed_items = 0
            
            for i in range(0, total_items, self.config.batch_size):
                batch = dataset[i:i + self.config.batch_size]
                batch_results = await self.process_batch(batch, data_types)
                results.extend(batch_results)
                processed_items += len(batch_results)
                
                # Record progress
                await self.monitor.record_metric({
                    "type": "processing_progress",
                    "total_items": total_items,
                    "processed_items": processed_items,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Compile statistics
            stats = {
                "total_items": total_items,
                "processed_items": processed_items,
                "success_rate": processed_items / total_items if total_items > 0 else 0,
                "processing_types": [dt.value for dt in (data_types or [])]
            }
            
            return {
                "results": results,
                "statistics": stats
            }
            
        except Exception as e:
            self.error_handler.handle_error(
                "DATASET_PROCESSING_ERROR",
                f"Failed to process dataset: {e}",
                details={"dataset_path": dataset_path}
            )
            raise
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.data_system.cleanup()
            self.processor.cleanup()
            self.monitor.cleanup()
        except Exception as e:
            self.error_handler.handle_error(
                "CLEANUP_ERROR",
                f"Failed to clean up resources: {e}"
            )
