"""
Unified processing system for NeuralFlow.
"""

from typing import Dict, Any, Optional, List, TypeVar, Generic
import logging
from datetime import datetime
from pydantic import BaseModel, Field

from .base_processor import ProcessingTool, ProcessingResult
from ...services.core.unified_service import UnifiedService
from ...utils.error.base_handler import BaseErrorHandler
from ...utils.logging.base_manager import BaseLogManager

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ProcessorConfig(BaseModel):
    """Processor configuration."""
    max_batch_size: int = 100
    timeout: float = 30.0
    retry_count: int = 3
    cache_enabled: bool = True
    parallel_processing: bool = True

class UnifiedProcessor(ProcessingTool):
    """Unified processor with integrated functionality."""
    
    def __init__(
        self,
        config: Optional[ProcessorConfig] = None,
        service: Optional[UnifiedService] = None
    ):
        """Initialize unified processor.
        
        Args:
            config: Optional processor configuration
            service: Optional unified service
        """
        super().__init__()
        self.config = config or ProcessorConfig()
        self.service = service or UnifiedService()
        self.error_handler = BaseErrorHandler()
        self.log_manager = BaseLogManager()
    
    async def process(
        self,
        data: Dict[str, Any],
        processor_type: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process data using appropriate processor.
        
        Args:
            data: Data to process
            processor_type: Type of processor to use
            context: Optional processing context
            
        Returns:
            ProcessingResult: Processing result
        """
        try:
            # Validate input
            if not self._validate_input(data):
                raise ValueError("Invalid input data")
            
            # Select processor
            processor = self._get_processor(processor_type)
            if not processor:
                raise ValueError(f"Processor not found: {processor_type}")
            
            # Process data
            result = await processor.process(data, context)
            
            # Validate output
            if not self._validate_output(result):
                raise ValueError("Invalid output format")
            
            return result
            
        except Exception as e:
            self.error_handler.handle_error(
                "PROCESSING_ERROR",
                f"Processing failed: {e}",
                details={
                    "processor_type": processor_type,
                    "data": data
                }
            )
            raise
    
    async def process_batch(
        self,
        batch: List[Dict[str, Any]],
        processor_type: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> List[ProcessingResult]:
        """Process a batch of data.
        
        Args:
            batch: List of data to process
            processor_type: Type of processor to use
            context: Optional processing context
            
        Returns:
            List[ProcessingResult]: List of processing results
        """
        try:
            # Validate batch size
            if len(batch) > self.config.max_batch_size:
                raise ValueError(f"Batch size exceeds maximum: {self.config.max_batch_size}")
            
            results = []
            for data in batch:
                result = await self.process(data, processor_type, context)
                results.append(result)
            
            return results
            
        except Exception as e:
            self.error_handler.handle_error(
                "BATCH_PROCESSING_ERROR",
                f"Batch processing failed: {e}",
                details={
                    "processor_type": processor_type,
                    "batch_size": len(batch)
                }
            )
            raise
    
    def _validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            bool: True if valid
        """
        return isinstance(data, dict) and "input" in data
    
    def _validate_output(self, result: ProcessingResult) -> bool:
        """Validate output format.
        
        Args:
            result: Processing result to validate
            
        Returns:
            bool: True if valid
        """
        return isinstance(result, ProcessingResult)
    
    def _get_processor(self, processor_type: str) -> Any:
        """Get processor by type.
        
        Args:
            processor_type: Type of processor
            
        Returns:
            Any: Processor instance
        """
        return self.service.get_provider(f"processor_{processor_type}")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            super().cleanup()
            if self.service:
                self.service.cleanup()
        except Exception as e:
            self.error_handler.handle_error(
                "CLEANUP_ERROR",
                f"Failed to clean up resources: {e}"
            )
