"""
Unified data system for NeuralFlow.
"""

from typing import Dict, Any, Optional, List, TypeVar, Generic
from datetime import datetime
import logging
from pydantic import BaseModel, Field

from .base_data import DataType
from ...utils.error.base_handler import BaseErrorHandler
from ...utils.logging.base_manager import BaseLogManager

logger = logging.getLogger(__name__)

T = TypeVar('T')

class DataConfig(BaseModel):
    """Data configuration."""
    chunk_size: int = 1000
    chunk_overlap: int = 100
    cache_enabled: bool = True
    parallel_processing: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB

class UnifiedData:
    """Unified data system with integrated functionality."""
    
    def __init__(
        self,
        config: Optional[DataConfig] = None,
        error_handler: Optional[BaseErrorHandler] = None,
        log_manager: Optional[BaseLogManager] = None
    ):
        """Initialize unified data system.
        
        Args:
            config: Optional data configuration
            error_handler: Optional error handler
            log_manager: Optional log manager
        """
        self.config = config or DataConfig()
        self.error_handler = error_handler or BaseErrorHandler()
        self.log_manager = log_manager or BaseLogManager()
        self.processors = {}
        self.validators = {}
    
    def register_processor(self, name: str, processor: Any) -> None:
        """Register a data processor.
        
        Args:
            name: Processor name
            processor: Processor instance
        """
        self.processors[name] = processor
    
    def register_validator(self, name: str, validator: Any) -> None:
        """Register a data validator.
        
        Args:
            name: Validator name
            validator: Validator instance
        """
        self.validators[name] = validator
    
    async def process_data(
        self,
        data: Dict[str, Any],
        processor: str = "default",
        data_type: Optional[DataType] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process data using specified processor.
        
        Args:
            data: Data to process
            processor: Processor to use
            data_type: Optional data type
            context: Optional processing context
            
        Returns:
            Processed data
        """
        if processor not in self.processors:
            raise ValueError(f"Processor not found: {processor}")
        
        try:
            # Validate input
            if data_type and not self._validate_data_type(data, data_type):
                raise ValueError(f"Invalid data type: {data_type}")
            
            # Process data
            result = await self.processors[processor].process(data, context)
            
            # Validate output
            if not self._validate_output(result):
                raise ValueError("Invalid output format")
            
            return result
        except Exception as e:
            self.error_handler.handle_error(
                "DATA_PROCESSING_ERROR",
                f"Data processing failed: {e}",
                details={"processor": processor, "data_type": data_type}
            )
            raise
    
    def _validate_data_type(self, data: Dict[str, Any], data_type: DataType) -> bool:
        """Validate data type.
        
        Args:
            data: Data to validate
            data_type: Expected data type
            
        Returns:
            True if valid
        """
        if data_type not in DataType:
            return False
        
        validator = self.validators.get(data_type.value)
        if not validator:
            return True
        
        return validator.validate(data)
    
    def _validate_output(self, result: Dict[str, Any]) -> bool:
        """Validate output format.
        
        Args:
            result: Processing result
            
        Returns:
            True if valid
        """
        return isinstance(result, dict) and "output" in result
    
    def cleanup(self):
        """Clean up resources."""
        try:
            for processor in self.processors.values():
                if hasattr(processor, "cleanup"):
                    processor.cleanup()
            for validator in self.validators.values():
                if hasattr(validator, "cleanup"):
                    validator.cleanup()
        except Exception as e:
            self.error_handler.handle_error(
                "CLEANUP_ERROR",
                f"Failed to clean up resources: {e}"
            )
