"""
Unified data processing system for NeuralFlow.
"""

from typing import Dict, Any, List, Optional, Generic, TypeVar
from abc import ABC, abstractmethod
from datetime import datetime
import logging
from pydantic import BaseModel

from .base import BaseDataProcessor, ProcessedData, DataType, ValidationResult
from ..validators.data_validator import DataValidator
from ..processors.data_augmentor import DataAugmentor
from ...workflow.base import WorkflowConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')

class DataProcessorConfig(WorkflowConfig):
    """Configuration for data processors."""
    validation_rules: Dict[str, Any] = {}
    augmentation_config: Dict[str, Any] = {}
    cache_enabled: bool = True
    max_cache_size: int = 1000
    cleanup_interval: int = 3600

class UnifiedDataProcessor(BaseDataProcessor[T]):
    """Unified data processor with integrated validation and augmentation."""
    
    def __init__(
        self,
        config: Optional[DataProcessorConfig] = None,
        validator: Optional[DataValidator] = None,
        augmentor: Optional[DataAugmentor] = None
    ):
        """Initialize the processor.
        
        Args:
            config: Optional processor configuration
            validator: Optional data validator
            augmentor: Optional data augmentor
        """
        super().__init__()
        self.config = config or DataProcessorConfig()
        self.validator = validator or DataValidator()
        self.augmentor = augmentor or DataAugmentor()
        
    async def process_data(
        self,
        data: Dict[str, Any],
        data_types: Optional[List[DataType]] = None
    ) -> Dict[str, ProcessedData]:
        """Process data with validation and augmentation.
        
        Args:
            data: Raw data to process
            data_types: Optional list of data types to process
            
        Returns:
            Dictionary containing processed data for each type
        """
        if data_types is None:
            data_types = [DataType.EMBEDDING, DataType.FINETUNING]
        
        processed_data = {}
        
        try:
            # Validate input data
            validated_data = self._validate_input_data(data)
            
            # Process each data type
            for data_type in data_types:
                # Validate data
                validation_results = self.validator.validate_all(validated_data)
                
                # Clean and augment data if valid
                if all(result.is_valid for result in validation_results):
                    cleaned_data = self.validator.clean_data(validated_data)
                    augmented_data = self.augmentor.augment_data(cleaned_data)
                    processed = await self._process_data_type(data_type, augmented_data)
                else:
                    logger.warning(f"Data validation failed for type {data_type}")
                    continue
                
                processed_data[data_type.value] = ProcessedData(
                    data_type=data_type,
                    content=processed.content,
                    metadata=processed.metadata,
                    validation_results=validation_results
                )
            
            # Cache processed data
            if self.config.cache_enabled:
                await self._cache_processed_data(data.get("id"), processed_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        try:
            super().cleanup()
            self.validator.cleanup()
            self.augmentor.cleanup()
        except Exception as e:
            logger.error(f"Failed to clean up resources: {e}")
