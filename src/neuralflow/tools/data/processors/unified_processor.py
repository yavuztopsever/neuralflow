"""
Unified data processor for NeuralFlow.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import torch
from enum import Enum

from ..core.base import DataType
from ...utils.error.base_handler import BaseErrorHandler
from ...utils.logging.base_manager import BaseLogManager
from ...utils.db.base_manager import BaseDBManager

logger = logging.getLogger(__name__)

class ProcessedDataType(Enum):
    EMBEDDING = "embedding"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"

class UnifiedDataProcessor:
    """Processes data for different models using unified infrastructure."""
    
    def __init__(
        self,
        error_handler: Optional[BaseErrorHandler] = None,
        log_manager: Optional[BaseLogManager] = None,
        db_manager: Optional[BaseDBManager] = None
    ):
        """Initialize the processor.
        
        Args:
            error_handler: Optional error handler
            log_manager: Optional log manager
            db_manager: Optional database manager
        """
        self.error_handler = error_handler or BaseErrorHandler()
        self.log_manager = log_manager or BaseLogManager()
        self.db_manager = db_manager or BaseDBManager()
        self.model = None
        
    async def process_data(
        self,
        data: Dict[str, Any],
        data_types: Optional[List[ProcessedDataType]] = None
    ) -> Dict[str, Any]:
        """Process data for different purposes.
        
        Args:
            data: Raw data to process
            data_types: Optional list of data types to process
            
        Returns:
            Dictionary containing processed data for each type
        """
        if data_types is None:
            data_types = [
                ProcessedDataType.EMBEDDING,
                ProcessedDataType.FINETUNING
            ]
        
        processed_data = {}
        
        try:
            # Extract and validate data
            validated_data = await self._validate_data(data)
            
            # Process each data type
            for data_type in data_types:
                try:
                    if data_type == ProcessedDataType.EMBEDDING:
                        processed_data[data_type.value] = await self._process_embedding(validated_data)
                    elif data_type == ProcessedDataType.FINETUNING:
                        processed_data[data_type.value] = await self._process_finetuning(validated_data)
                    elif data_type == ProcessedDataType.EVALUATION:
                        processed_data[data_type.value] = await self._process_evaluation(validated_data)
                except Exception as e:
                    self.error_handler.handle_error(
                        "PROCESSING_ERROR",
                        f"Failed to process {data_type.value} data: {e}",
                        details={"data_type": data_type.value}
                    )
            
            # Validate processed data
            await self._validate_processed_data(processed_data)
            
            # Record processing metrics
            await self._record_metrics(processed_data)
            
            return processed_data
            
        except Exception as e:
            self.error_handler.handle_error(
                "PROCESSING_ERROR",
                f"Failed to process data: {e}"
            )
            raise
    
    async def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            Validated data
        """
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")
        
        required_fields = ["content", "metadata"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        return data
    
    async def _process_embedding(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data for embeddings.
        
        Args:
            data: Validated data
            
        Returns:
            Processed embedding data
        """
        # Implementation specific to embedding processing
        return {
            "embeddings": [],  # Placeholder for actual embeddings
            "metadata": data.get("metadata", {})
        }
    
    async def _process_finetuning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data for fine-tuning.
        
        Args:
            data: Validated data
            
        Returns:
            Processed fine-tuning data
        """
        # Implementation specific to fine-tuning processing
        return {
            "training_data": [],  # Placeholder for actual training data
            "metadata": data.get("metadata", {})
        }
    
    async def _process_evaluation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data for evaluation.
        
        Args:
            data: Validated data
            
        Returns:
            Processed evaluation data
        """
        # Implementation specific to evaluation processing
        return {
            "evaluation_data": [],  # Placeholder for actual evaluation data
            "metadata": data.get("metadata", {})
        }
    
    async def _validate_processed_data(self, processed_data: Dict[str, Any]) -> None:
        """Validate processed data.
        
        Args:
            processed_data: Data to validate
        """
        for data_type, data in processed_data.items():
            if not isinstance(data, dict):
                raise ValueError(f"Invalid processed data format for {data_type}")
    
    async def _record_metrics(self, processed_data: Dict[str, Any]) -> None:
        """Record processing metrics.
        
        Args:
            processed_data: Processed data
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "processed_types": list(processed_data.keys()),
            "total_items": len(processed_data)
        }
        
        self.log_manager.log_metrics("data_processing", metrics)
    
    async def store_processed_data(
        self,
        data_id: str,
        processed_data: Dict[str, Any],
        model_name: str
    ) -> None:
        """Store processed data.
        
        Args:
            data_id: Data identifier
            processed_data: Processed data
            model_name: Name of the model
        """
        try:
            for data_type, data in processed_data.items():
                await self.db_manager.store_data(
                    data_id=data_id,
                    data_type=data_type,
                    model_name=model_name,
                    data=data
                )
        except Exception as e:
            self.error_handler.handle_error(
                "STORAGE_ERROR",
                f"Failed to store processed data: {e}",
                details={"data_id": data_id}
            )
            raise
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.model:
                self.model.cpu()
                del self.model
                torch.cuda.empty_cache()
        except Exception as e:
            self.error_handler.handle_error(
                "CLEANUP_ERROR",
                f"Failed to clean up resources: {e}"
            ) 