"""
Base data system for NeuralFlow.
Provides unified data processing and validation capabilities.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Union, List, TypeVar, Generic, Callable
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from ..storage.base import BaseStorage, StorageConfig
from ..models.base_model import BaseNamedModel
from ..services.base_service import BaseService
from ..utils.common.text import TextProcessor
from ..utils.error.base_handler import BaseErrorHandler
from ..config.manager import ConfigManager
from ..utils.logging.base_manager import BaseLogManager

logger = logging.getLogger(__name__)

T = TypeVar('T')

class DataType(Enum):
    """Types of data that can be processed."""
    EMBEDDING = "embedding"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"
    RAW = "raw"

class ValidationRule(BaseNamedModel):
    """Validation rule model."""
    type: str
    parameters: Dict[str, Any]
    description: Optional[str] = None

class ValidationResult(BaseNamedModel):
    """Validation result model."""
    rule_name: str
    is_valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None

class ProcessedData(BaseModel):
    """Base model for processed data."""
    data_type: DataType
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    validation_results: List[ValidationResult] = Field(default_factory=list)

class BaseDataProcessor(ABC, Generic[T]):
    """Base class for all data processors with integrated validation."""
    
    def __init__(
        self,
        storage_config: Optional[StorageConfig] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the data processor.
        
        Args:
            storage_config: Optional storage configuration
            embedding_model: Name of the embedding model to use
            device: Device to run the model on
        """
        self.storage = BaseStorage(storage_config) if storage_config else None
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model).to(device)
        self.model.eval()
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.error_handler = BaseErrorHandler()
        self.config_manager = ConfigManager()
        self.log_manager = BaseLogManager()
        
        # Initialize validation rules
        self.validation_rules: Dict[str, ValidationRule] = {}
        
        # Create cache directory
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def add_validation_rule(
        self,
        name: str,
        rule_type: str,
        parameters: Dict[str, Any],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationRule:
        """Add a validation rule.
        
        Args:
            name: Rule name
            rule_type: Rule type
            parameters: Rule parameters
            description: Optional description
            metadata: Optional metadata
            
        Returns:
            ValidationRule: Created rule
        """
        try:
            rule = ValidationRule(
                name=name,
                type=rule_type,
                parameters=parameters,
                description=description,
                metadata=metadata
            )
            
            self.validation_rules[name] = rule
            
            self.log_manager.log(
                "INFO",
                f"Validation rule added: {name}",
                extra={
                    "rule_name": name,
                    "rule_type": rule_type,
                    "rule_parameters": parameters,
                    "rule_description": description,
                    "rule_metadata": metadata
                }
            )
            
            return rule
            
        except Exception as e:
            self.error_handler.handle_error(
                "VALIDATION_RULE_ERROR",
                f"Failed to add validation rule: {e}",
                details={"rule_name": name}
            )
            raise
    
    def validate(
        self,
        rule_name: str,
        value: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate a value against a rule.
        
        Args:
            rule_name: Rule name
            value: Value to validate
            context: Optional validation context
            
        Returns:
            ValidationResult: Validation result
        """
        try:
            rule = self.validation_rules.get(rule_name)
            if not rule:
                raise ValueError(f"Validation rule not found: {rule_name}")
            
            is_valid = True
            message = "Validation successful"
            details = {}
            
            if rule.type == "required":
                is_valid = value is not None
                message = "Value is required" if not is_valid else "Value is present"
            
            elif rule.type == "type":
                expected_type = rule.parameters.get("type")
                is_valid = isinstance(value, eval(expected_type))
                message = f"Value must be of type {expected_type}" if not is_valid else f"Value is of type {expected_type}"
            
            elif rule.type == "range":
                min_val = rule.parameters.get("min")
                max_val = rule.parameters.get("max")
                if min_val is not None and value < min_val:
                    is_valid = False
                    message = f"Value must be >= {min_val}"
                if max_val is not None and value > max_val:
                    is_valid = False
                    message = f"Value must be <= {max_val}"
            
            elif rule.type == "length":
                min_len = rule.parameters.get("min")
                max_len = rule.parameters.get("max")
                length = len(value)
                if min_len is not None and length < min_len:
                    is_valid = False
                    message = f"Length must be >= {min_len}"
                if max_len is not None and length > max_len:
                    is_valid = False
                    message = f"Length must be <= {max_len}"
            
            elif rule.type == "pattern":
                pattern = rule.parameters.get("pattern")
                import re
                is_valid = bool(re.match(pattern, str(value)))
                message = f"Value must match pattern {pattern}" if not is_valid else f"Value matches pattern {pattern}"
            
            elif rule.type == "enum":
                allowed_values = rule.parameters.get("values", [])
                is_valid = value in allowed_values
                message = f"Value must be one of {allowed_values}" if not is_valid else f"Value is one of {allowed_values}"
            
            elif rule.type == "custom":
                validator_func = rule.parameters.get("validator")
                if validator_func:
                    is_valid, message, details = validator_func(value, context)
            
            result = ValidationResult(
                name=rule_name,
                rule_name=rule_name,
                is_valid=is_valid,
                message=message,
                details=details
            )
            
            self.log_manager.log(
                "INFO" if is_valid else "WARNING",
                f"Validation result for rule {rule_name}: {message}",
                extra={
                    "rule_name": rule_name,
                    "is_valid": is_valid,
                    "value": value,
                    "context": context,
                    "details": details
                }
            )
            
            return result
            
        except Exception as e:
            self.error_handler.handle_error(
                "VALIDATION_ERROR",
                f"Failed to validate value: {e}",
                details={
                    "rule_name": rule_name,
                    "value": value,
                    "context": context
                }
            )
            raise
    
    def validate_all(
        self,
        value: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate a value against all rules.
        
        Args:
            value: Value to validate
            context: Optional validation context
            
        Returns:
            List[ValidationResult]: List of validation results
        """
        results = []
        for rule_name in self.validation_rules:
            results.append(self.validate(rule_name, value, context))
        return results
    
    async def process_data(
        self,
        data: Dict[str, Any],
        data_types: Optional[List[DataType]] = None
    ) -> Dict[str, ProcessedData]:
        """Process data into specified types with validation.
        
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
            # Extract and validate data
            validated_data = self._validate_input_data(data)
            
            # Process each data type
            for data_type in data_types:
                processed = await self._process_data_type(data_type, validated_data)
                processed_data[data_type.value] = processed
            
            # Cache processed data if storage is configured
            if self.storage:
                await self._cache_processed_data(data.get("id"), processed_data)
            
            return processed_data
            
        except Exception as e:
            self.error_handler.handle_error(
                "DATA_PROCESSING_ERROR",
                f"Failed to process data: {e}",
                details={"data_types": data_types}
            )
            raise
    
    @abstractmethod
    async def _process_data_type(
        self,
        data_type: DataType,
        data: Dict[str, Any]
    ) -> ProcessedData:
        """Process data for a specific type.
        
        Args:
            data_type: Type of data to process
            data: Validated data to process
            
        Returns:
            ProcessedData: Processed data with validation results
        """
        pass
    
    @abstractmethod
    def _validate_input_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data.
        
        Args:
            data: Raw input data
            
        Returns:
            Dict[str, Any]: Validated data
        """
        pass
    
    async def _cache_processed_data(
        self,
        data_id: str,
        processed_data: Dict[str, ProcessedData]
    ) -> None:
        """Cache processed data.
        
        Args:
            data_id: ID of the data
            processed_data: Processed data to cache
        """
        try:
            if not self.storage:
                return
                
            cache_data = {
                "id": data_id,
                "timestamp": datetime.now().isoformat(),
                "data": {
                    k: v.dict() for k, v in processed_data.items()
                }
            }
            
            await self.storage.store(
                f"cache/{data_id}",
                cache_data,
                metadata={"type": "processed_data_cache"}
            )
            
        except Exception as e:
            self.error_handler.handle_error(
                "CACHE_ERROR",
                f"Failed to cache processed data: {e}",
                details={"data_id": data_id}
            )
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.storage:
                self.storage.cleanup()
            if hasattr(self, "model"):
                self.model.cpu()
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            self.error_handler.handle_error(
                "CLEANUP_ERROR",
                f"Failed to clean up resources: {e}"
            ) 