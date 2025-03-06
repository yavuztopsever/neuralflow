"""
Validation utilities for the LangGraph application.
This module provides functionality for input validation and schema checking.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List, Union, Type, Callable
from pathlib import Path
from datetime import datetime
from utils.common.text import TextProcessor
from utils.error.handlers import ErrorHandler
from config.manager import ConfigManager
from utils.logging.manager import LogManager

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass

class SchemaValidator:
    """Validates data against schemas."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the schema validator.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.log_manager = LogManager(self.config)
        self.text_processor = TextProcessor()
        self._schemas = {}
        self._initialize_schemas()
    
    def _initialize_schemas(self):
        """Initialize default schemas."""
        try:
            # Create default schemas
            self.add_schema('workflow', {
                'type': 'object',
                'required': ['id', 'steps'],
                'properties': {
                    'id': {'type': 'string'},
                    'steps': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'required': ['id', 'handler'],
                            'properties': {
                                'id': {'type': 'string'},
                                'handler': {'type': 'string'},
                                'inputs': {'type': 'array', 'items': {'type': 'string'}},
                                'outputs': {'type': 'array', 'items': {'type': 'string'}},
                                'metadata': {'type': 'object'}
                            }
                        }
                    }
                }
            })
            
            self.add_schema('state', {
                'type': 'object',
                'required': ['workflow_id', 'state_id', 'context'],
                'properties': {
                    'workflow_id': {'type': 'string'},
                    'state_id': {'type': 'string'},
                    'context': {'type': 'object'},
                    'metadata': {'type': 'object'},
                    'status': {'type': 'string'},
                    'results': {'type': 'object'},
                    'error': {'type': 'string'}
                }
            })
            
            self.add_schema('event', {
                'type': 'object',
                'required': ['id', 'type', 'workflow_id', 'data'],
                'properties': {
                    'id': {'type': 'string'},
                    'type': {'type': 'string'},
                    'workflow_id': {'type': 'string'},
                    'data': {'type': 'object'},
                    'metadata': {'type': 'object'},
                    'timestamp': {'type': 'string'},
                    'handled': {'type': 'boolean'}
                }
            })
            
            self.add_schema('metric', {
                'type': 'object',
                'required': ['id', 'type', 'workflow_id', 'value'],
                'properties': {
                    'id': {'type': 'string'},
                    'type': {'type': 'string'},
                    'workflow_id': {'type': 'string'},
                    'value': {},
                    'metadata': {'type': 'object'},
                    'timestamp': {'type': 'string'}
                }
            })
            
            logger.info("Initialized schemas")
        except Exception as e:
            logger.error(f"Failed to initialize schemas: {e}")
            raise
    
    def add_schema(self, schema_id: str, schema: Dict[str, Any]) -> None:
        """Add a schema.
        
        Args:
            schema_id: Unique identifier for the schema
            schema: Schema definition
            
        Raises:
            ValueError: If schema_id already exists
        """
        try:
            if schema_id in self._schemas:
                raise ValueError(f"Schema {schema_id} already exists")
            
            self._schemas[schema_id] = schema
            logger.info(f"Added schema {schema_id}")
        except Exception as e:
            logger.error(f"Failed to add schema {schema_id}: {e}")
            raise
    
    def get_schema(self, schema_id: str) -> Optional[Dict[str, Any]]:
        """Get a schema by ID.
        
        Args:
            schema_id: Schema ID
            
        Returns:
            Schema definition or None if not found
        """
        return self._schemas.get(schema_id)
    
    def validate(self, schema_id: str, data: Dict[str, Any]) -> bool:
        """Validate data against a schema.
        
        Args:
            schema_id: Schema ID
            data: Data to validate
            
        Returns:
            True if data is valid
            
        Raises:
            ValidationError: If data is invalid
        """
        try:
            schema = self.get_schema(schema_id)
            if not schema:
                raise ValidationError(f"Schema {schema_id} not found")
            
            # Check required fields
            required = schema.get('required', [])
            missing = [field for field in required if field not in data]
            if missing:
                raise ValidationError(f"Missing required fields: {missing}")
            
            # Check property types
            properties = schema.get('properties', {})
            for field, value in data.items():
                if field in properties:
                    field_schema = properties[field]
                    if not self._validate_type(value, field_schema.get('type')):
                        raise ValidationError(
                            f"Invalid type for field {field}: "
                            f"expected {field_schema.get('type')}, got {type(value).__name__}"
                        )
            
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise ValidationError(str(e))
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate a value's type.
        
        Args:
            value: Value to validate
            expected_type: Expected type
            
        Returns:
            True if value is of expected type
        """
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        if expected_type not in type_map:
            return True
        
        expected_types = type_map[expected_type]
        if isinstance(expected_types, tuple):
            return isinstance(value, expected_types)
        return isinstance(value, expected_types)

class InputValidator:
    """Validates input data."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the input validator.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.log_manager = LogManager(self.config)
        self.text_processor = TextProcessor()
        self.schema_validator = SchemaValidator(self.config)
    
    def validate_workflow(self, workflow: Dict[str, Any]) -> bool:
        """Validate workflow data.
        
        Args:
            workflow: Workflow data to validate
            
        Returns:
            True if workflow is valid
            
        Raises:
            ValidationError: If workflow is invalid
        """
        try:
            return self.schema_validator.validate('workflow', workflow)
        except Exception as e:
            logger.error(f"Workflow validation failed: {e}")
            raise ValidationError(str(e))
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate state data.
        
        Args:
            state: State data to validate
            
        Returns:
            True if state is valid
            
        Raises:
            ValidationError: If state is invalid
        """
        try:
            return self.schema_validator.validate('state', state)
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            raise ValidationError(str(e))
    
    def validate_event(self, event: Dict[str, Any]) -> bool:
        """Validate event data.
        
        Args:
            event: Event data to validate
            
        Returns:
            True if event is valid
            
        Raises:
            ValidationError: If event is invalid
        """
        try:
            return self.schema_validator.validate('event', event)
        except Exception as e:
            logger.error(f"Event validation failed: {e}")
            raise ValidationError(str(e))
    
    def validate_metric(self, metric: Dict[str, Any]) -> bool:
        """Validate metric data.
        
        Args:
            metric: Metric data to validate
            
        Returns:
            True if metric is valid
            
        Raises:
            ValidationError: If metric is invalid
        """
        try:
            return self.schema_validator.validate('metric', metric)
        except Exception as e:
            logger.error(f"Metric validation failed: {e}")
            raise ValidationError(str(e))
    
    def validate_input(self, input_data: Dict[str, Any],
                      required_fields: List[str],
                      field_types: Optional[Dict[str, Type]] = None) -> bool:
        """Validate input data.
        
        Args:
            input_data: Input data to validate
            required_fields: List of required fields
            field_types: Optional dictionary of field types
            
        Returns:
            True if input is valid
            
        Raises:
            ValidationError: If input is invalid
        """
        try:
            # Check required fields
            missing = [field for field in required_fields if field not in input_data]
            if missing:
                raise ValidationError(f"Missing required fields: {missing}")
            
            # Check field types if specified
            if field_types:
                for field, expected_type in field_types.items():
                    if field in input_data:
                        if not isinstance(input_data[field], expected_type):
                            raise ValidationError(
                                f"Invalid type for field {field}: "
                                f"expected {expected_type.__name__}, "
                                f"got {type(input_data[field]).__name__}"
                            )
            
            return True
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ValidationError(str(e))
    
    def validate_output(self, output_data: Dict[str, Any],
                       required_fields: List[str],
                       field_types: Optional[Dict[str, Type]] = None) -> bool:
        """Validate output data.
        
        Args:
            output_data: Output data to validate
            required_fields: List of required fields
            field_types: Optional dictionary of field types
            
        Returns:
            True if output is valid
            
        Raises:
            ValidationError: If output is invalid
        """
        try:
            # Check required fields
            missing = [field for field in required_fields if field not in output_data]
            if missing:
                raise ValidationError(f"Missing required fields: {missing}")
            
            # Check field types if specified
            if field_types:
                for field, expected_type in field_types.items():
                    if field in output_data:
                        if not isinstance(output_data[field], expected_type):
                            raise ValidationError(
                                f"Invalid type for field {field}: "
                                f"expected {expected_type.__name__}, "
                                f"got {type(output_data[field]).__name__}"
                            )
            
            return True
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            raise ValidationError(str(e)) 