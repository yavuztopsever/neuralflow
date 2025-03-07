"""
Validation service for the LangGraph project.
This module provides validation capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging
from ...models.base_model import BaseNamedModel
from ...services.base_service import BaseService
from ...utils.common.text import TextProcessor
from ...utils.error.base_handler import BaseErrorHandler
from ...config.manager import ConfigManager
from ...utils.logging.base_manager import BaseLogManager

logger = logging.getLogger(__name__)

class ValidationRule(BaseNamedModel):
    """Validation rule model."""
    type: str
    parameters: Dict[str, Any]

class ValidationResult(BaseNamedModel):
    """Validation result model."""
    rule_name: str
    is_valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None

class ValidationService(BaseService[ValidationRule]):
    """Service for handling validations in the LangGraph system."""
    
    def __init__(self):
        """Initialize the validation service."""
        super().__init__()
        self.rules: Dict[str, ValidationRule] = {}
        self.results: List[ValidationResult] = []
        self.text_processor = TextProcessor()
        self.error_handler = BaseErrorHandler()
        self.config_manager = ConfigManager()
        self.log_manager = BaseLogManager()
    
    def add_rule(
        self,
        name: str,
        rule_type: str,
        parameters: Dict[str, Any],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationRule:
        """
        Add a validation rule.
        
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
            # Create rule
            rule = ValidationRule(
                name=name,
                type=rule_type,
                parameters=parameters,
                description=description,
                metadata=metadata
            )
            
            # Store rule
            self.rules[name] = rule
            
            # Record in history
            self.record_history(
                "add_rule",
                details={
                    "rule_name": name,
                    "rule_type": rule_type,
                    "rule_parameters": parameters,
                    "rule_description": description
                },
                metadata=metadata
            )
            
            # Log rule
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
        """
        Validate a value against a rule.
        
        Args:
            rule_name: Rule name
            value: Value to validate
            context: Optional validation context
            
        Returns:
            ValidationResult: Validation result
        """
        try:
            # Get rule
            rule = self.rules.get(rule_name)
            if not rule:
                raise ValueError(f"Validation rule not found: {rule_name}")
            
            # Validate based on rule type
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
            
            # Create result
            result = ValidationResult(
                name=rule_name,
                rule_name=rule_name,
                is_valid=is_valid,
                message=message,
                details=details
            )
            
            # Store result
            self.results.append(result)
            
            # Record in history
            self.record_history(
                "validate",
                details={
                    "rule_name": rule_name,
                    "is_valid": is_valid,
                    "message": message,
                    "value": value,
                    "context": context,
                    "details": details
                }
            )
            
            # Log result
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
        """
        Validate a value against all rules.
        
        Args:
            value: Value to validate
            context: Optional validation context
            
        Returns:
            List[ValidationResult]: List of validation results
        """
        results = []
        for rule_name in self.rules:
            result = self.validate(rule_name, value, context)
            results.append(result)
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get validation summary.
        
        Returns:
            Dict[str, Any]: Validation summary
        """
        return {
            "total_rules": len(self.rules),
            "total_results": len(self.results),
            "valid_results": len([r for r in self.results if r.is_valid]),
            "invalid_results": len([r for r in self.results if not r.is_valid]),
            "latest_result": self.results[-1] if self.results else None
        }
    
    def reset(self) -> None:
        """Reset validation service."""
        super().reset()
        self.rules = {}
        self.results = []
        self.log_manager.log("INFO", "Validation service reset")

__all__ = ['ValidationService', 'ValidationRule', 'ValidationResult'] 