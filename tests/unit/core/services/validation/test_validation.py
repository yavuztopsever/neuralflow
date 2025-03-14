"""
Unit tests for validation service functionality.
"""
import pytest
from typing import Dict, Any, List
from datetime import datetime

from src.core.services.validation.validation_service import (
    ValidationService,
    ValidationRule,
    ValidationResult
)

class TestValidationService:
    """Test suite for validation service functionality."""
    
    @pytest.fixture
    def validation_service(self):
        """Create a validation service for testing."""
        return ValidationService()
    
    @pytest.fixture
    def sample_rules(self):
        """Create sample validation rules for testing."""
        return [
            ValidationRule(
                name="required_fields",
                condition=lambda data: all(
                    field in data for field in ["id", "name", "type"]
                ),
                message="Missing required fields"
            ),
            ValidationRule(
                name="valid_type",
                condition=lambda data: data["type"] in ["user", "admin", "guest"],
                message="Invalid user type"
            ),
            ValidationRule(
                name="valid_date",
                condition=lambda data: isinstance(data.get("created_at"), datetime),
                message="Invalid date format"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_validation_initialization(self, validation_service):
        """Test validation service initialization."""
        assert validation_service.rules == {}
        assert validation_service._validation_history == []
    
    @pytest.mark.asyncio
    async def test_rule_management(self, validation_service, sample_rules):
        """Test validation rule management."""
        # Add rules
        for rule in sample_rules:
            await validation_service.add_rule(rule)
        
        # Verify rules were added
        assert len(validation_service.rules) == len(sample_rules)
        assert all(rule.name in validation_service.rules for rule in sample_rules)
        
        # Remove rule
        await validation_service.remove_rule(sample_rules[0].name)
        assert sample_rules[0].name not in validation_service.rules
        
        # Update rule
        updated_rule = ValidationRule(
            name="required_fields",
            condition=lambda data: all(
                field in data for field in ["id", "name", "type", "email"]
            ),
            message="Missing required fields including email"
        )
        await validation_service.update_rule(updated_rule)
        assert validation_service.rules["required_fields"].message == "Missing required fields including email"
    
    @pytest.mark.asyncio
    async def test_data_validation(self, validation_service, sample_rules):
        """Test data validation functionality."""
        # Add rules
        for rule in sample_rules:
            await validation_service.add_rule(rule)
        
        # Test valid data
        valid_data = {
            "id": "123",
            "name": "Test User",
            "type": "user",
            "created_at": datetime.now()
        }
        result = await validation_service.validate_data(valid_data)
        assert result.is_valid
        assert len(result.errors) == 0
        
        # Test invalid data
        invalid_data = {
            "id": "123",
            "name": "Test User"
        }
        result = await validation_service.validate_data(invalid_data)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("Missing required fields" in error for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_batch_validation(self, validation_service, sample_rules):
        """Test batch data validation."""
        # Add rules
        for rule in sample_rules:
            await validation_service.add_rule(rule)
        
        # Test batch validation
        batch_data = [
            {
                "id": "1",
                "name": "User 1",
                "type": "user",
                "created_at": datetime.now()
            },
            {
                "id": "2",
                "name": "User 2",
                "type": "invalid",
                "created_at": datetime.now()
            }
        ]
        
        results = await validation_service.validate_batch(batch_data)
        assert len(results) == len(batch_data)
        assert results[0].is_valid
        assert not results[1].is_valid
        assert any("Invalid user type" in error for error in results[1].errors)
    
    @pytest.mark.asyncio
    async def test_validation_history(self, validation_service, sample_rules):
        """Test validation history tracking."""
        # Add rules
        for rule in sample_rules:
            await validation_service.add_rule(rule)
        
        # Perform validations
        test_data = {
            "id": "123",
            "name": "Test User",
            "type": "user",
            "created_at": datetime.now()
        }
        await validation_service.validate_data(test_data)
        
        # Check history
        assert len(validation_service._validation_history) > 0
        history_entry = validation_service._validation_history[-1]
        assert history_entry["timestamp"] is not None
        assert history_entry["data"] == test_data
        assert history_entry["result"].is_valid
    
    @pytest.mark.asyncio
    async def test_validation_metrics(self, validation_service, sample_rules):
        """Test validation metrics collection."""
        # Add rules
        for rule in sample_rules:
            await validation_service.add_rule(rule)
        
        # Perform validations
        test_data = {
            "id": "123",
            "name": "Test User",
            "type": "user",
            "created_at": datetime.now()
        }
        await validation_service.validate_data(test_data)
        
        # Get metrics
        metrics = await validation_service.get_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "total_validations" in metrics
        assert "success_rate" in metrics
        assert "rule_stats" in metrics
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, validation_service):
        """Test validation error handling."""
        # Test invalid rule
        with pytest.raises(ValueError):
            await validation_service.add_rule(None)
        
        # Test duplicate rule
        rule = ValidationRule(
            name="test_rule",
            condition=lambda x: True,
            message="Test rule"
        )
        await validation_service.add_rule(rule)
        with pytest.raises(ValueError):
            await validation_service.add_rule(rule)
        
        # Test invalid data
        with pytest.raises(ValueError):
            await validation_service.validate_data(None)
        
        # Test invalid batch
        with pytest.raises(ValueError):
            await validation_service.validate_batch(None) 