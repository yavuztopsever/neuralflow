"""
Unit tests for configuration system functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.config.manager import ConfigManager
from src.config.validator import ConfigValidator
from src.config.loader import ConfigLoader

class TestConfig:
    """Test suite for configuration system functionality."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a config manager for testing."""
        return ConfigManager()
    
    @pytest.fixture
    def config_validator(self):
        """Create a config validator for testing."""
        return ConfigValidator()
    
    @pytest.fixture
    def config_loader(self):
        """Create a config loader for testing."""
        return ConfigLoader()
    
    @pytest.mark.asyncio
    async def test_config_loading(self, config_loader):
        """Test configuration loading operations."""
        # Test loading configuration from file
        config_data = await config_loader.load_config("config.yaml")
        assert config_data is not None
        assert isinstance(config_data, dict)
        assert "model" in config_data
        assert "storage" in config_data
        
        # Test loading configuration from environment
        env_config = await config_loader.load_env_config()
        assert env_config is not None
        assert isinstance(env_config, dict)
    
    @pytest.mark.asyncio
    async def test_config_validation(self, config_validator):
        """Test configuration validation operations."""
        # Test validating configuration
        valid_config = {
            "model": {
                "name": "test_model",
                "temperature": 0.7,
                "max_tokens": 100
            },
            "storage": {
                "type": "vector",
                "dimension": 128
            }
        }
        
        is_valid = await config_validator.validate(valid_config)
        assert is_valid
        
        # Test invalid configuration
        invalid_config = {
            "model": {
                "name": "",  # Empty name
                "temperature": 2.0  # Invalid temperature
            }
        }
        
        is_valid = await config_validator.validate(invalid_config)
        assert not is_valid
    
    @pytest.mark.asyncio
    async def test_config_management(self, config_manager):
        """Test configuration management operations."""
        # Test setting configuration
        config_data = {
            "model": {
                "name": "test_model",
                "temperature": 0.7
            }
        }
        
        await config_manager.set_config(config_data)
        assert config_manager.get_config() == config_data
        
        # Test updating configuration
        update_data = {
            "model": {
                "temperature": 0.8
            }
        }
        
        await config_manager.update_config(update_data)
        assert config_manager.get_config()["model"]["temperature"] == 0.8
        
        # Test getting specific configuration
        model_config = await config_manager.get_section("model")
        assert model_config is not None
        assert model_config["name"] == "test_model"
        assert model_config["temperature"] == 0.8
    
    @pytest.mark.asyncio
    async def test_config_error_handling(self, config_manager, config_validator, config_loader):
        """Test configuration error handling."""
        # Test loading invalid configuration file
        with pytest.raises(FileNotFoundError):
            await config_loader.load_config("nonexistent.yaml")
        
        # Test validating invalid configuration
        with pytest.raises(ValueError):
            await config_validator.validate(None)
        
        # Test setting invalid configuration
        with pytest.raises(ValueError):
            await config_manager.set_config(None)
    
    @pytest.mark.asyncio
    async def test_config_metrics(self, config_manager):
        """Test configuration metrics collection."""
        # Test collecting configuration metrics
        metrics = await config_manager.collect_metrics()
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "total_sections" in metrics
        assert "last_update" in metrics
        assert "validation_status" in metrics
    
    @pytest.mark.asyncio
    async def test_config_optimization(self, config_manager):
        """Test configuration optimization operations."""
        # Test optimizing configuration
        optimization_params = {
            "cache_size": 1000,
            "validation_strictness": "high",
            "update_frequency": "real-time"
        }
        
        optimized_config = await config_manager.optimize(optimization_params)
        assert optimized_config is not None
        assert isinstance(optimized_config, dict)
        assert optimized_config["cache_size"] == 1000
        assert optimized_config["validation_strictness"] == "high"
        assert optimized_config["update_frequency"] == "real-time"
    
    @pytest.mark.asyncio
    async def test_config_persistence(self, config_manager):
        """Test configuration persistence operations."""
        # Test saving configuration
        config_data = {
            "model": {
                "name": "test_model",
                "temperature": 0.7
            }
        }
        
        await config_manager.set_config(config_data)
        await config_manager.save_config()
        
        # Test loading saved configuration
        loaded_config = await config_manager.load_config()
        assert loaded_config == config_data
        
        # Test configuration backup
        await config_manager.backup_config()
        backup_exists = await config_manager.check_backup()
        assert backup_exists 