"""
Unit tests for configuration functionality.
"""
import pytest
from src.config import Config

def test_config_initialization():
    """Test config initialization with valid parameters."""
    config = Config(
        model_name="test-model",
        temperature=0.7,
        max_tokens=100
    )
    assert config.model_name == "test-model"
    assert config.temperature == 0.7
    assert config.max_tokens == 100

def test_config_default_values():
    """Test config initialization with default values."""
    config = Config()
    assert config.model_name == "gpt-3.5-turbo"  # Default model
    assert config.temperature == 0.7  # Default temperature
    assert config.max_tokens == 1000  # Default max tokens

def test_config_validation():
    """Test config parameter validation."""
    with pytest.raises(ValueError):
        Config(temperature=2.0)  # Invalid temperature
    
    with pytest.raises(ValueError):
        Config(max_tokens=-1)  # Invalid max tokens
    
    with pytest.raises(ValueError):
        Config(model_name="")  # Invalid model name

def test_config_update():
    """Test config update functionality."""
    config = Config()
    config.update(temperature=0.8)
    assert config.temperature == 0.8

def test_config_to_dict():
    """Test config conversion to dictionary."""
    config = Config(
        model_name="test-model",
        temperature=0.7,
        max_tokens=100
    )
    config_dict = config.to_dict()
    assert config_dict["model_name"] == "test-model"
    assert config_dict["temperature"] == 0.7
    assert config_dict["max_tokens"] == 100 