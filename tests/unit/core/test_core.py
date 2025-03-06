"""
Unit tests for core functionality.
"""
import pytest
from src.core import Core
from src.config import Config

@pytest.fixture
def mock_config():
    return Config(
        model_name="test-model",
        temperature=0.7,
        max_tokens=100
    )

@pytest.fixture
def core(mock_config):
    return Core(mock_config)

def test_core_initialization(core, mock_config):
    """Test core module initialization."""
    assert core.config == mock_config
    assert core.model_name == mock_config.model_name
    assert core.temperature == mock_config.temperature
    assert core.max_tokens == mock_config.max_tokens

def test_core_process_input(core):
    """Test core input processing."""
    test_input = "Test input"
    processed = core.process_input(test_input)
    assert processed == test_input.strip()

def test_core_generate_response(core):
    """Test core response generation."""
    test_input = "Test input"
    response = core.generate_response(test_input)
    assert isinstance(response, str)
    assert len(response) > 0

def test_core_error_handling(core):
    """Test core error handling."""
    with pytest.raises(ValueError):
        core.process_input("")

def test_core_config_validation():
    """Test core configuration validation."""
    with pytest.raises(ValueError):
        Core(None)  # Should raise error for invalid config 