"""
Unit tests for main application functionality.
"""
import pytest
from src.main import Application
from src.config import Config
from src.core import Core
from src.storage import Storage
from src.ui import UI

@pytest.fixture
def config():
    """Create a test configuration."""
    return Config(
        model_name="test-model",
        temperature=0.7,
        max_tokens=100
    )

@pytest.fixture
def app(config):
    """Create a test application instance."""
    return Application(config)

def test_app_initialization(app, config):
    """Test application initialization."""
    assert app.config == config
    assert isinstance(app.core, Core)
    assert isinstance(app.storage, Storage)
    assert isinstance(app.ui, UI)

def test_app_startup(app):
    """Test application startup."""
    app.startup()
    assert app.is_running

def test_app_shutdown(app):
    """Test application shutdown."""
    app.startup()
    app.shutdown()
    assert not app.is_running

def test_app_process_input(app):
    """Test application input processing."""
    test_input = "Test input"
    response = app.process_input(test_input)
    assert isinstance(response, str)
    assert len(response) > 0

def test_app_save_conversation(app):
    """Test saving conversation history."""
    test_input = "Test input"
    test_response = "Test response"
    app.save_conversation(test_input, test_response)
    history = app.load_conversation_history()
    assert len(history) > 0
    assert history[-1]["input"] == test_input
    assert history[-1]["response"] == test_response

def test_app_error_handling(app):
    """Test application error handling."""
    with pytest.raises(ValueError):
        app.process_input("")  # Empty input should raise error

def test_app_config_update(app):
    """Test application configuration update."""
    new_temp = 0.8
    app.update_config(temperature=new_temp)
    assert app.config.temperature == new_temp 