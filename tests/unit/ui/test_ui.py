"""
Unit tests for UI functionality.
"""
import pytest
from src.ui import UI

@pytest.fixture
def ui():
    """Create a test UI instance."""
    return UI()

def test_ui_initialization(ui):
    """Test UI initialization."""
    assert ui is not None
    assert hasattr(ui, 'window')
    assert hasattr(ui, 'input_field')
    assert hasattr(ui, 'output_field')

def test_ui_input_handling(ui):
    """Test UI input handling."""
    test_input = "Test input"
    ui.set_input(test_input)
    assert ui.get_input() == test_input

def test_ui_output_display(ui):
    """Test UI output display."""
    test_output = "Test output"
    ui.display_output(test_output)
    assert ui.get_output() == test_output

def test_ui_clear_fields(ui):
    """Test clearing UI fields."""
    ui.set_input("Test input")
    ui.display_output("Test output")
    ui.clear_fields()
    assert ui.get_input() == ""
    assert ui.get_output() == ""

def test_ui_error_display(ui):
    """Test UI error display."""
    error_message = "Test error"
    ui.display_error(error_message)
    assert error_message in ui.get_output()

def test_ui_input_validation(ui):
    """Test UI input validation."""
    with pytest.raises(ValueError):
        ui.set_input(None)  # Invalid input type

def test_ui_output_formatting(ui):
    """Test UI output formatting."""
    test_output = "Test output"
    ui.display_output(test_output, format_type="markdown")
    assert ui.get_output() == test_output  # Should handle markdown formatting 