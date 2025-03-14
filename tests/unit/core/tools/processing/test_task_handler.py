"""
Unit tests for task handler functionality.
"""
import pytest
import subprocess
import resource
import re
from typing import Dict, Any, List
import asyncio

from src.core.tools.processing.task_processor import TaskHandler
from src.core.tools.generation.generation_tool import GenerationTool

class TestTaskHandler:
    """Test suite for task handler functionality."""
    
    @pytest.fixture
    def task_handler(self):
        """Create a task handler for testing."""
        return TaskHandler()
    
    @pytest.fixture
    def generation_tool(self):
        """Create a generation tool for testing."""
        return GenerationTool()
    
    def test_code_validation(self, task_handler):
        """Test code validation functionality."""
        # Test valid Python code
        valid_python = """
def test_function():
    return "Hello, World!"
"""
        assert task_handler._validate_code(valid_python, "python", "default")
        
        # Test invalid Python code
        invalid_python = """
import os
os.system("rm -rf /")
"""
        assert not task_handler._validate_code(invalid_python, "python", "default")
        
        # Test valid JavaScript code
        valid_js = """
function testFunction() {
    return "Hello, World!";
}
"""
        assert task_handler._validate_code(valid_js, "javascript", "default")
        
        # Test invalid JavaScript code
        invalid_js = """
eval("console.log('test')");
"""
        assert not task_handler._validate_code(invalid_js, "javascript", "default")
        
        # Test valid bash code
        valid_bash = """
echo "Hello, World!"
"""
        assert task_handler._validate_code(valid_bash, "bash", "default")
        
        # Test invalid bash code
        invalid_bash = """
rm -rf /
"""
        assert not task_handler._validate_code(invalid_bash, "bash", "default")
    
    def test_resource_limits(self, task_handler):
        """Test resource limit setting."""
        # Test default limits
        task_handler._set_resource_limits("default")
        cpu_limit, _ = resource.getrlimit(resource.RLIMIT_CPU)
        memory_limit, _ = resource.getrlimit(resource.RLIMIT_AS)
        assert cpu_limit == 5
        assert memory_limit == 50 * 1024 * 1024  # Convert MB to bytes
        
        # Test high limits
        task_handler._set_resource_limits("high")
        cpu_limit, _ = resource.getrlimit(resource.RLIMIT_CPU)
        memory_limit, _ = resource.getrlimit(resource.RLIMIT_AS)
        assert cpu_limit == 10
        assert memory_limit == 100 * 1024 * 1024  # Convert MB to bytes
    
    def test_process_start(self, task_handler):
        """Test process starting functionality."""
        # Test Python process
        python_code = "print('Hello, World!')"
        process = task_handler._start_process(python_code, "python")
        assert process is not None
        stdout, stderr = process.communicate()
        assert stdout.decode().strip() == "Hello, World!"
        assert not stderr
        
        # Test JavaScript process
        js_code = "console.log('Hello, World!')"
        process = task_handler._start_process(js_code, "javascript")
        assert process is not None
        stdout, stderr = process.communicate()
        assert "Hello, World!" in stdout.decode()
        assert not stderr
        
        # Test bash process
        bash_code = "echo 'Hello, World!'"
        process = task_handler._start_process(bash_code, "bash")
        assert process is not None
        stdout, stderr = process.communicate()
        assert stdout.decode().strip() == "Hello, World!"
        assert not stderr
        
        # Test unsupported language
        process = task_handler._start_process("test", "unsupported")
        assert process is None
    
    @pytest.mark.asyncio
    async def test_generation_operations(self, generation_tool):
        """Test generation tool operations."""
        prompt = "Generate a test text"
        generated_text = await generation_tool.generate_text(prompt)
        assert generated_text is not None
        assert isinstance(generated_text, str)
        assert len(generated_text) > 0
        
        code_prompt = "Generate a test function"
        generated_code = await generation_tool.generate_code(code_prompt)
        assert generated_code is not None
        assert isinstance(generated_code, str)
        assert "def" in generated_code or "function" in generated_code
        
        image_prompt = "Generate a test image"
        generated_image = await generation_tool.generate_image(image_prompt)
        assert generated_image is not None
        assert isinstance(generated_image, bytes)
        assert len(generated_image) > 0
    
    @pytest.mark.asyncio
    async def test_generation_metrics(self, generation_tool):
        """Test generation tool metrics collection."""
        generation_metrics = await generation_tool.collect_metrics()
        assert generation_metrics is not None
        assert isinstance(generation_metrics, dict)
        assert "total_generations" in generation_metrics
        assert "average_generation_time" in generation_metrics
    
    def test_error_handling(self, task_handler, generation_tool):
        """Test error handling in task handler and generation tool."""
        # Test invalid code
        with pytest.raises(ValueError):
            task_handler._validate_code(None, "python", "default")
        
        # Test invalid language
        with pytest.raises(ValueError):
            task_handler._validate_code("test", None, "default")
        
        # Test invalid mode
        with pytest.raises(ValueError):
            task_handler._set_resource_limits(None)
        
        # Test invalid process
        with pytest.raises(ValueError):
            task_handler._start_process(None, "python")
        
        # Test invalid generation prompt
        with pytest.raises(ValueError):
            asyncio.run(generation_tool.generate_text(None))
    
    def test_process_cleanup(self, task_handler):
        """Test process cleanup functionality."""
        # Start a process
        process = task_handler._start_process("import time; time.sleep(1)", "python")
        assert process is not None
        
        # Cleanup process
        task_handler._cleanup_process(process)
        
        # Verify process is terminated
        assert process.poll() is not None
    
    def test_resource_monitoring(self, task_handler):
        """Test resource monitoring functionality."""
        # Start a process
        process = task_handler._start_process("import time; time.sleep(1)", "python")
        assert process is not None
        
        # Monitor resources
        resources = task_handler._monitor_resources(process)
        
        assert resources is not None
        assert "cpu_percent" in resources
        assert "memory_percent" in resources
        assert "io_counters" in resources
        
        # Cleanup
        task_handler._cleanup_process(process) 