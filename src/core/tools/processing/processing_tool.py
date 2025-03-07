"""
Processing tool for the LangGraph project.
This module provides unified processing capabilities for tasks and documents.

The processing tool provides:
1. Task Processing: Single and batch task processing
2. LLM Integration: Language model integration for task processing
3. Tool Management: Dynamic tool addition and removal
4. Memory Management: Conversation memory for context
5. State Management: Tool state persistence

Features:
- Task validation and processing
- LLM agent integration
- Tool usage tracking
- Memory management
- State persistence
- Error handling

Example:
    ```python
    from src.core.tools.processing import ProcessingTool
    
    # Initialize processing tool
    processor = ProcessingTool(
        llm=my_llm,
        tools=[tool1, tool2],
        memory=my_memory
    )
    
    # Process single task
    result = await processor.execute(
        task={"input": "process this task"},
        task_type="process"
    )
    
    # Process batch of tasks
    results = await processor.execute(
        task=[
            {"input": "task 1"},
            {"input": "task 2"}
        ],
        task_type="batch"
    )
    
    # Add new tool
    processor.add_tool(new_tool)
    
    # Get usage stats
    stats = processor.get_tool_usage_stats()
    ```
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from ..base.base_tool import BaseTool
from langchain.llms import BaseLLM
from langchain.tools import Tool
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
import json

class ProcessingResult(BaseModel):
    """
    Processing result model.
    
    Attributes:
        success: Whether processing was successful
        output: Processing output (if successful)
        error: Error message (if failed)
        metadata: Optional result metadata
        timestamp: Processing timestamp
    """
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ProcessingTool(BaseTool):
    """
    Tool for unified processing capabilities.
    
    This tool provides:
    1. Task processing with LLM integration
    2. Tool management and usage tracking
    3. Memory management
    4. State persistence
    5. Error handling
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        tools: Optional[List[Tool]] = None,
        memory: Optional[ConversationBufferMemory] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the processing tool.
        
        The initialization process:
        1. Sets up LLM for processing
        2. Initializes available tools
        3. Sets up memory management
        4. Creates agent executor if LLM is provided
        5. Initializes usage tracking
        
        Args:
            llm: Language model instance
            tools: List of available tools
            memory: Memory instance
            config: Optional configuration
        """
        super().__init__(
            name="processing_tool",
            description="Unified processing capabilities",
            config=config
        )
        
        self.llm = llm
        self.tools = tools or []
        self.memory = memory or ConversationBufferMemory()
        
        # Initialize agent executor if LLM and tools are provided
        self.agent_executor = None
        if self.llm and self.tools:
            self.agent_executor = AgentExecutor.from_llm_and_tools(
                llm=self.llm,
                tools=self.tools,
                memory=self.memory
            )
        
        # Initialize tool usage stats
        self.tool_usage: Dict[str, int] = {}
    
    async def execute(
        self,
        task: Dict[str, Any],
        task_type: str = "process",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Execute processing task.
        
        The execution process:
        1. Records tool usage
        2. Validates task type
        3. Processes task or batch
        4. Records results
        5. Handles errors
        
        Args:
            task: Task to process
            task_type: Type of task ('process' or 'batch')
            metadata: Optional metadata
            
        Returns:
            ProcessingResult: Processing result
            
        Raises:
            ValueError: If task_type is invalid
        """
        try:
            self.record_usage()
            
            if task_type == "process":
                return await self._process_task(task, metadata)
            elif task_type == "batch":
                return await self._process_batch(task, metadata)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
        except Exception as e:
            self.record_error(e)
            raise
    
    async def _process_task(
        self,
        task: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process a single task.
        
        The processing steps:
        1. Validates task format
        2. Uses agent executor if available
        3. Records tool usage
        4. Handles errors
        
        Args:
            task: Task to process
            metadata: Optional metadata
            
        Returns:
            ProcessingResult: Processing result
            
        Raises:
            ValueError: If task format is invalid
        """
        try:
            # Validate task
            if not isinstance(task, dict):
                raise ValueError("Task must be a dictionary")
            
            if "input" not in task:
                raise ValueError("Task must contain 'input' field")
            
            # Process task using agent executor if available
            if self.agent_executor:
                result = await self.agent_executor.arun(
                    input=task["input"],
                    metadata=metadata
                )
                
                # Record tool usage
                for tool in self.tools:
                    if tool.name in str(result):
                        self.tool_usage[tool.name] = self.tool_usage.get(tool.name, 0) + 1
                
                return ProcessingResult(
                    success=True,
                    output=result,
                    metadata=metadata
                )
            else:
                # Simple task processing without agent
                return ProcessingResult(
                    success=True,
                    output=task["input"],
                    metadata=metadata
                )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e),
                metadata=metadata
            )
    
    async def _process_batch(
        self,
        tasks: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple tasks.
        
        The batch processing steps:
        1. Validates tasks format
        2. Processes each task
        3. Collects results
        
        Args:
            tasks: List of tasks to process
            metadata: Optional metadata
            
        Returns:
            List[ProcessingResult]: Processing results
            
        Raises:
            ValueError: If tasks format is invalid
        """
        results = []
        for task in tasks:
            result = await self._process_task(task, metadata)
            results.append(result)
        return results
    
    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool.
        
        The tool addition process:
        1. Adds tool to available tools
        2. Reinitializes agent executor
        
        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
        
        # Reinitialize agent executor
        if self.llm:
            self.agent_executor = AgentExecutor.from_llm_and_tools(
                llm=self.llm,
                tools=self.tools,
                memory=self.memory
            )
    
    def remove_tool(self, tool_name: str) -> None:
        """
        Remove a tool.
        
        The tool removal process:
        1. Removes tool from available tools
        2. Reinitializes agent executor
        
        Args:
            tool_name: Name of tool to remove
        """
        self.tools = [t for t in self.tools if t.name != tool_name]
        
        # Reinitialize agent executor
        if self.llm:
            self.agent_executor = AgentExecutor.from_llm_and_tools(
                llm=self.llm,
                tools=self.tools,
                memory=self.memory
            )
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """
        Get tool usage statistics.
        
        Returns a dictionary containing:
        1. Total usage count
        2. Per-tool usage counts
        3. Number of available tools
        
        Returns:
            Dict[str, Any]: Tool usage statistics
        """
        return {
            "total_usage": sum(self.tool_usage.values()),
            "tool_usage": self.tool_usage,
            "available_tools": len(self.tools)
        }
    
    def reset_tool_stats(self) -> None:
        """
        Reset tool usage statistics.
        
        This method:
        1. Clears all tool usage counts
        """
        self.tool_usage = {}
    
    def save_state(self, filepath: str) -> None:
        """
        Save tool state.
        
        The state saving process:
        1. Collects tool usage stats
        2. Collects memory state
        3. Saves to file
        
        Args:
            filepath: Path to save state
            
        Raises:
            IOError: If file cannot be written
        """
        state = {
            "tool_usage": self.tool_usage,
            "memory": self.memory.load_memory_variables({})
        }
        
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, filepath: str) -> None:
        """
        Load tool state.
        
        The state loading process:
        1. Loads state from file
        2. Restores tool usage stats
        3. Restores memory state
        
        Args:
            filepath: Path to load state from
            
        Raises:
            IOError: If file cannot be read
            ValueError: If state format is invalid
        """
        with open(filepath, "r") as f:
            state = json.load(f)
            self.tool_usage = state["tool_usage"]
            
            # Restore memory
            self.memory.clear()
            for msg in state["memory"].get("history", []):
                self.memory.save_context(
                    {"input": msg["input"]},
                    {"output": msg["output"]}
                )

__all__ = ['ProcessingTool', 'ProcessingResult'] 