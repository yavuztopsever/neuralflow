"""
Tools package for the LangGraph project.
This package provides unified tool capabilities for search, memory management, and task processing.

The tools package is organized into the following submodules:

Base Tools:
    - BaseTool: Abstract base class for all tools
    - ToolMetadata: Model for tool metadata

Search Tools:
    - UnifiedSearch: Combined vector and graph search capabilities
    - SearchResult: Model for search results

Memory Tools:
    - MemoryTool: Unified memory management for conversation and vector memory
    - MemoryEntry: Model for memory entries

Processing Tools:
    - ProcessingTool: Unified task processing with LLM agent integration
    - ProcessingResult: Model for processing results

Example:
    ```python
    from src.core.tools import UnifiedSearch, MemoryTool, ProcessingTool

    # Initialize search tool
    search_tool = UnifiedSearch(
        vector_store=my_vector_store,
        embeddings=my_embeddings
    )

    # Initialize memory tool
    memory_tool = MemoryTool(
        vector_store=my_vector_store,
        embeddings=my_embeddings
    )

    # Initialize processing tool
    processing_tool = ProcessingTool(
        llm=my_llm,
        tools=[search_tool, memory_tool]
    )

    # Execute tools
    search_results = await search_tool.execute(
        query="example query",
        search_type="hybrid"
    )

    memory_result = await memory_tool.execute(
        action="add",
        content="example content"
    )

    process_result = await processing_tool.execute(
        task={"input": "example task"}
    )
    ```
"""

from .base.base_tool import BaseTool, ToolMetadata
from .search.unified_search import UnifiedSearch, SearchResult
from .memory.memory_tool import MemoryTool, MemoryEntry
from .processing.processing_tool import ProcessingTool, ProcessingResult

__all__ = [
    # Base
    'BaseTool',
    'ToolMetadata',
    
    # Search
    'UnifiedSearch',
    'SearchResult',
    
    # Memory
    'MemoryTool',
    'MemoryEntry',
    
    # Processing
    'ProcessingTool',
    'ProcessingResult'
]
