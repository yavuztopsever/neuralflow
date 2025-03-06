"""
Basic example demonstrating the workflow system.
"""

import asyncio
from typing import Dict, Any

from src.core.graph.nodes import Node
from src.core.graph.edges import Edge
from src.core.graph.workflows import Workflow
from src.models.llm.base import BaseLLM
from src.storage.cache.base import BaseCache

class SimpleLLM(BaseLLM):
    """Simple implementation of LLM for demonstration."""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        return f"Generated text for: {prompt}"
        
    async def embed(self, text: str) -> list[float]:
        """Generate embeddings for text."""
        return [0.1, 0.2, 0.3]  # Dummy embeddings

class SimpleCache(BaseCache):
    """Simple implementation of cache for demonstration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._cache: Dict[str, Any] = {}
        
    async def get(self, key: str) -> Any:
        return self._cache.get(key)
        
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        self._cache[key] = value
        
    async def delete(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
            
    async def clear(self) -> None:
        self._cache.clear()

async def main():
    """Main function demonstrating workflow usage."""
    # Create nodes
    input_node = Node("input", "input", {})
    llm_node = Node("llm", "llm", {"provider": "simple"})
    cache_node = Node("cache", "cache", {"provider": "simple"})
    output_node = Node("output", "output", {})
    
    # Create edges
    edges = [
        Edge("edge1", "input", "llm", "prompt", {}),
        Edge("edge2", "llm", "cache", "data", {}),
        Edge("edge3", "cache", "output", "results", {})
    ]
    
    # Create workflow
    workflow = Workflow(
        "simple_workflow",
        "Demonstrate basic workflow functionality",
        [input_node, llm_node, cache_node, output_node],
        edges,
        {}
    )
    
    # Execute workflow
    result = await workflow.execute({
        "input": "Hello, world!"
    })
    
    print("Workflow execution completed!")
    print(f"Results: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 