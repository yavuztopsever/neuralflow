"""
Example demonstrating caching functionality.
"""

import asyncio
from typing import Dict, Any
from datetime import timedelta

from langgraph.src.core.graph import Workflow
from langgraph.src.core.nodes import Node
from langgraph.src.core.edges import Edge
from langgraph.src.models.llm import BaseLLM
from langgraph.src.storage.cache import BaseCache

async def generate_text(prompt: str) -> str:
    """Generate text using an LLM."""
    llm = BaseLLM({
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.7
    })
    return await llm.generate(prompt)

async def cache_with_redis(key: str, value: Any, ttl: timedelta = timedelta(hours=1)) -> None:
    """Cache value using Redis."""
    cache = BaseCache({
        "provider": "redis",
        "host": "localhost",
        "port": 6379
    })
    await cache.set(key, value, ttl)

async def cache_with_memcached(key: str, value: Any, ttl: timedelta = timedelta(hours=1)) -> None:
    """Cache value using Memcached."""
    cache = BaseCache({
        "provider": "memcached",
        "host": "localhost",
        "port": 11211
    })
    await cache.set(key, value, ttl)

async def cache_with_memory(key: str, value: Any, ttl: timedelta = timedelta(hours=1)) -> None:
    """Cache value using in-memory storage."""
    cache = BaseCache({
        "provider": "memory",
        "max_size": 1000
    })
    await cache.set(key, value, ttl)

async def get_from_cache(key: str, provider: str) -> Any:
    """Get value from cache."""
    cache = BaseCache({
        "provider": provider,
        "host": "localhost",
        "port": 6379 if provider == "redis" else 11211
    })
    return await cache.get(key)

async def main():
    """Main function."""
    # Create nodes
    input_node = Node("input", "input", {})
    llm_node = Node("llm", "llm", {"provider": "openai"})
    redis_node = Node("redis", "cache", {"provider": "redis"})
    memcached_node = Node("memcached", "cache", {"provider": "memcached"})
    memory_node = Node("memory", "cache", {"provider": "memory"})
    retrieve_node = Node("retrieve", "cache", {})
    output_node = Node("output", "output", {})

    # Create edges
    edges = [
        Edge("edge1", "input", "llm", "prompt", {}),
        Edge("edge2", "llm", "redis", "data", {}),
        Edge("edge3", "llm", "memcached", "data", {}),
        Edge("edge4", "llm", "memory", "data", {}),
        Edge("edge5", "input", "retrieve", "key", {}),
        Edge("edge6", "retrieve", "output", "results", {})
    ]

    # Create workflow
    workflow = Workflow(
        "caching_workflow",
        "Demonstrate caching with different providers",
        [input_node, llm_node, redis_node, memcached_node, memory_node, retrieve_node, output_node],
        edges,
        {}
    )

    # Execute workflow
    result = await workflow.execute({
        "input": "Write a haiku about programming",
        "key": "haiku_cache_key"
    })
    
    print("Workflow execution completed!")
    print(f"Cached results:\n{result}")

if __name__ == "__main__":
    asyncio.run(main()) 