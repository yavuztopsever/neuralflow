"""
Simple workflow example demonstrating basic LangGraph functionality.
"""

import asyncio
from typing import Dict, Any

from src.core.graph import Workflow
from src.core.nodes import Node
from src.core.edges import Edge
from src.models.llm import BaseLLM
from src.models.embeddings import BaseEmbedding
from src.storage.vector import BaseVectorStore
from src.storage.cache import BaseCache

async def process_text(text: str) -> str:
    """Process input text using an LLM."""
    llm = BaseLLM({"model": "gpt-3.5-turbo"})
    return await llm.generate(f"Process this text: {text}")

async def generate_embeddings(text: str) -> list[float]:
    """Generate embeddings for text."""
    embedding_model = BaseEmbedding({"model": "text-embedding-ada-002"})
    return await embedding_model.embed(text)

async def store_embeddings(embeddings: list[float], metadata: Dict[str, Any]) -> str:
    """Store embeddings in vector store."""
    vector_store = BaseVectorStore({"type": "pinecone"})
    return await vector_store.add([embeddings], [metadata])

async def cache_result(key: str, value: Any) -> None:
    """Cache the result."""
    cache = BaseCache({"type": "redis"})
    await cache.set(key, value)

async def main():
    """Main function."""
    # Create nodes
    input_node = Node("input", "input", {})
    process_node = Node("process", "llm", {"model": "gpt-3.5-turbo"})
    embedding_node = Node("embedding", "embedding", {"model": "text-embedding-ada-002"})
    storage_node = Node("storage", "vector_store", {"type": "pinecone"})
    cache_node = Node("cache", "cache", {"type": "redis"})
    output_node = Node("output", "output", {})

    # Create edges
    edges = [
        Edge("edge1", "input", "process", "data", {}),
        Edge("edge2", "process", "embedding", "data", {}),
        Edge("edge3", "embedding", "storage", "data", {}),
        Edge("edge4", "storage", "cache", "data", {}),
        Edge("edge5", "cache", "output", "data", {})
    ]

    # Create workflow
    workflow = Workflow(
        "text_processing_workflow",
        "Process text, generate embeddings, and store results",
        [input_node, process_node, embedding_node, storage_node, cache_node, output_node],
        edges,
        {}
    )

    # Execute workflow
    result = await workflow.execute({
        "input": "Hello, this is a test of the LangGraph workflow system!"
    })
    
    print("Workflow execution completed!")
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 