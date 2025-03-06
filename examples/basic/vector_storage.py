"""
Example demonstrating vector storage integration.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime

from src.core.graph import Workflow
from src.core.nodes import Node
from src.core.edges import Edge
from src.models.embeddings import BaseEmbedding
from src.storage.vector import BaseVectorStore

async def generate_embeddings(text: str) -> list[float]:
    """Generate embeddings for text."""
    embedding_model = BaseEmbedding({
        "provider": "openai",
        "model": "text-embedding-ada-002"
    })
    return await embedding_model.embed(text)

async def store_in_pinecone(embeddings: list[float], metadata: Dict[str, Any]) -> str:
    """Store embeddings in Pinecone."""
    vector_store = BaseVectorStore({
        "provider": "pinecone",
        "index_name": "test-index"
    })
    return await vector_store.add([embeddings], [metadata])

async def store_in_weaviate(embeddings: list[float], metadata: Dict[str, Any]) -> str:
    """Store embeddings in Weaviate."""
    vector_store = BaseVectorStore({
        "provider": "weaviate",
        "class_name": "TestClass"
    })
    return await vector_store.add([embeddings], [metadata])

async def store_in_milvus(embeddings: list[float], metadata: Dict[str, Any]) -> str:
    """Store embeddings in Milvus."""
    vector_store = BaseVectorStore({
        "provider": "milvus",
        "collection_name": "test_collection"
    })
    return await vector_store.add([embeddings], [metadata])

async def search_similar(query: str, provider: str) -> List[Dict[str, Any]]:
    """Search for similar vectors."""
    vector_store = BaseVectorStore({
        "provider": provider,
        "index_name": "test-index" if provider == "pinecone" else None,
        "class_name": "TestClass" if provider == "weaviate" else None,
        "collection_name": "test_collection" if provider == "milvus" else None
    })
    
    embedding_model = BaseEmbedding({
        "provider": "openai",
        "model": "text-embedding-ada-002"
    })
    query_embedding = await embedding_model.embed(query)
    
    return await vector_store.search(query_embedding, limit=5)

async def main():
    """Main function."""
    # Create nodes
    input_node = Node("input", "input", {})
    embedding_node = Node("embedding", "embedding", {"provider": "openai"})
    pinecone_node = Node("pinecone", "vector_store", {"provider": "pinecone"})
    weaviate_node = Node("weaviate", "vector_store", {"provider": "weaviate"})
    milvus_node = Node("milvus", "vector_store", {"provider": "milvus"})
    search_node = Node("search", "search", {})
    output_node = Node("output", "output", {})

    # Create edges
    edges = [
        Edge("edge1", "input", "embedding", "data", {}),
        Edge("edge2", "embedding", "pinecone", "data", {}),
        Edge("edge3", "embedding", "weaviate", "data", {}),
        Edge("edge4", "embedding", "milvus", "data", {}),
        Edge("edge5", "input", "search", "query", {}),
        Edge("edge6", "search", "output", "results", {})
    ]

    # Create workflow
    workflow = Workflow(
        "vector_storage_workflow",
        "Store and search vectors using different providers",
        [input_node, embedding_node, pinecone_node, weaviate_node, milvus_node, search_node, output_node],
        edges,
        {}
    )

    # Test data
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog jumps over the lazy fox",
        "The lazy fox jumps over the quick brown dog",
        "A lazy brown fox jumps over the quick dog",
        "The quick dog jumps over the lazy brown fox"
    ]

    # Execute workflow
    result = await workflow.execute({
        "input": texts,
        "query": "Tell me about a quick fox"
    })
    
    print("Workflow execution completed!")
    print(f"Search results:\n{result}")

if __name__ == "__main__":
    asyncio.run(main()) 