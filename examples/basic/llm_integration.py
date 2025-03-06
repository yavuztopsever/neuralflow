"""
Example demonstrating LLM integration with different providers.
"""

import asyncio
from typing import Dict, Any, List

from src.core.graph import Workflow
from src.core.nodes import Node
from src.core.edges import Edge
from src.models.llm import BaseLLM

async def generate_with_openai(prompt: str) -> str:
    """Generate text using OpenAI's GPT model."""
    llm = BaseLLM({
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000
    })
    return await llm.generate(prompt)

async def generate_with_anthropic(prompt: str) -> str:
    """Generate text using Anthropic's Claude model."""
    llm = BaseLLM({
        "provider": "anthropic",
        "model": "claude-2",
        "temperature": 0.7,
        "max_tokens": 1000
    })
    return await llm.generate(prompt)

async def generate_with_huggingface(prompt: str) -> str:
    """Generate text using Hugging Face models."""
    llm = BaseLLM({
        "provider": "huggingface",
        "model": "gpt2",
        "temperature": 0.7,
        "max_tokens": 1000
    })
    return await llm.generate(prompt)

async def combine_results(results: List[str]) -> str:
    """Combine results from different LLMs."""
    return "\n\n".join([
        f"Result from {provider}:\n{result}"
        for provider, result in zip(["OpenAI", "Anthropic", "Hugging Face"], results)
    ])

async def main():
    """Main function."""
    # Create nodes
    input_node = Node("input", "input", {})
    openai_node = Node("openai", "llm", {"provider": "openai"})
    anthropic_node = Node("anthropic", "llm", {"provider": "anthropic"})
    huggingface_node = Node("huggingface", "llm", {"provider": "huggingface"})
    combine_node = Node("combine", "combine", {})
    output_node = Node("output", "output", {})

    # Create edges
    edges = [
        Edge("edge1", "input", "openai", "data", {}),
        Edge("edge2", "input", "anthropic", "data", {}),
        Edge("edge3", "input", "huggingface", "data", {}),
        Edge("edge4", "openai", "combine", "data", {}),
        Edge("edge5", "anthropic", "combine", "data", {}),
        Edge("edge6", "huggingface", "combine", "data", {}),
        Edge("edge7", "combine", "output", "data", {})
    ]

    # Create workflow
    workflow = Workflow(
        "llm_comparison_workflow",
        "Compare text generation from different LLM providers",
        [input_node, openai_node, anthropic_node, huggingface_node, combine_node, output_node],
        edges,
        {}
    )

    # Execute workflow
    prompt = "Write a short story about a robot learning to paint."
    result = await workflow.execute({"input": prompt})
    
    print("Workflow execution completed!")
    print(f"Results:\n{result}")

if __name__ == "__main__":
    asyncio.run(main()) 