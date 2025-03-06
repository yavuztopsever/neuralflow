"""
Example demonstrating the use of LangChain workflow system with advanced features.
"""
import asyncio
import os
from dotenv import load_dotenv
from src.config.langchain_config import (
    LangChainConfig,
    LLMConfig,
    MemoryConfig,
    VectorStoreConfig,
    DocumentLoaderConfig
)
from src.core.workflow.workflow_manager import WorkflowManager, WorkflowConfig

# Load environment variables
load_dotenv()

async def main():
    # Initialize LangChain configuration
    langchain_config = LangChainConfig(
        llm=LLMConfig(
            provider="openai",
            model_name="gpt-4-turbo-preview",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        memory=MemoryConfig(
            type="token",
            max_tokens=2000,
            return_messages=True
        ),
        vector_store=VectorStoreConfig(
            type="chroma",
            persist_directory="./data/vectorstore"
        ),
        document_loader=DocumentLoaderConfig(
            type="text",
            file_path="./data/sample.txt",
            chunk_size=1000,
            chunk_overlap=200
        )
    )
    
    # Initialize workflow configuration
    workflow_config = WorkflowConfig(
        max_context_items=5,
        max_parallel_tasks=3,
        response_format="text",
        include_sources=True,
        include_metadata=True,
        execution_mode="safe",
        priority=0,
        add_thinking=True,
        langchain_config=langchain_config
    )
    
    # Create workflow manager
    workflow_manager = WorkflowManager(workflow_config)
    
    # Example queries demonstrating different capabilities
    queries = [
        # Basic question answering
        "What is the capital of France?",
        
        # Mathematical calculation
        "Calculate 25 * 4",
        
        # Python code execution
        "Write a Python function to reverse a string",
        
        # Document-based question
        "What are the key points from the sample document?",
        
        # Wikipedia search
        "Tell me about quantum computing",
        
        # Complex task with multiple tools
        "Search for information about Python programming, then write a simple example"
    ]
    
    # Process queries
    for query in queries:
        print(f"\nProcessing query: {query}")
        
        # Run with progress updates
        async for progress, message in workflow_manager.run_with_progress(query):
            print(f"Progress: {progress:.2%} - {message}")
        
        # Get final response
        response = await workflow_manager.run(query)
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main()) 