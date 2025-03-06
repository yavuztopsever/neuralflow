# Basic Usage Example

This example demonstrates how to use the LangGraph API to create and execute a simple workflow.

## Prerequisites

- Python 3.8 or higher
- LangGraph API key
- Required dependencies installed

## Installation

```bash
pip install langgraph
```

## Basic Example

### 1. Initialize the API

```python
from langgraph.api import LangGraphAPI

# Initialize the API with your key
api = LangGraphAPI(api_key="your-api-key")
```

### 2. Create a Simple Workflow

```python
# Define workflow nodes
nodes = [
    {
        "id": "input",
        "type": "input",
        "config": {
            "prompt": "Enter your question:"
        }
    },
    {
        "id": "process",
        "type": "llm",
        "config": {
            "model": "gpt-4",
            "temperature": 0.7
        }
    },
    {
        "id": "output",
        "type": "output",
        "config": {
            "format": "text"
        }
    }
]

# Define workflow edges
edges = [
    {
        "from": "input",
        "to": "process"
    },
    {
        "from": "process",
        "to": "output"
    }
]

# Create the workflow
workflow = api.create_workflow({
    "name": "Simple Q&A",
    "description": "A basic question-answering workflow",
    "nodes": nodes,
    "edges": edges
})
```

### 3. Execute the Workflow

```python
# Execute the workflow with input
result = api.execute_workflow(workflow.id, {
    "input": {
        "question": "What is the capital of France?"
    }
})

# Print the result
print(result["output"])
```

## Complete Example

Here's a complete example that demonstrates error handling and workflow management:

```python
from langgraph.api import LangGraphAPI
import time

def create_and_execute_workflow():
    try:
        # Initialize API
        api = LangGraphAPI(api_key="your-api-key")
        
        # Create workflow
        workflow = api.create_workflow({
            "name": "Enhanced Q&A",
            "description": "A workflow with error handling and retries",
            "nodes": [
                {
                    "id": "input",
                    "type": "input",
                    "config": {
                        "prompt": "Enter your question:",
                        "validation": {
                            "required": True,
                            "min_length": 3
                        }
                    }
                },
                {
                    "id": "process",
                    "type": "llm",
                    "config": {
                        "model": "gpt-4",
                        "temperature": 0.7,
                        "max_tokens": 1000,
                        "retry_count": 3,
                        "retry_delay": 1
                    }
                },
                {
                    "id": "output",
                    "type": "output",
                    "config": {
                        "format": "text",
                        "include_metadata": True
                    }
                }
            ],
            "edges": [
                {
                    "from": "input",
                    "to": "process"
                },
                {
                    "from": "process",
                    "to": "output"
                }
            ]
        })
        
        # Execute workflow
        result = api.execute_workflow(workflow.id, {
            "input": {
                "question": "What is the capital of France?"
            },
            "options": {
                "timeout": 30,
                "max_steps": 10
            }
        })
        
        # Process result
        if result["status"] == "success":
            print("Answer:", result["output"])
            print("Metadata:", result["metadata"])
        else:
            print("Error:", result["error"])
            
        # Clean up
        api.delete_workflow(workflow.id)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    create_and_execute_workflow()
```

## Using Memory

This example demonstrates how to use the memory system to maintain context:

```python
from langgraph.api import LangGraphAPI

def memory_example():
    api = LangGraphAPI(api_key="your-api-key")
    
    # Create memory store
    store = api.create_memory_store({
        "name": "Conversation History",
        "type": "conversation",
        "config": {
            "max_items": 100,
            "ttl": 3600
        }
    })
    
    # Create workflow with memory
    workflow = api.create_workflow({
        "name": "Conversational Q&A",
        "description": "A workflow that maintains conversation history",
        "nodes": [
            {
                "id": "input",
                "type": "input",
                "config": {
                    "prompt": "Enter your question:"
                }
            },
            {
                "id": "process",
                "type": "llm",
                "config": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "memory_store_id": store.id
                }
            },
            {
                "id": "output",
                "type": "output",
                "config": {
                    "format": "text"
                }
            }
        ],
        "edges": [
            {
                "from": "input",
                "to": "process"
            },
            {
                "from": "process",
                "to": "output"
            }
        ]
    })
    
    # Execute workflow multiple times
    questions = [
        "What is the capital of France?",
        "What is its population?",
        "What is its main tourist attraction?"
    ]
    
    for question in questions:
        result = api.execute_workflow(workflow.id, {
            "input": {
                "question": question
            }
        })
        print(f"Q: {question}")
        print(f"A: {result['output']}\n")
    
    # Clean up
    api.delete_workflow(workflow.id)
    api.delete_memory_store(store.id)

if __name__ == "__main__":
    memory_example()
```

## Using Context

This example shows how to use the context system to provide additional information:

```python
from langgraph.api import LangGraphAPI

def context_example():
    api = LangGraphAPI(api_key="your-api-key")
    
    # Create context
    context = api.create_context({
        "name": "Document Analysis",
        "type": "document",
        "content": {
            "text": "Paris is the capital of France. It has a population of approximately 2.2 million people. The Eiffel Tower is its most famous landmark.",
            "metadata": {
                "source": "encyclopedia",
                "date": "2024-03-06"
            }
        }
    })
    
    # Create workflow with context
    workflow = api.create_workflow({
        "name": "Contextual Q&A",
        "description": "A workflow that uses document context",
        "nodes": [
            {
                "id": "input",
                "type": "input",
                "config": {
                    "prompt": "Enter your question:"
                }
            },
            {
                "id": "process",
                "type": "llm",
                "config": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "context_id": context.id
                }
            },
            {
                "id": "output",
                "type": "output",
                "config": {
                    "format": "text"
                }
            }
        ],
        "edges": [
            {
                "from": "input",
                "to": "process"
            },
            {
                "from": "process",
                "to": "output"
            }
        ]
    })
    
    # Execute workflow
    result = api.execute_workflow(workflow.id, {
        "input": {
            "question": "What is the population of Paris?"
        }
    })
    
    print(f"Q: What is the population of Paris?")
    print(f"A: {result['output']}")
    
    # Clean up
    api.delete_workflow(workflow.id)
    api.delete_context(context.id)

if __name__ == "__main__":
    context_example()
```

## Error Handling

This example demonstrates proper error handling:

```python
from langgraph.api import LangGraphAPI
from langgraph.exceptions import APIError, WorkflowError

def error_handling_example():
    api = LangGraphAPI(api_key="your-api-key")
    
    try:
        # Create workflow with invalid configuration
        workflow = api.create_workflow({
            "name": "Error Test",
            "nodes": [
                {
                    "id": "invalid",
                    "type": "unknown_type"  # Invalid node type
                }
            ]
        })
        
    except APIError as e:
        print(f"API Error: {str(e)}")
    except WorkflowError as e:
        print(f"Workflow Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
    finally:
        # Clean up if workflow was created
        if 'workflow' in locals():
            try:
                api.delete_workflow(workflow.id)
            except:
                pass

if __name__ == "__main__":
    error_handling_example()
```

## Best Practices

1. **Error Handling**
   - Always use try-except blocks
   - Handle specific exceptions
   - Clean up resources in finally blocks

2. **Resource Management**
   - Delete workflows when no longer needed
   - Monitor memory usage
   - Use appropriate timeouts

3. **Security**
   - Never expose API keys
   - Use environment variables
   - Implement proper access controls

4. **Performance**
   - Use appropriate batch sizes
   - Implement caching when possible
   - Monitor rate limits

5. **Maintenance**
   - Keep dependencies updated
   - Monitor API changes
   - Implement logging 