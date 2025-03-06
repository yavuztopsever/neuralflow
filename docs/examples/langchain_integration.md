# LangChain Integration Example

This example demonstrates how to use LangGraph with LangChain to create powerful AI workflows.

## Prerequisites

- Python 3.8 or higher
- LangGraph API key
- LangChain installed
- Required dependencies installed

## Installation

```bash
pip install langgraph langchain openai
```

## Basic LangChain Integration

### 1. Initialize LangChain with LangGraph

```python
from langgraph.api import LangChainAPI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize LangGraph API
api = LangChainAPI(api_key="your-api-key")

# Initialize LangChain components
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}"
)
chain = LLMChain(llm=llm, prompt=prompt)

# Initialize LangChain in LangGraph
api.initialize_langchain({
    "config": {
        "openai_api_key": "your-openai-key",
        "model_name": "gpt-4",
        "temperature": 0.7
    }
})
```

### 2. Create a Workflow with LangChain Components

```python
# Create workflow with LangChain integration
workflow = api.create_workflow({
    "name": "LangChain Q&A",
    "description": "A workflow using LangChain components",
    "nodes": [
        {
            "id": "input",
            "type": "input",
            "config": {
                "prompt": "Enter your question:"
            }
        },
        {
            "id": "langchain",
            "type": "langchain",
            "config": {
                "chain_type": "qa",
                "chain_config": {
                    "llm": {
                        "model": "gpt-4",
                        "temperature": 0.7
                    },
                    "prompt": {
                        "template": "Answer the following question: {question}",
                        "input_variables": ["question"]
                    }
                }
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
            "to": "langchain"
        },
        {
            "from": "langchain",
            "to": "output"
        }
    ]
})
```

## Advanced LangChain Integration

### Using LangChain Tools

```python
from langgraph.api import LangChainAPI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

def tools_example():
    api = LangChainAPI(api_key="your-api-key")
    
    # Define custom tools
    tools = [
        Tool(
            name="Search",
            func=lambda x: "Search results for: " + x,
            description="Search for information"
        ),
        Tool(
            name="Calculator",
            func=lambda x: eval(x),
            description="Calculate mathematical expressions"
        )
    ]
    
    # Initialize LangChain with tools
    api.initialize_langchain({
        "config": {
            "openai_api_key": "your-openai-key",
            "model_name": "gpt-4",
            "tools": tools
        }
    })
    
    # Create workflow with tools
    workflow = api.create_workflow({
        "name": "Tool-Enabled Q&A",
        "description": "A workflow using LangChain tools",
        "nodes": [
            {
                "id": "input",
                "type": "input",
                "config": {
                    "prompt": "Enter your question:"
                }
            },
            {
                "id": "agent",
                "type": "langchain",
                "config": {
                    "chain_type": "agent",
                    "agent_type": "zero-shot-react-description",
                    "tools": tools
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
                "to": "agent"
            },
            {
                "from": "agent",
                "to": "output"
            }
        ]
    })
    
    # Execute workflow
    result = api.execute_workflow(workflow.id, {
        "input": {
            "question": "What is 2+2? Use the calculator tool."
        }
    })
    
    print(result["output"])
    
    # Clean up
    api.delete_workflow(workflow.id)

if __name__ == "__main__":
    tools_example()
```

### Using LangChain Memory

```python
from langgraph.api import LangChainAPI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def memory_example():
    api = LangChainAPI(api_key="your-api-key")
    
    # Initialize LangChain with memory
    api.initialize_langchain({
        "config": {
            "openai_api_key": "your-openai-key",
            "model_name": "gpt-4",
            "memory_type": "buffer",
            "memory_key": "history"
        }
    })
    
    # Create workflow with memory
    workflow = api.create_workflow({
        "name": "Conversational Agent",
        "description": "A workflow using LangChain memory",
        "nodes": [
            {
                "id": "input",
                "type": "input",
                "config": {
                    "prompt": "Enter your message:"
                }
            },
            {
                "id": "conversation",
                "type": "langchain",
                "config": {
                    "chain_type": "conversation",
                    "memory": {
                        "type": "buffer",
                        "memory_key": "history"
                    }
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
                "to": "conversation"
            },
            {
                "from": "conversation",
                "to": "output"
            }
        ]
    })
    
    # Execute workflow multiple times
    messages = [
        "Hello, how are you?",
        "What's your name?",
        "What did I ask you earlier?"
    ]
    
    for message in messages:
        result = api.execute_workflow(workflow.id, {
            "input": {
                "message": message
            }
        })
        print(f"User: {message}")
        print(f"Assistant: {result['output']}\n")
    
    # Clean up
    api.delete_workflow(workflow.id)

if __name__ == "__main__":
    memory_example()
```

### Using LangChain with Vector Stores

```python
from langgraph.api import LangChainAPI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

def vector_store_example():
    api = LangChainAPI(api_key="your-api-key")
    
    # Initialize LangChain with vector store
    api.initialize_langchain({
        "config": {
            "openai_api_key": "your-openai-key",
            "model_name": "gpt-4",
            "vector_store_type": "chroma",
            "embedding_function": "text-embedding-ada-002"
        }
    })
    
    # Create and populate vector store
    texts = [
        "Paris is the capital of France.",
        "The Eiffel Tower is in Paris.",
        "France is known for its wine."
    ]
    
    text_splitter = CharacterTextSplitter()
    documents = text_splitter.create_documents(texts)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # Create workflow with vector store
    workflow = api.create_workflow({
        "name": "Document Q&A",
        "description": "A workflow using LangChain vector store",
        "nodes": [
            {
                "id": "input",
                "type": "input",
                "config": {
                    "prompt": "Enter your question:"
                }
            },
            {
                "id": "qa",
                "type": "langchain",
                "config": {
                    "chain_type": "qa_with_sources",
                    "vector_store": vectorstore
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
                "to": "qa"
            },
            {
                "from": "qa",
                "to": "output"
            }
        ]
    })
    
    # Execute workflow
    result = api.execute_workflow(workflow.id, {
        "input": {
            "question": "What is the capital of France?"
        }
    })
    
    print(f"Q: What is the capital of France?")
    print(f"A: {result['output']}")
    print(f"Sources: {result['sources']}")
    
    # Clean up
    api.delete_workflow(workflow.id)

if __name__ == "__main__":
    vector_store_example()
```

## Best Practices

1. **Component Configuration**
   - Use appropriate model settings
   - Configure memory based on needs
   - Select suitable tools

2. **Error Handling**
   - Handle API errors
   - Manage tool failures
   - Implement retries

3. **Performance**
   - Use caching when possible
   - Optimize prompt templates
   - Monitor token usage

4. **Security**
   - Secure API keys
   - Validate inputs
   - Implement access controls

5. **Maintenance**
   - Keep dependencies updated
   - Monitor API changes
   - Document custom components 