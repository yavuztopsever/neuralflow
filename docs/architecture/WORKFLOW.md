# LangGraph Workflow Architecture

This document outlines the core workflow architecture of the LangGraph application. It serves as a technical reference for developers working with the codebase.

## Core Workflow Components

The LangGraph application implements a dynamic workflow with the following key components:

### 1. User Input Processing

- Entry point for all user queries
- Initializes structured context containers for the workflow
- Sets up state tracking for context sufficiency

```python
# Key component: graph/graph_workflow.py - user_input_node
state["retrieved_context"] = {
    "short_term_memory": [],
    "mid_term_memory": [],
    "long_term_memory": [],
    "vector_results": [],
    "graph_results": [],
    "document_results": [],
    "web_search_results": [],
    "context_sufficient": False
}
```

### 2. Context Manager

The Context Manager is responsible for gathering all relevant information and determining if it's sufficient to generate a high-quality response.

#### Context Retrieval Process:

1. **Memory Retrieval**
   - Gets short-term memory for recent conversation context
   - Gets mid-term memory for session-level context
   - Gets long-term memory with query relevance for historical context
   
2. **Vector Search**
   - Performs semantic search against the vector database
   - Returns content semantically similar to the user query
   
3. **Graph Search**
   - Searches the knowledge graph for related concepts
   - Extracts relationships and connected information
   
4. **Document RAG**
   - Determines document relevance to the query
   - Retrieves and processes relevant documents
   
5. **Context Sufficiency Evaluation**
   - Evaluates if current context is enough for a good response
   - Routes to Task Execution if more context is needed
   - Routes directly to Response Generation if context is sufficient
   
```python
# Key logic for routing decision in context_retrieval_node
context_sufficient = (
    len(state["retrieved_context"]["vector_results"]) > 0 or
    len(state["retrieved_context"]["graph_results"]) > 0 or
    len(state["retrieved_context"]["document_results"]) > 0 or
    len(state["retrieved_context"]["short_term_memory"]) > 0
)

if context_sufficient:
    return "response_generation"  # Direct to response if we have enough context
else:
    return "task_execution"  # Get more context if needed
```

### 3. Task Execution (Conditional)

Task Execution is triggered when the Context Manager determines that additional information is needed.

#### Key Operations:

- **Web Search**: Performs web searches for additional information
- **Function Calling**: Executes functions based on query requirements
- **Document Creation**: Creates research summaries from gathered information
- **Content Processing**: Structures additional information for context

```python
# Web search triggered when more context is needed
if state.get("needs_more_context", False):
    # Try to use web search tool if available
    if hasattr(executor, 'function_caller') and hasattr(executor.function_caller, 'web_search'):
        web_search = executor.function_caller.web_search
        search_results = web_search.search(state["user_query"])
        
        # Store results in context
        state["retrieved_context"]["web_search_results"] = search_results
```

### 4. Response Generation

Response Generation uses all available context, along with style and emotion understanding, to create tailored responses.

#### Response Process:

1. **Style & Emotion Analysis**
   - Analyzes the style of the user's query
   - Detects the emotional content of the message
   
2. **Context Integration**
   - Combines all context sources into a unified format
   - Prioritizes context based on relevance and freshness
   
3. **LLM Prompt Creation**
   - Creates a prompt with all context and style information
   - Formats the prompt for optimal LLM response
   
4. **Response Generation**
   - Generates the final natural language response
   - Ensures response quality and relevance
   
5. **Memory Integration**
   - Passes the interaction to Memory Manager for storage
   
```python
# Context integration for high-quality responses
all_context = {
    "user_query": state["user_query"],
    "short_term_memory": state["retrieved_context"].get("short_term_memory", []),
    "mid_term_memory": state["retrieved_context"].get("mid_term_memory", []),
    "long_term_memory": state["retrieved_context"].get("long_term_memory", []),
    "vector_results": state["retrieved_context"].get("vector_results", []),
    "graph_results": state["retrieved_context"].get("graph_results", []),
    "document_results": state["retrieved_context"].get("document_results", []),
    "web_search_results": state["retrieved_context"].get("web_search_results", []),
    "style": style_label,
    "emotion": emotion_label
}

# Generate response with all context
state["final_response"] = await response_generator.generate(
    state["execution_result"], 
    all_context
)
```

### 5. Memory Management

Memory Manager handles the storage and retrieval of conversation history across different time horizons.

#### Memory Operations:

- **Memory Split**: Categorizes information into appropriate memory stores
- **Context Storage**: Updates vector database with new information
- **Session Management**: Maintains user session information
- **Memory Pruning**: Handles automatic cleanup of old memories

```python
# Memory storage in Response Generation
if hasattr(response_generator, 'model_manager') and hasattr(response_generator.model_manager, 'memory_manager'):
    memory_manager = response_generator.model_manager.memory_manager
    
    # Save interaction to memory
    await memory_manager.save_interaction(
        interaction_type="conversation",
        user_query=state["user_query"],
        response=state["final_response"],
        document_path=None,
        search_results=str(all_context.get("vector_results", []))
    )
```

## Graph Structure

The LangGraph workflow is implemented as a directed graph with nodes representing each major component and edges defining the possible transitions.

```
START → user_input → context_retrieval → task_execution → response_generation → END
                            ↓
                    response_generation
```

The most important detail of this architecture is the dynamic routing from context_retrieval. Based on context sufficiency, it can either:
1. Route to task_execution if additional context is needed
2. Route directly to response_generation if sufficient context is available

This dynamic routing optimizes performance by avoiding unnecessary processing steps when they aren't needed.

## Implementation Details

The workflow is implemented in `graph/graph_workflow.py` with these key functions:

- `user_input_node`: Initializes the workflow
- `context_retrieval_node`: Gathers and evaluates context
- `task_execution_node`: Acquires additional context when needed
- `response_generation_node`: Generates the final response
- `create_workflow_graph`: Builds the complete workflow graph

Each node makes routing decisions based on the current state, allowing for a flexible and adaptive conversation experience.

# Graph Workflow System Documentation

This document provides detailed information about the LangGraph workflow system, including its components, execution flow, and management capabilities.

## Overview

The LangGraph workflow system is a powerful graph-based orchestration engine that manages the execution of complex AI workflows. It provides a flexible and extensible framework for building, executing, and monitoring AI-powered applications.

## Core Components

### 1. Workflow Definition
```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "version": "string",
  "nodes": [
    {
      "id": "string",
      "type": "string",
      "config": {},
      "metadata": {}
    }
  ],
  "edges": [
    {
      "from": "string",
      "to": "string",
      "condition": "string",
      "metadata": {}
    }
  ],
  "variables": {},
  "settings": {}
}
```

### 2. Node Types
- **Input Nodes**: Handle input processing and validation
- **Processing Nodes**: Perform specific tasks (LLM, memory, context)
- **Conditional Nodes**: Implement branching logic
- **Output Nodes**: Format and deliver results
- **Custom Nodes**: User-defined processing units

### 3. Edge Types
- **Direct Edges**: Simple node-to-node connections
- **Conditional Edges**: Branch based on conditions
- **Parallel Edges**: Enable concurrent execution
- **Error Edges**: Handle error cases
- **Loop Edges**: Enable workflow iteration

## Workflow Execution

### 1. Execution Flow
1. Workflow Initialization
2. Input Processing
3. Node Execution
4. Edge Traversal
5. Output Generation
6. Cleanup

### 2. Execution States
- **Pending**: Workflow is created but not started
- **Running**: Workflow is actively executing
- **Paused**: Workflow execution is temporarily stopped
- **Completed**: Workflow has finished successfully
- **Failed**: Workflow execution encountered an error
- **Cancelled**: Workflow was manually stopped

### 3. Error Handling
- **Node Errors**: Individual node failure handling
- **Edge Errors**: Connection failure handling
- **Workflow Errors**: Overall workflow failure handling
- **Recovery Mechanisms**: Error recovery and retry logic

## Workflow Management

### 1. Version Control
- **Versioning**: Track workflow versions
- **Migration**: Handle version updates
- **Compatibility**: Ensure version compatibility
- **Rollback**: Version rollback capabilities

### 2. Monitoring
- **Execution Metrics**: Track execution performance
- **Resource Usage**: Monitor resource consumption
- **Error Tracking**: Track and analyze errors
- **Usage Statistics**: Collect usage data

### 3. Optimization
- **Performance Tuning**: Optimize execution speed
- **Resource Management**: Efficient resource usage
- **Caching**: Implement caching strategies
- **Parallelization**: Enable parallel execution

## Node Configuration

### 1. Common Node Settings
```json
{
  "id": "string",
  "type": "string",
  "config": {
    "timeout": "number",
    "retry_count": "number",
    "retry_delay": "number",
    "max_attempts": "number"
  },
  "metadata": {
    "description": "string",
    "tags": ["string"],
    "version": "string"
  }
}
```

### 2. Node-Specific Settings
- **LLM Nodes**: Model configuration, temperature, tokens
- **Memory Nodes**: Memory type, capacity, retention
- **Context Nodes**: Context processing rules
- **Output Nodes**: Formatting and delivery options

## Edge Configuration

### 1. Edge Settings
```json
{
  "from": "string",
  "to": "string",
  "condition": {
    "type": "string",
    "expression": "string",
    "parameters": {}
  },
  "metadata": {
    "description": "string",
    "priority": "number"
  }
}
```

### 2. Condition Types
- **Boolean**: Simple true/false conditions
- **Expression**: Complex logical expressions
- **Threshold**: Numeric threshold conditions
- **Custom**: User-defined conditions

## Workflow Variables

### 1. Variable Types
- **Input Variables**: Workflow input parameters
- **Output Variables**: Workflow output values
- **Internal Variables**: Temporary processing values
- **Global Variables**: Shared across workflows

### 2. Variable Management
- **Declaration**: Define variable types and constraints
- **Assignment**: Set variable values
- **Validation**: Validate variable values
- **Scope**: Manage variable scope

## Best Practices

### 1. Workflow Design
- Keep workflows modular and reusable
- Implement proper error handling
- Use meaningful node and edge names
- Document workflow purpose and usage

### 2. Performance
- Optimize node execution
- Implement caching where appropriate
- Monitor resource usage
- Use parallel execution when possible

### 3. Security
- Validate all inputs
- Implement proper access controls
- Secure sensitive data
- Monitor for security issues

### 4. Maintenance
- Regular workflow updates
- Performance monitoring
- Error tracking and resolution
- Documentation updates

## SDK Examples

### Python
```python
from langgraph.api import WorkflowAPI

api = WorkflowAPI(api_key="your-api-key")

# Create workflow
workflow = api.create_workflow({
    "name": "Example Workflow",
    "description": "A sample workflow",
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
    }
})
```

### JavaScript
```javascript
const { WorkflowAPI } = require('langgraph');

const api = new WorkflowAPI('your-api-key');

// Create workflow
const workflow = await api.createWorkflow({
    name: 'Example Workflow',
    description: 'A sample workflow',
    nodes: [
        {
            id: 'input',
            type: 'input',
            config: {
                prompt: 'Enter your question:'
            }
        },
        {
            id: 'process',
            type: 'llm',
            config: {
                model: 'gpt-4',
                temperature: 0.7
            }
        },
        {
            id: 'output',
            type: 'output',
            config: {
                format: 'text'
            }
        }
    ],
    edges: [
        {
            from: 'input',
            to: 'process'
        },
        {
            from: 'process',
            to: 'output'
        }
    ]
});

// Execute workflow
const result = await api.executeWorkflow(workflow.id, {
    input: {
        question: 'What is the capital of France?'
    }
});
```