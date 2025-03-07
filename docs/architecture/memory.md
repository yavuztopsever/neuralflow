# Memory System Documentation

This document provides detailed information about the NeuralFlow memory system, including its components, management, and integration with workflows.

## Overview

The NeuralFlow memory system implements a conversation-based memory architecture that integrates with LangChain's memory management system. It provides efficient storage and retrieval of conversation history and context.

## Core Components

### 1. Conversation Memory

The conversation memory system is built on LangChain's `ConversationBufferMemory`:

```python
from langchain.memory import ConversationBufferMemory

class WorkflowManager:
    def __init__(self, config: Optional[WorkflowConfig] = None):
        # Initialize memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
```

### 2. State Management

The workflow state includes memory-related fields:

```python
class WorkflowState(BaseModel):
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    thinking: List[str] = Field(default_factory=list)
    conversation_id: Optional[str] = None
```

## Memory Operations

### 1. Message Storage

Messages are stored in the conversation memory:

```python
# Add user message to memory
self.memory.chat_memory.add_user_message(user_query)

# Add AI message to memory
self.memory.chat_memory.add_ai_message(response)
```

### 2. Context Management

Context is managed through the workflow state:

```python
# Store context in state
state["retrieved_context"] = {
    "search_results": context,
    "context_processed": True
}
```

### 3. Memory Integration

Memory is integrated into the workflow chain:

```python
def _create_workflow_chain(self):
    # Create prompts with memory
    context_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant that retrieves relevant context for queries."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{query}")
    ])
    
    # Add memory to chain
    context_chain = context_prompt | self.llm | StrOutputParser()
```

## Memory Features

### 1. Conversation History
- Maintains complete conversation history
- Supports message retrieval
- Enables context-aware responses

### 2. State Persistence
- Tracks workflow state
- Maintains processing flags
- Stores execution results

### 3. Context Management
- Stores retrieved context
- Manages context processing
- Enables context reuse

## Memory Integration

### 1. Workflow Integration
- Integrated with workflow nodes
- Supports state management
- Enables context sharing

### 2. LangChain Integration
- Uses LangChain memory system
- Supports conversation buffers
- Enables chain integration

### 3. Tool Integration
- Supports tool execution
- Maintains tool results
- Enables result reuse

## Best Practices

### 1. Memory Usage
- Use appropriate memory keys
- Maintain clean state
- Implement proper cleanup

### 2. Performance
- Optimize memory usage
- Implement efficient retrieval
- Monitor memory growth

### 3. Security
- Secure sensitive data
- Implement access controls
- Monitor access patterns

### 4. Maintenance
- Regular cleanup
- Performance optimization
- Security updates

## Future Improvements

Planned enhancements include:
- Enhanced memory types
- Advanced context management
- Improved state persistence
- Better memory optimization
- Extended tool support

## Memory Types

### 1. Short-term Memory
- **Purpose**: Store recent context and session data
- **Characteristics**:
  - Fast access
  - Limited capacity
  - Temporary storage
  - Session-specific

### 2. Mid-term Memory
- **Purpose**: Store session-level information
- **Characteristics**:
  - Moderate access speed
  - Medium capacity
  - Session persistence
  - Context-aware

### 3. Long-term Memory
- **Purpose**: Store historical data and knowledge
- **Characteristics**:
  - Persistent storage
  - Large capacity
  - Semantic search
  - Knowledge base

## Memory Components

### 1. Memory Store
```json
{
  "id": "string",
  "type": "string",
  "name": "string",
  "description": "string",
  "config": {
    "capacity": "number",
    "ttl": "number",
    "persistence": "string",
    "indexing": "string"
  },
  "metadata": {}
}
```

### 2. Memory Item
```json
{
  "id": "string",
  "store_id": "string",
  "content": "string",
  "metadata": {
    "timestamp": "string",
    "type": "string",
    "tags": ["string"],
    "source": "string"
  },
  "embedding": "array",
  "relationships": []
}
```

## Memory Operations

### 1. Storage Operations
- **Add Item**: Store new memory items
- **Update Item**: Modify existing items
- **Delete Item**: Remove items
- **Batch Operations**: Handle multiple items

### 2. Retrieval Operations
- **Get Item**: Retrieve specific items
- **Search Items**: Find items by query
- **List Items**: Enumerate items
- **Filter Items**: Apply filters

### 3. Management Operations
- **Create Store**: Initialize new store
- **Delete Store**: Remove store
- **Update Store**: Modify store settings
- **Optimize Store**: Improve performance

## Memory APIs

### 1. Store Management API
```python
# Create store
store = api.create_memory_store({
    "type": "short_term",
    "name": "Session Store",
    "config": {
        "capacity": 1000,
        "ttl": 3600
    }
})

# Add items
api.add_memory_items(store.id, {
    "items": [
        {
            "content": "Sample content",
            "metadata": {
                "type": "text",
                "timestamp": "2024-03-06T12:00:00Z"
            }
        }
    ]
})
```

### 2. Retrieval API
```python
# Get items
items = api.get_memory_items(store.id, {
    "query": "search query",
    "limit": 10,
    "offset": 0
})

# Search items
results = api.search_memory_items(store.id, {
    "query": "search query",
    "filters": {
        "type": "text",
        "timestamp": {
            "start": "2024-03-01T00:00:00Z",
            "end": "2024-03-07T23:59:59Z"
        }
    }
})
```

### 3. Management API
```python
# Update store
api.update_memory_store(store.id, {
    "config": {
        "capacity": 2000,
        "ttl": 7200
    }
})

# Optimize store
api.optimize_memory_store(store.id)
```

## Error Handling

The memory system includes comprehensive error handling:
- Input validation
- Storage errors
- Retrieval errors
- Performance monitoring

## Security Considerations

Security features include:
- Access control
- Data encryption
- Audit logging
- Rate limiting

## Performance Optimization

Performance features include:
- Caching
- Batch operations
- Indexing
- Query optimization

## Monitoring and Logging

Monitoring features include:
- Usage metrics
- Performance metrics
- Error tracking
- Audit logs 