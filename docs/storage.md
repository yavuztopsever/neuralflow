# Storage Components

This document describes the storage components in the NeuralFlow framework.

## Overview

The storage system consists of four main components:

1. State Persistence
2. Vector Store
3. Cache
4. Document Storage

Each component is designed to be modular, extensible, and efficient.

## State Persistence

The state persistence component handles the storage and retrieval of workflow states.

### Usage

```python
from core.state.persistence.state_persistence import StatePersistence
from pathlib import Path

# Initialize
persistence = StatePersistence(storage_dir=Path("storage/states"))

# Save state
persistence.save(state)

# Load state
state = persistence.load(state_id)

# Load all states
states = persistence.load_all()

# Delete state
persistence.delete(state_id)

# Cache management
persistence.cleanup_cache(max_idle=3600, max_age=86400)
stats = persistence.get_cache_stats()
```

### Features

- File-based persistence
- In-memory caching
- TTL support
- Thread-safe operations
- Comprehensive statistics

## Vector Store

The vector store component provides vector storage and similarity search capabilities.

### Usage

```python
from storage.vector.base import BaseVectorStore, VectorStoreConfig

# Configure
config = VectorStoreConfig(
    name="my_store",
    store_type="chroma",
    dimensions=1536,
    metric="cosine",
    embedder_model="all-MiniLM-L6-v2"
)

# Initialize
store = MyVectorStore(config)
store.initialize()

# Add vectors
ids = store.add_vectors(vectors, metadata)

# Add texts (auto-embedding)
ids = store.add_texts(texts, metadata)

# Search
results = store.search(query_vector, k=10)
results = store.search_texts(query_text, k=10)

# Delete
store.delete_vector(vector_id)
```

### Features

- Multiple vector store backends
- Built-in text embedding
- Metadata filtering
- Cosine similarity search
- Disk persistence

## Cache

The cache component provides in-memory caching with TTL support.

### Usage

```python
from storage.storage.memory.cache.base import BaseCache, CacheConfig

# Configure
config = CacheConfig(
    max_size=1000,
    ttl=3600  # 1 hour
)

# Initialize
cache = MyCache[str](config)  # Generic type parameter
cache.initialize()

# Set value
cache.set("key", "value", ttl=300)  # 5 minutes TTL

# Get value
value = cache.get("key")

# Check existence
exists = cache.exists("key")

# Delete value
cache.delete("key")

# Clear cache
cache.clear()

# Cleanup expired entries
removed = cache.cleanup(max_idle=3600)

# Get statistics
stats = cache.get_stats()
```

### Features

- Generic type support
- TTL support
- Maximum size limit
- Thread-safe operations
- Automatic cleanup
- Comprehensive statistics

## Document Storage

The document storage component provides document storage, retrieval, and search capabilities.

### Usage

```python
from storage.persistent.document.retriever import DocumentRetriever, DocumentConfig, Document

# Configure
config = DocumentConfig(
    storage_dir="storage/documents",
    vector_store_type="chroma",
    embedder_model="all-MiniLM-L6-v2"
)

# Initialize
retriever = DocumentRetriever(config)

# Create document
doc = Document(
    doc_id="doc1",
    content="Document content",
    metadata={"type": "article"}
)

# Add document
retriever.add_document(doc)

# Get document
doc = retriever.get_document(doc_id)

# Update document
retriever.update_document(
    doc_id="doc1",
    content="Updated content",
    metadata={"type": "article", "updated": True}
)

# Delete document
retriever.delete_document(doc_id)

# Search documents
results = retriever.search_documents(
    query="search query",
    top_k=5,
    metadata_filter={"type": "article"}
)

# List documents
docs = retriever.list_documents()

# Get statistics
stats = retriever.get_stats()
```

### Features

- File system storage
- Vector store integration
- Semantic search
- Metadata filtering
- Document versioning
- Comprehensive statistics

## Implementation Details

### Base Classes

All components follow a similar pattern with base classes defining the interface:

- `StatePersistence`: State storage and caching
- `BaseVectorStore`: Vector storage and search
- `BaseCache`: Generic caching functionality
- `DocumentRetriever`: Document storage and retrieval

### Configuration

Each component has its own configuration class:

- `VectorStoreConfig`: Vector store settings
- `CacheConfig`: Cache settings
- `DocumentConfig`: Document storage settings

### Thread Safety

All components are designed to be thread-safe:

- `StatePersistence` uses `RLock` for cache operations
- `BaseCache` uses `RLock` for all operations
- `DocumentRetriever` uses thread-safe vector store

### Error Handling

Components use consistent error handling:

- Proper exception types
- Detailed error messages
- Comprehensive logging
- Graceful degradation

### Testing

Each component has comprehensive test coverage:

- Unit tests
- Integration tests
- Mock implementations
- Test fixtures

## Best Practices

1. **Configuration**
   - Use appropriate configuration for each component
   - Set reasonable TTL values
   - Configure appropriate vector dimensions

2. **Error Handling**
   - Always handle exceptions
   - Log errors appropriately
   - Provide fallback behavior

3. **Resource Management**
   - Clean up resources when done
   - Use context managers where appropriate
   - Monitor memory usage

4. **Performance**
   - Use caching appropriately
   - Configure vector store indexes
   - Clean up expired entries

5. **Thread Safety**
   - Use thread-safe operations
   - Avoid long-held locks
   - Handle concurrent access

## Migration Guide

If you're migrating from the old storage implementations:

1. Update imports to use new paths:
   - `from core.state.persistence.state_persistence import StatePersistence`
   - `from storage.vector.base import BaseVectorStore`
   - `from storage.storage.memory.cache.base import BaseCache`
   - `from storage.persistent.document.retriever import DocumentRetriever`

2. Update configuration:
   - Use new configuration classes
   - Review and update settings

3. Update method calls:
   - Check method signatures
   - Update parameter names
   - Handle new return types

4. Update error handling:
   - Review exception types
   - Update error messages
   - Check logging

5. Update tests:
   - Use new test fixtures
   - Update assertions
   - Check thread safety 