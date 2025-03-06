# Vector Store Integration Example

This example demonstrates how to use LangGraph with vector stores for document storage and retrieval.

## Prerequisites

- Python 3.8 or higher
- LangGraph API key
- Required dependencies installed

## Installation

```bash
pip install langgraph chromadb sentence-transformers
```

## Basic Vector Store Usage

### 1. Initialize Vector Store

```python
from langgraph.api import VectorStoreAPI

# Initialize the API
api = VectorStoreAPI(api_key="your-api-key")

# Initialize vector store
store = api.initialize_store({
    "type": "chroma",
    "config": {
        "persist_directory": "./data",
        "collection_name": "documents",
        "embedding_function": "all-MiniLM-L6-v2"
    }
})
```

### 2. Add Documents

```python
# Add documents to the store
documents = [
    {
        "id": "doc1",
        "content": "Paris is the capital of France.",
        "metadata": {
            "source": "encyclopedia",
            "date": "2024-03-06"
        }
    },
    {
        "id": "doc2",
        "content": "The Eiffel Tower is in Paris.",
        "metadata": {
            "source": "encyclopedia",
            "date": "2024-03-06"
        }
    }
]

api.add_documents(store.id, {
    "documents": documents
})
```

### 3. Search Documents

```python
# Search for similar documents
results = api.search_documents(store.id, {
    "query": "What is the capital of France?",
    "options": {
        "k": 2,
        "score_threshold": 0.7
    }
})

# Print results
for result in results["results"]:
    print(f"Score: {result['score']}")
    print(f"Content: {result['content']}")
    print(f"Metadata: {result['metadata']}\n")
```

## Advanced Vector Store Usage

### Using Multiple Collections

```python
def collections_example():
    api = VectorStoreAPI(api_key="your-api-key")
    
    # Initialize multiple stores
    news_store = api.initialize_store({
        "type": "chroma",
        "config": {
            "persist_directory": "./data/news",
            "collection_name": "news_articles",
            "embedding_function": "all-MiniLM-L6-v2"
        }
    })
    
    docs_store = api.initialize_store({
        "type": "chroma",
        "config": {
            "persist_directory": "./data/docs",
            "collection_name": "documents",
            "embedding_function": "all-MiniLM-L6-v2"
        }
    })
    
    # Add documents to different stores
    news_docs = [
        {
            "id": "news1",
            "content": "Breaking news about Paris.",
            "metadata": {"type": "news"}
        }
    ]
    
    doc_docs = [
        {
            "id": "doc1",
            "content": "Document about Paris.",
            "metadata": {"type": "document"}
        }
    ]
    
    api.add_documents(news_store.id, {"documents": news_docs})
    api.add_documents(docs_store.id, {"documents": doc_docs})
    
    # Search across stores
    news_results = api.search_documents(news_store.id, {
        "query": "Paris news",
        "options": {"k": 1}
    })
    
    doc_results = api.search_documents(docs_store.id, {
        "query": "Paris document",
        "options": {"k": 1}
    })
    
    # Clean up
    api.delete_store(news_store.id)
    api.delete_store(docs_store.id)

if __name__ == "__main__":
    collections_example()
```

### Using Custom Embeddings

```python
from sentence_transformers import SentenceTransformer

def custom_embeddings_example():
    api = VectorStoreAPI(api_key="your-api-key")
    
    # Initialize custom embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize store with custom embeddings
    store = api.initialize_store({
        "type": "chroma",
        "config": {
            "persist_directory": "./data",
            "collection_name": "custom_embeddings",
            "embedding_function": "custom"
        }
    })
    
    # Add documents with custom embeddings
    documents = [
        {
            "id": "doc1",
            "content": "Sample document 1",
            "embedding": model.encode("Sample document 1").tolist(),
            "metadata": {"source": "custom"}
        }
    ]
    
    api.add_documents(store.id, {
        "documents": documents,
        "options": {
            "use_custom_embeddings": True
        }
    })
    
    # Search with custom query embedding
    query_embedding = model.encode("Sample query").tolist()
    results = api.search_documents(store.id, {
        "query_embedding": query_embedding,
        "options": {
            "k": 1,
            "use_custom_embeddings": True
        }
    })
    
    # Clean up
    api.delete_store(store.id)

if __name__ == "__main__":
    custom_embeddings_example()
```

### Using Metadata Filtering

```python
def metadata_filtering_example():
    api = VectorStoreAPI(api_key="your-api-key")
    
    # Initialize store
    store = api.initialize_store({
        "type": "chroma",
        "config": {
            "persist_directory": "./data",
            "collection_name": "filtered_docs",
            "embedding_function": "all-MiniLM-L6-v2"
        }
    })
    
    # Add documents with metadata
    documents = [
        {
            "id": "doc1",
            "content": "Document about Paris",
            "metadata": {
                "city": "Paris",
                "country": "France",
                "date": "2024-03-06"
            }
        },
        {
            "id": "doc2",
            "content": "Document about London",
            "metadata": {
                "city": "London",
                "country": "UK",
                "date": "2024-03-06"
            }
        }
    ]
    
    api.add_documents(store.id, {"documents": documents})
    
    # Search with metadata filters
    results = api.search_documents(store.id, {
        "query": "capital city",
        "options": {
            "k": 2,
            "filter": {
                "country": "France",
                "date": "2024-03-06"
            }
        }
    })
    
    # Clean up
    api.delete_store(store.id)

if __name__ == "__main__":
    metadata_filtering_example()
```

### Using Batch Operations

```python
def batch_operations_example():
    api = VectorStoreAPI(api_key="your-api-key")
    
    # Initialize store
    store = api.initialize_store({
        "type": "chroma",
        "config": {
            "persist_directory": "./data",
            "collection_name": "batch_docs",
            "embedding_function": "all-MiniLM-L6-v2"
        }
    })
    
    # Generate batch of documents
    documents = [
        {
            "id": f"doc{i}",
            "content": f"Sample document {i}",
            "metadata": {"batch": "test"}
        }
        for i in range(100)
    ]
    
    # Add documents in batches
    batch_size = 20
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        api.add_documents(store.id, {
            "documents": batch,
            "options": {
                "batch_size": batch_size
            }
        })
    
    # Search with pagination
    page_size = 10
    page = 1
    
    while True:
        results = api.search_documents(store.id, {
            "query": "sample",
            "options": {
                "k": page_size,
                "page": page
            }
        })
        
        if not results["results"]:
            break
            
        print(f"Page {page}:")
        for result in results["results"]:
            print(f"- {result['content']}")
        
        page += 1
    
    # Clean up
    api.delete_store(store.id)

if __name__ == "__main__":
    batch_operations_example()
```

## Best Practices

1. **Store Configuration**
   - Choose appropriate embedding model
   - Configure persistence settings
   - Set up proper indexing

2. **Document Management**
   - Use meaningful IDs
   - Include relevant metadata
   - Implement versioning

3. **Search Optimization**
   - Use appropriate k values
   - Set score thresholds
   - Implement filtering

4. **Performance**
   - Use batch operations
   - Implement caching
   - Monitor store size

5. **Maintenance**
   - Regular cleanup
   - Monitor performance
   - Update embeddings 