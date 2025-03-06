# Web Search Integration Example

This example demonstrates how to use LangGraph with web search capabilities to gather information from the internet.

## Prerequisites

- Python 3.8 or higher
- LangGraph API key
- Required dependencies installed

## Installation

```bash
pip install langgraph requests beautifulsoup4
```

## Basic Web Search Usage

### 1. Initialize Web Search

```python
from langgraph.api import WebSearchAPI

# Initialize the API
api = WebSearchAPI(api_key="your-api-key")

# Initialize web search
api.initialize_search({
    "config": {
        "search_engine": "google",
        "api_key": "your-search-api-key",
        "max_results": 10
    }
})
```

### 2. Perform Search

```python
# Perform a web search
results = api.search({
    "query": "What is the capital of France?",
    "options": {
        "type": "web",
        "language": "en",
        "max_results": 5
    }
})

# Print results
for result in results["results"]:
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Snippet: {result['snippet']}\n")
```

### 3. Extract Content

```python
# Extract content from a webpage
content = api.extract_content({
    "url": "https://example.com",
    "options": {
        "extract_images": True,
        "extract_links": True
    }
})

print(f"Title: {content['title']}")
print(f"Content: {content['content']}")
print(f"Images: {len(content['images'])}")
print(f"Links: {len(content['links'])}")
```

## Advanced Web Search Usage

### Using Multiple Search Engines

```python
def multi_engine_example():
    api = WebSearchAPI(api_key="your-api-key")
    
    # Initialize with multiple engines
    api.initialize_search({
        "config": {
            "search_engine": "google",
            "api_key": "your-google-api-key",
            "max_results": 5
        }
    })
    
    # Perform search with Google
    google_results = api.search({
        "query": "Paris tourism",
        "options": {
            "type": "web",
            "language": "en"
        }
    })
    
    # Switch to Bing
    api.initialize_search({
        "config": {
            "search_engine": "bing",
            "api_key": "your-bing-api-key",
            "max_results": 5
        }
    })
    
    # Perform search with Bing
    bing_results = api.search({
        "query": "Paris tourism",
        "options": {
            "type": "web",
            "language": "en"
        }
    })
    
    # Compare results
    print("Google Results:")
    for result in google_results["results"]:
        print(f"- {result['title']}")
    
    print("\nBing Results:")
    for result in bing_results["results"]:
        print(f"- {result['title']}")

if __name__ == "__main__":
    multi_engine_example()
```

### Using Advanced Search Options

```python
def advanced_search_example():
    api = WebSearchAPI(api_key="your-api-key")
    
    # Initialize search
    api.initialize_search({
        "config": {
            "search_engine": "google",
            "api_key": "your-search-api-key",
            "max_results": 10
        }
    })
    
    # Perform advanced search
    results = api.search({
        "query": "Paris tourism",
        "options": {
            "type": "web",
            "language": "fr",
            "region": "FR",
            "time_range": "past_year",
            "safe_search": True,
            "max_results": 5,
            "filter": {
                "site": "tripadvisor.com",
                "file_type": "html"
            }
        }
    })
    
    # Process results
    for result in results["results"]:
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Date: {result['metadata']['date']}")
        print(f"Language: {result['metadata']['language']}\n")

if __name__ == "__main__":
    advanced_search_example()
```

### Using Content Extraction

```python
def content_extraction_example():
    api = WebSearchAPI(api_key="your-api-key")
    
    # Search for relevant pages
    results = api.search({
        "query": "Paris tourist attractions",
        "options": {
            "type": "web",
            "max_results": 3
        }
    })
    
    # Extract content from each result
    for result in results["results"]:
        content = api.extract_content({
            "url": result["url"],
            "options": {
                "extract_images": True,
                "extract_links": True,
                "extract_metadata": True,
                "timeout": 30
            }
        })
        
        print(f"\nExtracted from: {result['url']}")
        print(f"Title: {content['title']}")
        print(f"Description: {content['description']}")
        print(f"Images found: {len(content['images'])}")
        print(f"Links found: {len(content['links'])}")
        print(f"Metadata: {content['metadata']}")

if __name__ == "__main__":
    content_extraction_example()
```

### Using Search History

```python
def search_history_example():
    api = WebSearchAPI(api_key="your-api-key")
    
    # Perform multiple searches
    queries = [
        "Paris tourism",
        "Paris hotels",
        "Paris restaurants"
    ]
    
    for query in queries:
        api.search({
            "query": query,
            "options": {
                "type": "web",
                "max_results": 5
            }
        })
    
    # Get search history
    history = api.get_search_history({
        "limit": 10,
        "sort": "timestamp",
        "order": "desc"
    })
    
    print("Search History:")
    for entry in history["history"]:
        print(f"Query: {entry['query']}")
        print(f"Timestamp: {entry['timestamp']}")
        print(f"Results: {entry['result_count']}\n")
    
    # Clear search history
    api.clear_search_history()

if __name__ == "__main__":
    search_history_example()
```

### Using Result Filtering

```python
def result_filtering_example():
    api = WebSearchAPI(api_key="your-api-key")
    
    # Perform search
    results = api.search({
        "query": "Paris tourism",
        "options": {
            "type": "web",
            "max_results": 20
        }
    })
    
    # Filter results
    filtered_results = api.filter_results(results["search_id"], {
        "filters": {
            "date_range": {
                "start": "2024-01-01",
                "end": "2024-03-06"
            },
            "language": "en",
            "domain": "tripadvisor.com",
            "content_type": "article"
        }
    })
    
    print("Filtered Results:")
    for result in filtered_results["results"]:
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Date: {result['metadata']['date']}\n")

if __name__ == "__main__":
    result_filtering_example()
```

## Best Practices

1. **Search Configuration**
   - Choose appropriate search engine
   - Set reasonable result limits
   - Configure language and region

2. **Query Optimization**
   - Use specific keywords
   - Include relevant filters
   - Consider time ranges

3. **Content Extraction**
   - Set appropriate timeouts
   - Handle extraction errors
   - Validate extracted content

4. **Performance**
   - Use pagination
   - Implement caching
   - Monitor rate limits

5. **Maintenance**
   - Clear search history
   - Update API keys
   - Monitor usage 