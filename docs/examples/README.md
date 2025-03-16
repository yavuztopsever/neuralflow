# NeuralFlow Examples

This directory contains example code and notebooks demonstrating NeuralFlow's features and capabilities.

## Basic Examples

### 1. Simple LLM Integration

```python
from neuralflow.tools.llm import OpenAIProvider

# Initialize provider
provider = OpenAIProvider(
    api_key="your_api_key",
    model_name="gpt-3.5-turbo"
)

# Generate text
response = await provider.generate(
    prompt="Explain quantum computing",
    temperature=0.7,
    max_tokens=100
)

print(response.text)
```

### 2. Vector Store Usage

```python
from neuralflow.tools.vector_store import FAISSVectorStore
import numpy as np

# Initialize store
store = FAISSVectorStore(
    dimension=1536,
    metric="cosine"
)

# Add vectors
vectors = [np.random.rand(1536) for _ in range(10)]
metadata = [{"id": i, "text": f"Document {i}"} for i in range(10)]
ids = store.add_vectors(vectors, metadata)

# Search
query_vector = np.random.rand(1536)
results = store.search(query_vector, k=3)
```

### 3. Document Processing

```python
from neuralflow.tools.processing import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(
    chunk_size=512,
    chunk_overlap=50
)

# Process document
document = """
Long document text...
"""

result = processor.process_document(
    text=document,
    metadata={"source": "example.txt"}
)

print(f"Number of chunks: {len(result.chunks)}")
```

## Advanced Examples

### 1. Custom Training Pipeline

```python
from neuralflow.training import Trainer, DataPipeline
from transformers import AutoTokenizer
import pandas as pd

# Prepare data
data = pd.DataFrame({
    "text": ["Example 1", "Example 2"],
    "label": [0, 1]
})

# Initialize pipeline
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
pipeline = DataPipeline(
    config=DatasetConfig(
        text_column="text",
        label_column="label"
    ),
    tokenizer=tokenizer
)

# Prepare dataset
datasets = pipeline.prepare_dataset(data)

# Initialize trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer
)

# Train
metrics = trainer.train(
    train_dataloader=datasets["train"],
    val_dataloader=datasets["validation"],
    epochs=3
)
```

### 2. Memory Management

```python
from neuralflow.memory import MemoryManager

# Initialize manager
manager = MemoryManager(
    max_tokens=4096,
    ttl=3600
)

# Add context
context_id = manager.add_context(
    text="Important information to remember",
    metadata={"source": "user_input"}
)

# Retrieve context
context = manager.get_context(context_id)
```

### 3. Experiment Tracking

```python
from neuralflow.tracking import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(
    experiment_name="model_comparison",
    save_dir="experiments"
)

# Log metrics
tracker.log_metrics(
    metrics={
        "accuracy": 0.85,
        "loss": 0.23
    },
    step=100
)

# Save checkpoint
checkpoint_path = tracker.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=1
)
```

## Integration Examples

### 1. Full Application Setup

```python
from neuralflow.tools.llm import OpenAIProvider
from neuralflow.tools.vector_store import FAISSVectorStore
from neuralflow.memory import MemoryManager
from neuralflow.processing import DocumentProcessor

class Application:
    def __init__(self):
        # Initialize components
        self.llm = OpenAIProvider(
            api_key="your_api_key",
            model_name="gpt-3.5-turbo"
        )
        
        self.vector_store = FAISSVectorStore(
            dimension=1536,
            metric="cosine"
        )
        
        self.memory = MemoryManager(
            max_tokens=4096,
            ttl=3600
        )
        
        self.processor = DocumentProcessor(
            chunk_size=512,
            chunk_overlap=50
        )
    
    async def process_query(self, query: str) -> str:
        # Process query
        context = self.memory.get_relevant_context(query)
        
        # Generate response
        response = await self.llm.generate(
            prompt=f"Context: {context}\nQuery: {query}",
            temperature=0.7
        )
        
        return response.text

# Usage
app = Application()
response = await app.process_query("What is quantum computing?")
```

### 2. API Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
application = Application()

class Query(BaseModel):
    text: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/query")
async def process_query(query: Query):
    try:
        response = await application.process_query(
            query=query.text
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 3. Batch Processing

```python
from neuralflow.processing import BatchProcessor
from typing import List

class BatchProcessor:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.application = Application()
    
    async def process_batch(
        self,
        queries: List[str]
    ) -> List[str]:
        results = []
        
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            batch_results = await asyncio.gather(
                *[self.application.process_query(q) for q in batch]
            )
            results.extend(batch_results)
        
        return results

# Usage
processor = BatchProcessor(batch_size=32)
queries = ["Query 1", "Query 2", "Query 3"]
results = await processor.process_batch(queries)
```

## Testing Examples

### 1. Unit Tests

```python
import pytest
from neuralflow.tools.llm import OpenAIProvider

@pytest.fixture
def llm_provider():
    return OpenAIProvider(
        api_key="test_key",
        model_name="test_model"
    )

def test_generate(llm_provider):
    response = await llm_provider.generate(
        prompt="Test prompt",
        temperature=0.7
    )
    assert response.text is not None
    assert len(response.text) > 0
```

### 2. Integration Tests

```python
import pytest
from neuralflow.application import Application

@pytest.fixture
async def application():
    app = Application()
    yield app
    await app.cleanup()

async def test_full_pipeline(application):
    query = "What is quantum computing?"
    response = await application.process_query(query)
    
    assert response is not None
    assert len(response) > 0
```

### 3. Performance Tests

```python
import pytest
import time
from neuralflow.application import Application

async def test_response_time(application):
    query = "What is quantum computing?"
    
    start_time = time.time()
    response = await application.process_query(query)
    duration = time.time() - start_time
    
    assert duration < 2.0  # Response should be under 2 seconds
``` 