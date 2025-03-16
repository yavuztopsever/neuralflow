# NeuralFlow API Reference

## Core APIs

### LLM Service

#### LLMProvider

Base class for LLM providers.

```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 100,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        pass
```

#### OpenAIProvider

OpenAI-specific implementation.

```python
class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        organization: Optional[str] = None
    ):
        pass
```

### Vector Store Service

#### VectorStore

Base class for vector stores.

```python
class VectorStore(ABC):
    @abstractmethod
    def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]]
    ) -> List[str]:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        pass
```

### Memory Service

#### MemoryManager

Manages context and token tracking.

```python
class MemoryManager:
    def __init__(
        self,
        max_tokens: int = 4096,
        ttl: int = 3600
    ):
        pass

    def add_context(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        pass

    def get_context(
        self,
        context_id: str
    ) -> Optional[Dict[str, Any]]:
        pass
```

## Data Processing

### DocumentProcessor

Handles document processing and chunking.

```python
class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        pass

    def process_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        pass
```

### DataPipeline

Manages data preprocessing and preparation.

```python
class DataPipeline:
    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: AutoTokenizer
    ):
        pass

    def prepare_dataset(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Dataset]:
        pass
```

## Training

### Trainer

Handles model training and evaluation.

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda"
    ):
        pass

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 3
    ) -> Dict[str, List[float]]:
        pass

    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        pass
```

### ExperimentTracker

Tracks training experiments and metrics.

```python
class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        save_dir: str = "experiments"
    ):
        pass

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int
    ) -> None:
        pass

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> str:
        pass
```

## Infrastructure

### Configuration

```python
class Config:
    def __init__(
        self,
        config_path: str
    ):
        pass

    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        pass

    def set(
        self,
        key: str,
        value: Any
    ) -> None:
        pass
```

### Logging

```python
class Logger:
    def __init__(
        self,
        name: str,
        level: str = "INFO"
    ):
        pass

    def info(
        self,
        message: str,
        **kwargs
    ) -> None:
        pass

    def error(
        self,
        message: str,
        exc_info: bool = True,
        **kwargs
    ) -> None:
        pass
```

## Utilities

### TokenCounter

```python
class TokenCounter:
    def __init__(
        self,
        tokenizer: AutoTokenizer
    ):
        pass

    def count_tokens(
        self,
        text: str
    ) -> int:
        pass
```

### TextCleaner

```python
class TextCleaner:
    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_numbers: bool = False
    ):
        pass

    def clean(
        self,
        text: str
    ) -> str:
        pass
``` 