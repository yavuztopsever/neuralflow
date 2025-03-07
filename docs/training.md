# Model Training Module

## Overview
The model training module implements a comprehensive training pipeline that follows industry best practices for machine learning model training. It handles data preparation, model training, evaluation, and deployment.

## Components

### Data Science Pipeline
The `DataSciencePipeline` class orchestrates the entire training process:

1. **Data Preparation**
   - Data validation and quality checks
   - Data cleaning and preprocessing
   - Feature engineering
   - Data augmentation

2. **Model Training**
   - Batch size optimization
   - Learning rate scheduling
   - Mixed precision training
   - Gradient clipping
   - Early stopping

3. **Evaluation**
   - Cross-validation
   - Performance metrics
   - Model comparison
   - Error analysis

4. **Deployment**
   - Model versioning
   - Root model updates
   - Model persistence
   - Configuration management

### Model Training Node
The `ModelTrainingNode` class coordinates the training workflow:

- **Session Management**
  - Training session tracking
  - Data retrieval
  - Result storage
  - Resource cleanup

- **Pipeline Integration**
  - Pipeline initialization
  - Configuration management
  - Result processing
  - Error handling

- **Model Management**
  - Model directory handling
  - Root model updates
  - Version control
  - Resource cleanup

## Usage

### Basic Usage
```python
from core.data_ingestion.workflow_nodes import ModelTrainingNode

# Initialize training node
training_node = ModelTrainingNode(
    config=NodeConfig(
        model_dir="models",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="deepseek-1.5b"
    ),
    db_manager=db_manager
)

# Execute training
results = training_node.execute({
    "session_id": "session_123",
    "model_type": "all"
})
```

### Pipeline Configuration
```python
pipeline_config = {
    "output_dir": "training_output",
    "n_splits": 5,
    "random_state": 42,
    "models_root_dir": "/path/to/models"
}
```

### Training Configuration
```python
training_config = {
    "embedding": {
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "use_mixed_precision": True,
        "max_grad_norm": 1.0,
        "early_stopping_patience": 3,
        "logging_steps": 100
    },
    "llm": {
        "batch_size": 8,
        "learning_rate": 1e-5,
        "num_epochs": 3,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "use_mixed_precision": True,
        "max_grad_norm": 1.0,
        "early_stopping_patience": 3,
        "logging_steps": 100
    }
}
```

## Output Structure

### Training Results
```python
{
    "session_id": "session_123",
    "model_type": "all",
    "timestamp": "2024-03-07T12:00:00",
    "results": {
        "timestamp": "2024-03-07T12:00:00",
        "model_type": "all",
        "session_id": "session_123",
        "steps": {
            "validation": {...},
            "cleaning": {...},
            "augmentation": {...},
            "feature_engineering": {...},
            "cross_validation": {...},
            "training": {...}
        }
    }
}
```

## Best Practices

1. **Data Quality**
   - Validate data before training
   - Monitor training metrics
   - Track data quality
   - Handle outliers

2. **Training Process**
   - Use appropriate batch sizes
   - Monitor learning curves
   - Implement early stopping
   - Save checkpoints

3. **Model Management**
   - Version control models
   - Track configurations
   - Monitor performance
   - Clean up resources

4. **Error Handling**
   - Validate inputs
   - Handle exceptions
   - Log errors
   - Provide feedback

## Dependencies
- torch
- transformers
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn 