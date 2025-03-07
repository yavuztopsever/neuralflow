# Data Ingestion Module

## Overview
The data ingestion module handles the collection, processing, and preparation of data for model training. It implements a comprehensive data science pipeline that follows industry best practices.

## Components

### Data Science Pipeline
The `DataSciencePipeline` class orchestrates the entire data processing workflow:

1. **Data Validation and Quality Analysis**
   - Validates data format and structure
   - Checks data quality metrics
   - Detects outliers and anomalies
   - Ensures data meets minimum requirements

2. **Data Cleaning**
   - Removes invalid or corrupted data
   - Standardizes data format
   - Handles missing values
   - Filters out low-quality samples

3. **Data Augmentation**
   - Generates synthetic training examples
   - Applies text transformations
   - Creates variations of existing data
   - Maintains data quality during augmentation

4. **Feature Engineering**
   - Prepares features for model training
   - Computes statistical features
   - Handles text preprocessing
   - Creates derived features

5. **Cross-Validation**
   - Splits data into training and validation sets
   - Ensures balanced data distribution
   - Maintains data integrity
   - Supports multiple validation strategies

6. **Model Training Configuration**
   - Prepares training parameters
   - Optimizes batch sizes
   - Sets learning rates
   - Configures optimization settings

7. **Reporting and Visualization**
   - Generates quality reports
   - Creates visualization plots
   - Tracks metrics over time
   - Provides insights into data quality

### Data Validation
The `DataValidator` class provides comprehensive data validation:

- **Input Validation**
  - Checks data format and structure
  - Validates required fields
  - Ensures data consistency
  - Verifies data types

- **Quality Metrics**
  - Text quality scoring
  - Duplicate detection
  - Outlier identification
  - Statistical analysis

- **Data Cleaning**
  - Removes invalid samples
  - Standardizes formats
  - Handles edge cases
  - Maintains data integrity

### Data Augmentation
The `DataAugmentor` class handles data augmentation:

- **Text Augmentation**
  - Synonym replacement
  - Back-translation
  - Random insertion/deletion
  - Context-aware augmentation

- **Quality Control**
  - Maintains semantic meaning
  - Preserves data quality
  - Ensures consistency
  - Validates augmented data

## Usage

### Basic Usage
```python
from core.data_ingestion.training import DataSciencePipeline

# Initialize pipeline
pipeline = DataSciencePipeline(
    output_dir="data_science_output",
    models_root_dir="/path/to/models"
)

# Run pipeline
results = pipeline.run_pipeline(
    data=training_data,
    model_type="all",
    session_id="session_123"
)
```

### Data Validation
```python
from core.data_ingestion.training import DataValidator

# Initialize validator
validator = DataValidator()

# Validate embedding data
result = validator.validate_embedding_data(texts, embeddings)

# Validate LLM data
result = validator.validate_llm_data(examples)
```

### Data Augmentation
```python
from core.data_ingestion.training import DataAugmentor

# Initialize augmentor
augmentor = DataAugmentor()

# Augment embedding data
augmented_data = augmentor.augment_embedding_data(texts, embeddings)

# Augment LLM data
augmented_data = augmentor.augment_llm_data(examples)
```

## Configuration

### Pipeline Configuration
```python
pipeline_config = {
    "output_dir": "data_science_output",
    "n_splits": 5,
    "random_state": 42,
    "models_root_dir": "/path/to/models"
}
```

### Validation Configuration
```python
validation_config = {
    "min_sequence_length": 1,
    "max_sequence_length": 512,
    "min_examples": 5,
    "max_examples": 10000,
    "min_embedding_dim": 384,
    "max_embedding_dim": 768,
    "quality_threshold": 0.7,
    "min_unique_words": 10,
    "max_duplicate_ratio": 0.3,
    "min_tfidf_score": 0.1,
    "outlier_threshold": 2.0
}
```

## Output Structure

### Pipeline Results
```python
{
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
```

### Validation Results
```python
{
    "is_valid": True,
    "message": "Data validation successful",
    "metrics": {
        "num_samples": 1000,
        "avg_sequence_length": 10.5,
        "std_sequence_length": 2.3,
        "quality_score": 0.85,
        "duplicate_ratio": 0.1,
        "tfidf_score": 0.75
    },
    "timestamp": "2024-03-07T12:00:00"
}
```

## Best Practices

1. **Data Quality**
   - Always validate data before processing
   - Monitor quality metrics continuously
   - Clean data before augmentation
   - Verify augmented data quality

2. **Performance**
   - Use appropriate batch sizes
   - Optimize data processing
   - Cache intermediate results
   - Monitor memory usage

3. **Reproducibility**
   - Set random seeds
   - Document configurations
   - Version control data
   - Track all transformations

4. **Error Handling**
   - Validate input data
   - Handle edge cases
   - Provide clear error messages
   - Log important events

## Dependencies
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- nltk

## Configuration

### DataIngestionConfig
```python
config = DataIngestionConfig(
    # Database settings
    db_url="sqlite:///data/data_ingestion.db",
    db_pool_size=5,
    db_max_overflow=10,
    
    # Data processing settings
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    max_sequence_length=512,
    batch_size=32,
    
    # Storage settings
    data_dir=Path("data"),
    model_dir=Path("models"),
    cache_dir=Path("data/cache"),
    
    # Training settings
    min_session_messages=10,
    max_session_messages=1000,
    min_training_examples=5,
    validation_split=0.2,
    training_batch_size=32,
    training_learning_rate=2e-5,
    training_num_epochs=3,
    
    # Session settings
    session_timeout=3600,
    max_session_duration=86400,
    min_session_duration=300
)
```

## Usage

### 1. Session End Handling
```python
# When a session ends
session_end_data = {
    "session_data": {
        "session_id": "session_123",
        "user_id": "user_456",
        "conversation_data": {
            "messages": [...],
            "metadata": {...}
        }
    }
}

# Create and execute session end node
session_end_node = SessionEndNode("session_end", config)
result = await session_end_node.execute(session_end_data)
```

### 2. Data Processing
```python
# Process session data
processed_data = data_processor.process_session_data(
    session_data,
    data_types=[DataType.EMBEDDING, DataType.FINETUNING]
)

# Store processed data
data_processor.store_processed_data(
    session_id="session_123",
    processed_data=processed_data,
    model_name="default"
)
```

### 3. Training
```python
# Training is triggered automatically when:
# 1. Session ends
# 2. Minimum message threshold is met
# 3. Session duration is within limits
# 4. Sufficient training examples are available

# Manual training trigger
training_node = ModelTrainingNode("model_training", config)
result = await training_node.execute({
    "session_id": "session_123",
    "model_type": "all"
})
```

## Data Flow

1. **Session End**:
   - User ends session
   - SessionEndNode is triggered
   - Data is validated and stored

2. **Data Processing**:
   - Raw session data is processed
   - Data is validated and cached
   - Processed data is stored in database

3. **Training**:
   - Training is triggered asynchronously
   - Models are trained on processed data
   - Training results are logged and stored

## Requirements

### Session Data Format
```json
{
    "session_id": "string",
    "user_id": "string",
    "conversation_data": {
        "messages": [
            {
                "role": "user|assistant",
                "content": "string",
                "timestamp": "ISO8601"
            }
        ],
        "metadata": {
            "session_id": "string",
            "start_time": "ISO8601",
            "end_time": "ISO8601"
        }
    }
}
```

### Training Requirements
- Minimum 10 messages per session
- Maximum 1000 messages per session
- Minimum 5 training examples
- Session duration between 5 minutes and 24 hours

## Error Handling

The module includes comprehensive error handling:
- Input validation
- Data format validation
- Processing error handling
- Training error handling
- Resource cleanup

## Logging

All operations are logged with appropriate levels:
- INFO: Normal operations
- WARNING: Non-critical issues
- ERROR: Critical failures

## Resource Management

- Automatic GPU/CPU selection
- Memory management
- Database connection pooling
- Resource cleanup on completion

## Best Practices

1. **Session Management**:
   - Always use unique session IDs
   - Include complete metadata
   - Validate session duration

2. **Data Processing**:
   - Monitor data quality
   - Use appropriate batch sizes
   - Cache processed data

3. **Training**:
   - Monitor training progress
   - Validate training results
   - Clean up resources

4. **Configuration**:
   - Adjust settings based on available resources
   - Monitor memory usage
   - Regular database maintenance 