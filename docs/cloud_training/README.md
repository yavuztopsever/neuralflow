# Cloud-Based Model Training with NeuralFlow

This guide explains how to train models in the cloud using NeuralFlow's SageMaker integration.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Security](#security)
- [Training Workflows](#training-workflows)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

1. AWS Account Setup:
   - Active AWS account
   - AWS CLI installed and configured
   - SageMaker Domain created
   - Appropriate IAM roles and permissions

2. Environment Setup:
   ```bash
   # Clone the repository
   git clone https://github.com/your-org/neuralflow.git
   cd neuralflow

   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. Configuration:
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env with your settings
   nano .env
   ```

## Configuration

### Domain Configuration
```env
SAGEMAKER_DOMAIN_ID=d-dmrlqv7dxicz
SAGEMAKER_DOMAIN_NAME=QuickSetupDomain-20250316T183768
SAGEMAKER_VPC_ID=vpc-0a87459cb380bbdee
```

### Network Configuration
```env
SAGEMAKER_SUBNET_1=subnet-0b77f2ba7ea3add36
SAGEMAKER_SUBNET_2=subnet-0238e00b9ed8017d7
SAGEMAKER_SUBNET_3=subnet-004e61d766db435e9
SAGEMAKER_NETWORK_MODE=Public
```

### Security Configuration
```env
AWS_KMS_KEY_ID=your_kms_key_id
AWS_VPC_SECURITY_GROUPS=sg-xxxxx,sg-yyyyy
ENABLE_NETWORK_ISOLATION=true
ENABLE_CONTAINER_ENCRYPTION=true
```

## Security

### Network Security
- VPC Configuration
- Subnet Management
- Security Groups
- Network Isolation

### Data Security
- KMS Encryption
- S3 Bucket Encryption
- Inter-Container Traffic Encryption

### Authentication
- IAM Roles
- Execution Roles
- Access Management

## Training Workflows

### Basic Training Flow
```python
from neuralflow.core.training import SageMakerTrainer

# Initialize trainer
trainer = SageMakerTrainer()

# Create estimator
estimator = trainer.create_estimator(
    entry_point="train.py",
    hyperparameters={
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001
    }
)

# Upload and train
train_data = trainer.upload_data("data/train", "training/data")
val_data = trainer.upload_data("data/validation", "validation/data")

trainer.train(
    estimator=estimator,
    train_data=train_data,
    validation_data=val_data
)
```

### Distributed Training
```python
# Configure distributed training
estimator = trainer.create_estimator(
    entry_point="distributed_train.py",
    instance_count=4,
    instance_type="ml.p3.8xlarge",
    hyperparameters={
        'epochs': 20,
        'batch_size': 64,
        'learning_rate': 0.001
    }
)
```

### Custom Training Script
```python
# train.py
import argparse
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--validation', type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load data
    train_data = tf.data.Dataset.load(args.train)
    val_data = tf.data.Dataset.load(args.validation) if args.validation else None
    
    # Build model
    model = tf.keras.Sequential([...])
    
    # Train
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    model.save('/opt/ml/model')

if __name__ == '__main__':
    main()
```

## Monitoring

### CloudWatch Integration
- Metrics monitoring
- Log analysis
- Container insights

### Training Metrics
```python
# Enable metrics in estimator
estimator = trainer.create_estimator(
    entry_point="train.py",
    enable_cloudwatch_metrics=True,
    enable_container_insights=True
)
```

### Custom Metrics
```python
# In training script
import sagemaker

def log_custom_metric(name, value, iteration):
    sagemaker.log_metric(
        metric_name=name,
        value=value,
        iteration_number=iteration
    )
```

## Troubleshooting

### Common Issues

1. Permission Errors:
   - Verify IAM roles
   - Check VPC configurations
   - Validate KMS key permissions

2. Network Issues:
   - Verify VPC endpoints
   - Check security group rules
   - Validate subnet configurations

3. Resource Issues:
   - Check instance limits
   - Monitor resource utilization
   - Verify instance types

### Debug Mode
```python
# Enable debug mode
estimator = trainer.create_estimator(
    entry_point="train.py",
    debugger_hook_config=True,
    debugger_rule_configs=True
)
```

### Logging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

1. Security:
   - Always enable network isolation
   - Use KMS encryption
   - Implement proper IAM roles

2. Performance:
   - Choose appropriate instance types
   - Optimize data loading
   - Use distributed training when needed

3. Monitoring:
   - Enable CloudWatch metrics
   - Monitor resource utilization
   - Track training progress

4. Cost Management:
   - Use appropriate instance types
   - Monitor training duration
   - Clean up resources after training 