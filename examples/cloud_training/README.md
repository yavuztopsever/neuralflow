# Cloud Training Examples

This directory contains example scripts for training models on AWS SageMaker. The examples demonstrate different training scenarios:

## Basic Training (`train.py`)

A basic training script that shows how to:
- Parse command line arguments for training parameters
- Load and preprocess datasets
- Create and compile a model
- Configure training callbacks
- Log metrics to SageMaker
- Save model checkpoints and artifacts

### Usage

```bash
python train.py \
    --train s3://bucket/train \
    --validation s3://bucket/validation \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 0.001
```

## Distributed Training (`distributed_train.py`)

An example of distributed training using TensorFlow's `MultiWorkerMirroredStrategy`. The script demonstrates:
- Setting up distribution strategy based on SageMaker environment
- Adjusting batch sizes for multiple workers
- Coordinating model saving across workers
- Handling metric logging in distributed settings

### Usage

```bash
python distributed_train.py \
    --train s3://bucket/train \
    --validation s3://bucket/validation \
    --epochs 10 \
    --batch-size 256  # Global batch size
```

## Hyperparameter Tuning (`hyperparameter_tuning.py`)

A script for hyperparameter optimization, showing how to:
- Define tunable hyperparameters
- Create models with variable architectures
- Report metrics for optimization
- Implement early stopping
- Save best performing models

### Usage

```bash
python hyperparameter_tuning.py \
    --train s3://bucket/train \
    --validation s3://bucket/validation \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --dropout-rate 0.2 \
    --num-filters 32 \
    --dense-units 64
```

## Common Features

All scripts include:
- Comprehensive logging
- Model checkpointing
- TensorBoard integration
- Error handling
- Progress reporting

## Environment Variables

The scripts expect the following environment variables (set by SageMaker):
- `SM_MODEL_DIR`: Directory for saving model artifacts
- `TF_CONFIG`: JSON string containing cluster configuration (for distributed training)
- `RANK`: Worker rank in distributed training

## Security Best Practices

The examples follow AWS security best practices:
- Using IAM roles for authentication
- Encrypting data in transit and at rest
- Implementing network isolation when required
- Following the principle of least privilege

## Monitoring and Debugging

All scripts include:
- Metric logging to CloudWatch
- TensorBoard integration for visualization
- Comprehensive error messages
- Progress tracking
- Model checkpointing

## Error Handling

The scripts implement robust error handling for:
- Data loading failures
- Network issues in distributed training
- Invalid hyperparameters
- Resource constraints
- Model saving errors

## Requirements

Required Python packages:
```
tensorflow>=2.4.0
sagemaker-training>=3.9.2
boto3>=1.26.0
```

## Best Practices

1. **Data Preparation**
   - Use appropriate data formats (TFRecord recommended)
   - Implement proper shuffling
   - Handle class imbalance
   - Validate data before training

2. **Resource Management**
   - Choose appropriate instance types
   - Set batch sizes based on available memory
   - Enable auto-scaling when needed
   - Monitor resource utilization

3. **Security**
   - Use encrypted S3 buckets
   - Implement network isolation
   - Follow IAM best practices
   - Regularly update dependencies

4. **Monitoring**
   - Set up CloudWatch alarms
   - Monitor training metrics
   - Track resource utilization
   - Enable debugging when needed

## Troubleshooting

Common issues and solutions:
1. **Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Choose larger instance type

2. **Slow Training**
   - Optimize data pipeline
   - Use more workers
   - Enable mixed precision training

3. **Network Issues**
   - Check VPC configuration
   - Verify security groups
   - Monitor network metrics

## Contributing

When contributing new examples:
1. Follow the existing code structure
2. Add comprehensive documentation
3. Implement proper error handling
4. Include security best practices
5. Add monitoring and logging
6. Test thoroughly before submitting 