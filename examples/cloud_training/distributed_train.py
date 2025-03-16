"""Example script for distributed training on SageMaker."""

import os
import json
import argparse
import tensorflow as tf
import sagemaker
from tensorflow.distribute import MultiWorkerMirroredStrategy

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    # Data paths
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--validation', type=str)
    
    # Model parameters
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--input-shape', type=tuple, default=(224, 224, 3))
    
    return parser.parse_args()

def get_distribution_strategy():
    """Configure distribution strategy based on environment.
    
    Returns:
        TensorFlow distribution strategy.
    """
    # Get TF_CONFIG from SageMaker environment
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    
    if tf_config:
        strategy = MultiWorkerMirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    
    return strategy

def create_model(strategy, num_classes: int, input_shape: tuple) -> tf.keras.Model:
    """Create model architecture within distribution strategy scope.
    
    Args:
        strategy: TensorFlow distribution strategy.
        num_classes: Number of output classes.
        input_shape: Input shape tuple.
        
    Returns:
        Compiled model.
    """
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def load_dataset(data_path: str, batch_size: int, strategy) -> tf.data.Dataset:
    """Load and prepare dataset for distributed training.
    
    Args:
        data_path: Path to data.
        batch_size: Global batch size.
        strategy: Distribution strategy.
        
    Returns:
        TensorFlow dataset.
    """
    # Adjust batch size per replica
    per_replica_batch_size = batch_size // strategy.num_replicas_in_sync
    
    dataset = tf.data.Dataset.load(data_path)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(per_replica_batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def log_metrics(logs: Dict, step: int) -> None:
    """Log metrics to SageMaker.
    
    Args:
        logs: Dict containing metrics.
        step: Current step/epoch.
    """
    # Only log metrics from chief worker
    if os.environ.get('RANK', '0') == '0':
        for metric_name, metric_value in logs.items():
            sagemaker.log_metric(
                metric_name=metric_name,
                value=float(metric_value),
                iteration_number=step
            )

class DistributedMetricsCallback(tf.keras.callbacks.Callback):
    """Callback for logging metrics in distributed training."""
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Log metrics at epoch end.
        
        Args:
            epoch: Current epoch.
            logs: Dict containing metrics.
        """
        log_metrics(logs, epoch)

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Initialize distribution strategy
    strategy = get_distribution_strategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
    
    # Load datasets
    train_dataset = load_dataset(args.train, args.batch_size, strategy)
    val_dataset = None
    if args.validation:
        val_dataset = load_dataset(args.validation, args.batch_size, strategy)
    
    # Create and compile model within strategy scope
    model = create_model(strategy, args.num_classes, args.input_shape)
    
    # Configure callbacks
    callbacks = [
        DistributedMetricsCallback(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_dir, 'checkpoints'),
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.model_dir, 'logs')
        )
    ]
    
    # Train model
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save final model (only on chief worker)
    if os.environ.get('RANK', '0') == '0':
        model.save(os.path.join(args.model_dir, 'model'))

if __name__ == '__main__':
    main() 