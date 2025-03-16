"""Example script for hyperparameter tuning on SageMaker."""

import os
import json
import argparse
import tensorflow as tf
import sagemaker

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    parser.add_argument('--num-filters', type=int, default=32)
    parser.add_argument('--dense-units', type=int, default=64)
    
    # Data paths
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--validation', type=str)
    
    # Model parameters
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--input-shape', type=tuple, default=(224, 224, 3))
    
    return parser.parse_args()

def create_model(args) -> tf.keras.Model:
    """Create model architecture with tunable hyperparameters.
    
    Args:
        args: Parsed command line arguments containing hyperparameters.
        
    Returns:
        Compiled model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=args.input_shape),
        tf.keras.layers.Conv2D(args.num_filters, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(args.dropout_rate),
        tf.keras.layers.Conv2D(args.num_filters * 2, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(args.dropout_rate),
        tf.keras.layers.Conv2D(args.num_filters * 2, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(args.dense_units, activation='relu'),
        tf.keras.layers.Dropout(args.dropout_rate),
        tf.keras.layers.Dense(args.num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_dataset(data_path: str, batch_size: int) -> tf.data.Dataset:
    """Load and prepare dataset.
    
    Args:
        data_path: Path to data.
        batch_size: Batch size.
        
    Returns:
        TensorFlow dataset.
    """
    dataset = tf.data.Dataset.load(data_path)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def log_metrics(logs: Dict, step: int) -> None:
    """Log metrics to SageMaker.
    
    Args:
        logs: Dict containing metrics.
        step: Current step/epoch.
    """
    for metric_name, metric_value in logs.items():
        sagemaker.log_metric(
            metric_name=metric_name,
            value=float(metric_value),
            iteration_number=step
        )

class HPTuningMetricsCallback(tf.keras.callbacks.Callback):
    """Callback for logging metrics during hyperparameter tuning."""
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Log metrics at epoch end.
        
        Args:
            epoch: Current epoch.
            logs: Dict containing metrics.
        """
        log_metrics(logs, epoch)
        
        # Report objective metric for hyperparameter tuning
        if 'val_accuracy' in logs:
            sagemaker.report_intermediate_objective_value(
                objective_name='validation:accuracy',
                objective_value=float(logs['val_accuracy']),
                step_number=epoch
            )

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load datasets
    train_dataset = load_dataset(args.train, args.batch_size)
    val_dataset = None
    if args.validation:
        val_dataset = load_dataset(args.validation, args.batch_size)
    
    # Create and compile model with tunable hyperparameters
    model = create_model(args)
    
    # Configure callbacks
    callbacks = [
        HPTuningMetricsCallback(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_dir, 'checkpoints'),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.model_dir, 'logs')
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(os.path.join(args.model_dir, 'model'))
    
    # Report final objective metric
    if val_dataset is not None:
        final_accuracy = max(history.history['val_accuracy'])
        sagemaker.report_final_objective_value(
            objective_name='validation:accuracy',
            objective_value=float(final_accuracy)
        )

if __name__ == '__main__':
    main() 