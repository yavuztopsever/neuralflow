"""Example script for remote model training using SageMaker."""

import os
from pathlib import Path
from ..core.training.sagemaker_trainer import SageMakerTrainer
from ..config.aws_config import AWSConfig

def main():
    """Run remote model training example."""
    
    # Initialize trainer
    config = AWSConfig()
    trainer = SageMakerTrainer(config)
    
    # Define paths
    base_dir = Path(__file__).parent.parent.parent.parent
    data_dir = base_dir / "data"
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    model_dir = base_dir / "models"
    
    # Upload data to S3
    train_s3_path = trainer.upload_data(
        local_path=train_dir,
        s3_key_prefix="data/train"
    )
    val_s3_path = trainer.upload_data(
        local_path=val_dir,
        s3_key_prefix="data/validation"
    )
    
    # Create and configure estimator
    estimator = trainer.create_estimator(
        entry_point="train.py",
        hyperparameters={
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
        }
    )
    
    # Start training
    print("Starting remote training job...")
    trainer.train(
        estimator=estimator,
        train_data=train_s3_path,
        validation_data=val_s3_path,
        wait=True
    )
    
    # Download trained model
    print("Training complete. Downloading model artifacts...")
    model_dir.mkdir(exist_ok=True)
    trainer.download_model(
        estimator=estimator,
        local_path=model_dir / "latest"
    )
    print("Model artifacts downloaded successfully.")

if __name__ == "__main__":
    main() 