#!/usr/bin/env python3
"""
Training script for NeuralFlow models.
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.neuralflow.tools.data.core.unified_data import UnifiedData, DataConfig
from src.neuralflow.tools.models.core.unified_model import UnifiedModel, ModelConfig
from src.neuralflow.tools.monitoring.core.unified_monitor import UnifiedMonitor, MonitoringConfig

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        if config_path.endswith('.json'):
            return json.load(f)
        return yaml.safe_load(f)

async def train_model(
    model_type: str,
    model_name: str,
    data_path: str,
    config: Dict[str, Any]
) -> None:
    """Train a model.
    
    Args:
        model_type: Type of model to train
        model_name: Name of the model
        data_path: Path to training data
        config: Training configuration
    """
    # Initialize components
    data_system = UnifiedData(DataConfig(**config.get('data', {})))
    model_system = UnifiedModel(ModelConfig(**config.get('model', {})))
    monitor = UnifiedMonitor(MonitoringConfig(
        monitor_id=f"train_{model_name}",
        monitor_type="training"
    ))
    
    try:
        # Load and process data
        raw_data = load_training_data(data_path)
        processed_data = await data_system.process_data(
            data=raw_data,
            processor="training",
            data_type=model_type
        )
        
        # Train model
        training_metrics = await model_system.train(
            model_type=model_type,
            model_name=model_name,
            training_data=processed_data,
            **config.get('training', {})
        )
        
        # Record metrics
        await monitor.record_metric({
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'model_type': model_type,
            'metrics': training_metrics
        })
        
        # Save model
        save_path = Path("storage/models") / model_name
        await model_system.save(model_name, str(save_path))
        
        print(f"Model trained and saved to {save_path}")
        print("\nTraining metrics:")
        for metric, value in training_metrics.items():
            print(f"- {metric}: {value}")
            
    finally:
        # Cleanup
        data_system.cleanup()
        model_system.cleanup()
        monitor.cleanup()

def load_training_data(data_path: str) -> Dict[str, Any]:
    """Load training data from file.
    
    Args:
        data_path: Path to training data file
        
    Returns:
        Training data
    """
    with open(data_path) as f:
        if data_path.endswith('.json'):
            return json.load(f)
        return yaml.safe_load(f)

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train a NeuralFlow model")
    parser.add_argument(
        "--model-type",
        required=True,
        help="Type of model to train"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Name of the model"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    try:
        await train_model(args.model_type, args.model_name, args.data_path, config)
    except Exception as e:
        print(f"Error training model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
