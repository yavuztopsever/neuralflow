"""Data science pipeline module implementing industry standard steps."""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path
import shutil

from .validation import DataValidator
from .augmentation import DataAugmentor

logger = logging.getLogger(__name__)

class DataSciencePipeline:
    """Implements industry standard data science pipeline steps."""
    
    def __init__(
        self,
        output_dir: str = "data_science_output",
        n_splits: int = 5,
        random_state: int = 42,
        models_root_dir: str = "/Volumes/HomeX/yavuztopsever/neuralflow/models"
    ):
        """Initialize the data science pipeline.
        
        Args:
            output_dir: Directory to save outputs
            n_splits: Number of cross-validation splits
            random_state: Random state for reproducibility
            models_root_dir: Root directory for models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_root_dir = Path(models_root_dir)
        self.models_root_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_splits = n_splits
        self.random_state = random_state
        
        # Initialize components
        self.validator = DataValidator()
        self.data_augmentor = DataAugmentor()
        
        # Initialize cross-validation
        self.cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
    
    def run_pipeline(
        self,
        data: Dict[str, Any],
        model_type: str = "all",
        session_id: str = None
    ) -> Dict[str, Any]:
        """Run the complete data science pipeline.
        
        Args:
            data: Input data
            model_type: Type of model to train
            session_id: Training session ID
            
        Returns:
            Dictionary containing pipeline results
        """
        pipeline_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_type": model_type,
            "session_id": session_id,
            "steps": {}
        }
        
        try:
            # 1. Data Validation and Quality Analysis
            validation_results = self._run_validation(data, model_type)
            pipeline_results["steps"]["validation"] = validation_results
            
            if not all(result.is_valid for result in validation_results.values()):
                raise ValueError("Data validation failed")
            
            # 2. Data Cleaning
            cleaned_data = self._run_cleaning(data, model_type)
            pipeline_results["steps"]["cleaning"] = {
                "original_size": len(data),
                "cleaned_size": len(cleaned_data),
                "removed_samples": len(data) - len(cleaned_data)
            }
            
            # 3. Data Augmentation
            augmented_data = self._run_augmentation(cleaned_data, model_type)
            pipeline_results["steps"]["augmentation"] = {
                "original_size": len(cleaned_data),
                "augmented_size": len(augmented_data),
                "new_samples": len(augmented_data) - len(cleaned_data)
            }
            
            # 4. Feature Engineering
            engineered_data = self._run_feature_engineering(augmented_data, model_type)
            pipeline_results["steps"]["feature_engineering"] = {
                "features": list(engineered_data.keys()),
                "feature_stats": self._compute_feature_stats(engineered_data)
            }
            
            # 5. Cross-Validation
            cv_results = self._run_cross_validation(engineered_data, model_type)
            pipeline_results["steps"]["cross_validation"] = cv_results
            
            # 6. Model Training Configuration
            training_config = self._prepare_training_config(engineered_data, model_type)
            pipeline_results["steps"]["training"] = {
                "config": training_config
            }
            
            # 7. Generate Reports
            self._generate_reports(pipeline_results)
            
            # 8. Update Root Models
            if session_id:
                self._update_root_models(session_id, model_type)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def _run_validation(
        self,
        data: Dict[str, Any],
        model_type: str
    ) -> Dict[str, Any]:
        """Run data validation.
        
        Args:
            data: Input data
            model_type: Type of model to train
            
        Returns:
            Validation results
        """
        validation_results = {}
        
        if model_type in ["all", "embedding"]:
            validation_results["embedding"] = self.validator.validate_embedding_data(
                data["embedding"]["texts"],
                data["embedding"]["embeddings"]
            )
        
        if model_type in ["all", "llm"]:
            validation_results["llm"] = self.validator.validate_llm_data(
                data["llm"]["examples"]
            )
        
        return validation_results
    
    def _run_cleaning(
        self,
        data: Dict[str, Any],
        model_type: str
    ) -> Dict[str, Any]:
        """Run data cleaning.
        
        Args:
            data: Input data
            model_type: Type of model to train
            
        Returns:
            Cleaned data
        """
        cleaned_data = {}
        
        if model_type in ["all", "embedding"]:
            cleaned_texts, cleaned_embeddings = self.validator.clean_embedding_data(
                data["embedding"]["texts"],
                data["embedding"]["embeddings"]
            )
            cleaned_data["embedding"] = {
                "texts": cleaned_texts,
                "embeddings": cleaned_embeddings
            }
        
        if model_type in ["all", "llm"]:
            cleaned_data["llm"] = {
                "examples": self.validator.clean_llm_data(data["llm"]["examples"])
            }
        
        return cleaned_data
    
    def _run_augmentation(
        self,
        data: Dict[str, Any],
        model_type: str
    ) -> Dict[str, Any]:
        """Run data augmentation.
        
        Args:
            data: Input data
            model_type: Type of model to train
            
        Returns:
            Augmented data
        """
        augmented_data = {}
        
        if model_type in ["all", "embedding"]:
            augmented_data["embedding"] = self.data_augmentor.augment_embedding_data(
                data["embedding"]["texts"],
                data["embedding"]["embeddings"]
            )
        
        if model_type in ["all", "llm"]:
            augmented_data["llm"] = self.data_augmentor.augment_llm_data(
                data["llm"]["examples"]
            )
        
        return augmented_data
    
    def _run_feature_engineering(
        self,
        data: Dict[str, Any],
        model_type: str
    ) -> Dict[str, Any]:
        """Run feature engineering.
        
        Args:
            data: Input data
            model_type: Type of model to train
            
        Returns:
            Engineered features
        """
        engineered_data = {}
        
        if model_type in ["all", "embedding"]:
            engineered_data["embedding"] = {
                "texts": data["embedding"]["texts"],
                "embeddings": data["embedding"]["embeddings"]
            }
        
        if model_type in ["all", "llm"]:
            engineered_data["llm"] = {
                "examples": data["llm"]["examples"]
            }
        
        return engineered_data
    
    def _run_cross_validation(
        self,
        data: Dict[str, Any],
        model_type: str
    ) -> Dict[str, Any]:
        """Run cross-validation.
        
        Args:
            data: Input data
            model_type: Type of model to train
            
        Returns:
            Cross-validation results
        """
        cv_results = {}
        
        if model_type in ["all", "embedding"]:
            # Split data into training and validation sets
            texts = data["embedding"]["texts"]
            embeddings = data["embedding"]["embeddings"]
            
            # Use stratified k-fold for balanced splits
            splits = list(self.cv.split(texts, embeddings))
            train_idx, val_idx = splits[0]  # Use first split for now
            
            cv_results["embedding"] = {
                "training_data": {
                    "texts": [texts[i] for i in train_idx],
                    "embeddings": [embeddings[i] for i in train_idx]
                },
                "validation_data": {
                    "texts": [texts[i] for i in val_idx],
                    "embeddings": [embeddings[i] for i in val_idx]
                }
            }
        
        if model_type in ["all", "llm"]:
            # Split data into training and validation sets
            examples = data["llm"]["examples"]
            
            # Use stratified k-fold for balanced splits
            splits = list(self.cv.split(examples, [0] * len(examples)))  # Dummy labels
            train_idx, val_idx = splits[0]  # Use first split for now
            
            cv_results["llm"] = {
                "training_data": {
                    "examples": [examples[i] for i in train_idx]
                },
                "validation_data": {
                    "examples": [examples[i] for i in val_idx]
                }
            }
        
        return cv_results
    
    def _prepare_training_config(
        self,
        data: Dict[str, Any],
        model_type: str
    ) -> Dict[str, Any]:
        """Prepare training configuration.
        
        Args:
            data: Input data
            model_type: Type of model to train
            
        Returns:
            Training configuration
        """
        config = {}
        
        if model_type in ["all", "embedding"]:
            config["embedding"] = {
                "batch_size": min(32, len(data["embedding"]["texts"])),
                "learning_rate": 2e-5,
                "num_epochs": 3,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "use_mixed_precision": True,
                "max_grad_norm": 1.0,
                "early_stopping_patience": 3,
                "logging_steps": 100
            }
        
        if model_type in ["all", "llm"]:
            config["llm"] = {
                "batch_size": min(8, len(data["llm"]["examples"])),
                "learning_rate": 1e-5,
                "num_epochs": 3,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "use_mixed_precision": True,
                "max_grad_norm": 1.0,
                "early_stopping_patience": 3,
                "logging_steps": 100
            }
        
        return config
    
    def _compute_feature_stats(self, engineered_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute feature statistics.
        
        Args:
            engineered_data: Engineered features
            
        Returns:
            Feature statistics
        """
        feature_stats = {}
        
        for model_type, data in engineered_data.items():
            if model_type == "embedding":
                feature_stats["text_length"] = {
                    "mean": np.mean([len(text.split()) for text in data["texts"]]),
                    "std": np.std([len(text.split()) for text in data["texts"]]),
                    "max": max([len(text.split()) for text in data["texts"]]),
                    "min": min([len(text.split()) for text in data["texts"]])
                }
            elif model_type == "llm":
                feature_stats["input_length"] = {
                    "mean": np.mean([len(example["input"].split()) for example in data["examples"]]),
                    "std": np.std([len(example["input"].split()) for example in data["examples"]]),
                    "max": max([len(example["input"].split()) for example in data["examples"]]),
                    "min": min([len(example["input"].split()) for example in data["examples"]])
                }
                feature_stats["output_length"] = {
                    "mean": np.mean([len(example["output"].split()) for example in data["examples"]]),
                    "std": np.std([len(example["output"].split()) for example in data["examples"]]),
                    "max": max([len(example["output"].split()) for example in data["examples"]]),
                    "min": min([len(example["output"].split()) for example in data["examples"]])
                }
        
        return feature_stats
    
    def _generate_reports(self, pipeline_results: Dict[str, Any]):
        """Generate reports from pipeline results.
        
        Args:
            pipeline_results: Pipeline results
        """
        # Save pipeline results
        results_file = self.output_dir / "pipeline_results.json"
        with open(results_file, "w") as f:
            json.dump(pipeline_results, f, indent=2)
        
        # Generate plots
        self._plot_validation_results(pipeline_results["steps"]["validation"])
        self._plot_feature_importance(pipeline_results["steps"]["feature_engineering"])
        self._plot_cross_validation(pipeline_results["steps"]["cross_validation"])
    
    def _plot_validation_results(self, validation_results: Dict[str, Any]):
        """Plot validation results.
        
        Args:
            validation_results: Validation results
        """
        plt.figure(figsize=(10, 6))
        
        # Plot quality scores
        for model_type, result in validation_results.items():
            plt.bar(model_type, result.metrics["quality_score"])
        
        plt.title("Data Quality Scores")
        plt.xlabel("Model Type")
        plt.ylabel("Quality Score")
        plt.ylim(0, 1)
        plt.savefig(self.output_dir / "validation_results.png")
        plt.close()
    
    def _plot_feature_importance(self, feature_engineering: Dict[str, Any]):
        """Plot feature importance.
        
        Args:
            feature_engineering: Feature engineering results
        """
        plt.figure(figsize=(10, 6))
        
        # Plot feature statistics
        for feature, stats in feature_engineering["feature_stats"].items():
            plt.bar(feature, stats["mean"])
        
        plt.title("Feature Statistics")
        plt.xlabel("Feature")
        plt.ylabel("Mean Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance.png")
        plt.close()
    
    def _plot_cross_validation(self, cross_validation: Dict[str, Any]):
        """Plot cross-validation results.
        
        Args:
            cross_validation: Cross-validation results
        """
        plt.figure(figsize=(10, 6))
        
        # Plot training and validation sizes
        for model_type, results in cross_validation.items():
            train_size = len(results["training_data"]["texts"])
            val_size = len(results["validation_data"]["texts"])
            
            plt.bar(f"{model_type}_train", train_size)
            plt.bar(f"{model_type}_val", val_size)
        
        plt.title("Cross-Validation Split")
        plt.xlabel("Dataset")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "cross_validation.png")
        plt.close()
    
    def _update_root_models(self, session_id: str, model_type: str):
        """Update root models after training.
        
        Args:
            session_id: Training session ID
            model_type: Type of model to train
        """
        session_dir = self.output_dir / session_id
        if not session_dir.exists():
            return
        
        # Update embedding model
        if model_type in ["all", "embedding"]:
            embedding_model_dir = session_dir / "embedding_model"
            if embedding_model_dir.exists():
                root_embedding_dir = self.models_root_dir / "embedding"
                root_embedding_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(embedding_model_dir, root_embedding_dir, dirs_exist_ok=True)
        
        # Update LLM model
        if model_type in ["all", "llm"]:
            llm_model_dir = session_dir / "llm_model"
            if llm_model_dir.exists():
                root_llm_dir = self.models_root_dir / "llm"
                root_llm_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(llm_model_dir, root_llm_dir, dirs_exist_ok=True)
    
    def cleanup(self):
        """Clean up resources."""
        # Clean up any resources used by the pipeline
        pass 