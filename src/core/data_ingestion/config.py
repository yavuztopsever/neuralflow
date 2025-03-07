from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import json

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion module."""
    
    # Database settings
    db_url: Optional[str] = None
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800
    
    # Data processing settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_sequence_length: int = 512
    batch_size: int = 32
    device: str = "cuda"  # Will be updated based on availability
    
    # Storage settings
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    cache_dir: Path = Path("data/cache")
    
    # Memory settings
    memory_cleanup_days: int = 30
    max_memory_items: int = 1000
    
    # Processing settings
    default_data_types: List[str] = None  # Will be set in __post_init__
    
    # Training settings
    min_session_messages: int = 10  # Minimum number of messages required for training
    max_session_messages: int = 1000  # Maximum number of messages to process
    min_training_examples: int = 5  # Minimum number of training examples required
    validation_split: float = 0.2  # Validation data split ratio
    
    # Advanced training settings
    training_batch_size: int = 32
    training_learning_rate: float = 2e-5
    training_num_epochs: int = 3
    training_warmup_steps: int = 100
    training_weight_decay: float = 0.01
    
    # Early stopping settings
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
    
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"  # Options: linear, cosine, step
    lr_scheduler_warmup_ratio: float = 0.1
    lr_scheduler_cycle_momentum: bool = True
    
    # Gradient settings
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Mixed precision training
    use_mixed_precision: bool = True
    fp16_opt_level: str = "O1"
    
    # Checkpointing
    save_strategy: str = "epoch"  # Options: epoch, steps
    save_steps: int = 1000
    save_total_limit: int = 3  # Maximum number of checkpoints to keep
    
    # Monitoring
    logging_steps: int = 100
    evaluation_strategy: str = "epoch"  # Options: epoch, steps
    eval_steps: int = 1000
    eval_accumulation_steps: int = 1
    
    # Resource management
    max_gpu_memory_fraction: float = 0.9
    auto_batch_size: bool = True
    min_batch_size: int = 8
    max_batch_size: int = 128
    
    # Session settings
    session_timeout: int = 3600  # Session timeout in seconds
    max_session_duration: int = 86400  # Maximum session duration in seconds (24 hours)
    min_session_duration: int = 300
    
    # New fields
    models_root_dir: Path = Path("/Volumes/HomeX/yavuztopsever/neuralflow/models")
    llm_model: str = "deepseek-1.5b"
    
    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.default_data_types is None:
            self.default_data_types = ["embedding", "llm"]
        
        # Update device based on availability
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        
        # Validate numeric parameters
        assert 0 < self.validation_split < 1, "validation_split must be between 0 and 1"
        assert 0 < self.max_gpu_memory_fraction <= 1, "max_gpu_memory_fraction must be between 0 and 1"
        assert self.min_batch_size <= self.max_batch_size, "min_batch_size must be less than or equal to max_batch_size"
        
        # Create required directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_root_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataIngestionConfig":
        """Create a DataIngestionConfig instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            DataIngestionConfig instance
        """
        # Convert path strings to Path objects
        for key in ["data_dir", "model_dir", "cache_dir", "models_root_dir"]:
            if key in config_dict and isinstance(config_dict[key], str):
                config_dict[key] = Path(config_dict[key])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary containing configuration values
        """
        return {
            "db_url": self.db_url,
            "db_pool_size": self.db_pool_size,
            "db_max_overflow": self.db_max_overflow,
            "db_pool_timeout": self.db_pool_timeout,
            "db_pool_recycle": self.db_pool_recycle,
            "embedding_model": self.embedding_model,
            "max_sequence_length": self.max_sequence_length,
            "batch_size": self.batch_size,
            "device": self.device,
            "data_dir": str(self.data_dir),
            "model_dir": str(self.model_dir),
            "cache_dir": str(self.cache_dir),
            "memory_cleanup_days": self.memory_cleanup_days,
            "max_memory_items": self.max_memory_items,
            "default_data_types": self.default_data_types,
            "min_session_messages": self.min_session_messages,
            "max_session_messages": self.max_session_messages,
            "min_training_examples": self.min_training_examples,
            "validation_split": self.validation_split,
            "training_batch_size": self.training_batch_size,
            "training_learning_rate": self.training_learning_rate,
            "training_num_epochs": self.training_num_epochs,
            "training_warmup_steps": self.training_warmup_steps,
            "training_weight_decay": self.training_weight_decay,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "lr_scheduler_type": self.lr_scheduler_type,
            "lr_scheduler_warmup_ratio": self.lr_scheduler_warmup_ratio,
            "lr_scheduler_cycle_momentum": self.lr_scheduler_cycle_momentum,
            "max_grad_norm": self.max_grad_norm,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "use_mixed_precision": self.use_mixed_precision,
            "fp16_opt_level": self.fp16_opt_level,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "eval_accumulation_steps": self.eval_accumulation_steps,
            "max_gpu_memory_fraction": self.max_gpu_memory_fraction,
            "auto_batch_size": self.auto_batch_size,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "session_timeout": self.session_timeout,
            "max_session_duration": self.max_session_duration,
            "min_session_duration": self.min_session_duration,
            "models_root_dir": str(self.models_root_dir),
            "llm_model": self.llm_model
        }
    
    def save(self, path: str):
        """Save configuration to file.
        
        Args:
            path: Path to save configuration
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "DataIngestionConfig":
        """Load configuration from file.
        
        Args:
            path: Path to load configuration from
            
        Returns:
            DataIngestionConfig instance
        """
        with open(path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict) 