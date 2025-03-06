import os
import logging
import asyncio
import gc
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import pandas as pd
from config.config import Config

class TrainingManager:
    """Manages model training and fine-tuning operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.training_data_dir = config.TRAINING_DATA_DIR
        self.checkpoints_dir = config.CHECKPOINTS_DIR
        self.last_processed_timestamp = None
        
        # Create necessary directories
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Load last processed timestamp
        self._load_last_processed_timestamp()
        
    def _load_last_processed_timestamp(self):
        """Load the last processed timestamp from the log file."""
        try:
            if self.config.LAST_PROCESSED_TIMESTAMP_LOG.exists():
                with open(self.config.LAST_PROCESSED_TIMESTAMP_LOG, 'r') as f:
                    self.last_processed_timestamp = datetime.fromisoformat(f.read().strip())
        except Exception as e:
            self.logger.error(f"Error loading last processed timestamp: {e}")
            self.last_processed_timestamp = None
            
    def _save_last_processed_timestamp(self):
        """Save the current timestamp to the log file."""
        try:
            current_time = datetime.now()
            with open(self.config.LAST_PROCESSED_TIMESTAMP_LOG, 'w') as f:
                f.write(current_time.isoformat())
            self.last_processed_timestamp = current_time
        except Exception as e:
            self.logger.error(f"Error saving last processed timestamp: {e}")
            
    async def prepare_training_data(self, memory_manager) -> Optional[Dataset]:
        """Prepare training data from memory and documents."""
        try:
            # Get interactions from memory
            interactions = memory_manager.get_interactions(
                limit=self.config.MID_TERM_LIMIT
            )
            
            if not interactions:
                self.logger.warning("No training data available")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(interactions)
            
            # Filter by timestamp if we have a last processed time
            if self.last_processed_timestamp:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df[df['timestamp'] > self.last_processed_timestamp]
                
            if df.empty:
                self.logger.info("No new training data since last processing")
                return None
                
            # Prepare text data
            texts = []
            for _, row in df.iterrows():
                # Combine query and response
                text = f"User: {row['user_query']}\nAssistant: {row['response']}\n"
                texts.append(text)
                
            # Create dataset
            dataset = Dataset.from_dict({"text": texts})
            
            # Save processed data
            processed_data_path = self.training_data_dir / "processed_data.parquet"
            dataset.to_parquet(processed_data_path)
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return None
            
    async def train_model(self, dataset: Dataset, model_name: str):
        """Train or fine-tune the model on the prepared dataset."""
        try:
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision for memory efficiency
                device_map="auto"  # Automatically handle model placement
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Prepare training arguments
            training_args = TrainingArguments(
                output_dir=str(self.checkpoints_dir),
                num_train_epochs=self.config.TRAINING_ARGS["num_train_epochs"],
                per_device_train_batch_size=self.config.TRAINING_ARGS["per_device_train_batch_size"],
                gradient_accumulation_steps=4,  # Accumulate gradients for memory efficiency
                learning_rate=2e-5,
                weight_decay=self.config.TRAINING_ARGS["weight_decay"],
                logging_dir=str(self.config.LOGS_DIR / "training"),
                logging_steps=self.config.TRAINING_ARGS["logging_steps"],
                save_strategy="epoch",
                save_total_limit=2,  # Keep only last 2 checkpoints
                fp16=True,  # Use mixed precision training
                optim="adamw_torch"
            )
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator
            )
            
            # Train the model
            trainer.train()
            
            # Save the final model
            final_model_path = self.checkpoints_dir / "final_model"
            trainer.save_model(str(final_model_path))
            tokenizer.save_pretrained(str(final_model_path))
            
            # Update last processed timestamp
            self._save_last_processed_timestamp()
            
            # Clean up
            del model
            del trainer
            gc.collect()
            
            return final_model_path
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return None
            
    async def run_training_cycle(self, memory_manager):
        """Run a complete training cycle."""
        try:
            # Check if we should run training
            if not self.config.AUTO_UPDATE_MODELS:
                self.logger.info("Auto model updates are disabled")
                return
                
            # Prepare training data
            dataset = await self.prepare_training_data(memory_manager)
            if not dataset:
                return
                
            # Train model
            model_path = await self.train_model(dataset, self.config.LLM_MODEL)
            if model_path:
                self.logger.info(f"Model training completed successfully. Saved to {model_path}")
            else:
                self.logger.error("Model training failed")
                
        except Exception as e:
            self.logger.error(f"Error in training cycle: {e}")
            
    async def start_training_monitor(self, memory_manager):
        """Start the training monitor loop."""
        while True:
            try:
                await self.run_training_cycle(memory_manager)
                # Wait for next training interval
                await asyncio.sleep(self.config.MODEL_UPDATE_INTERVAL)
            except Exception as e:
                self.logger.error(f"Error in training monitor: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying 