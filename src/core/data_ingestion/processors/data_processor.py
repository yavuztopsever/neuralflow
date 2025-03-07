from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path

from ..database.models.training import DataType
from ..database.manager import DatabaseManager

logger = logging.getLogger(__name__)

class DataProcessor:
    """Processes session data into training data for different models."""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the data processor.
        
        Args:
            db_manager: Database manager instance
            embedding_model: Name of the embedding model to use
            device: Device to run the model on
        """
        self.db_manager = db_manager
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model).to(device)
        self.model.eval()
        
        # Create cache directory
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def process_session_data(
        self,
        session_data: Dict[str, Any],
        data_types: Optional[List[DataType]] = None
    ) -> Dict[str, Any]:
        """Process session data into training data.
        
        Args:
            session_data: Raw session data
            data_types: Optional list of data types to process
            
        Returns:
            Dictionary containing processed data for each type
        """
        if data_types is None:
            data_types = [DataType.EMBEDDING, DataType.FINETUNING]
        
        processed_data = {}
        
        try:
            # Extract and validate conversation data
            conversation_data = self._extract_conversation_data(session_data)
            
            # Process each data type
            for data_type in data_types:
                if data_type == DataType.EMBEDDING:
                    processed_data["embedding"] = self._process_embedding_data(conversation_data)
                elif data_type == DataType.FINETUNING:
                    processed_data["finetuning"] = self._process_finetuning_data(conversation_data)
                elif data_type == DataType.EVALUATION:
                    processed_data["evaluation"] = self._process_evaluation_data(conversation_data)
            
            # Validate processed data
            self._validate_processed_data(processed_data)
            
            # Cache processed data
            self._cache_processed_data(session_data["session_id"], processed_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to process session data: {str(e)}")
            raise
    
    def _extract_conversation_data(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate conversation data from session data.
        
        Args:
            session_data: Raw session data
            
        Returns:
            Validated conversation data
        """
        try:
            conversation_data = json.loads(session_data["conversation_data"])
            
            # Validate required fields
            required_fields = ["messages", "metadata"]
            if not all(field in conversation_data for field in required_fields):
                raise ValueError("Missing required fields in conversation data")
            
            # Validate message format
            for msg in conversation_data["messages"]:
                if not all(field in msg for field in ["role", "content", "timestamp"]):
                    raise ValueError("Invalid message format")
            
            return conversation_data
            
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in conversation data")
    
    def _process_embedding_data(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process conversation data into embedding training data.
        
        Args:
            conversation_data: Validated conversation data
            
        Returns:
            Processed embedding data
        """
        texts = []
        embeddings = []
        
        try:
            # Extract messages
            messages = conversation_data["messages"]
            
            # Process each message
            for msg in messages:
                text = msg["content"]
                texts.append(text)
                
                # Generate embedding
                with torch.no_grad():
                    inputs = self.tokenizer(
                        text,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                    embeddings.append(embedding.tolist())
            
            return {
                "texts": texts,
                "embeddings": embeddings
            }
            
        except Exception as e:
            logger.error(f"Failed to process embedding data: {str(e)}")
            raise
    
    def _process_finetuning_data(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process conversation data into LLM fine-tuning data.
        
        Args:
            conversation_data: Validated conversation data
            
        Returns:
            Processed fine-tuning data
        """
        examples = []
        
        try:
            # Extract messages
            messages = conversation_data["messages"]
            
            # Process message pairs
            for i in range(0, len(messages) - 1, 2):
                if i + 1 >= len(messages):
                    break
                
                user_msg = messages[i]
                assistant_msg = messages[i + 1]
                
                if user_msg["role"] != "user" or assistant_msg["role"] != "assistant":
                    continue
                
                example = {
                    "input": user_msg["content"],
                    "output": assistant_msg["content"],
                    "metadata": {
                        "timestamp": user_msg["timestamp"],
                        "session_id": conversation_data["metadata"]["session_id"]
                    }
                }
                examples.append(example)
            
            return {
                "examples": examples
            }
            
        except Exception as e:
            logger.error(f"Failed to process fine-tuning data: {str(e)}")
            raise
    
    def _process_evaluation_data(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process conversation data into evaluation data.
        
        Args:
            conversation_data: Validated conversation data
            
        Returns:
            Processed evaluation data
        """
        # Implementation for evaluation data processing
        # This can be customized based on specific evaluation needs
        return {}
    
    def _validate_processed_data(self, processed_data: Dict[str, Any]) -> None:
        """Validate processed data.
        
        Args:
            processed_data: Processed data to validate
            
        Raises:
            ValueError: If data validation fails
        """
        if "embedding" in processed_data:
            embedding_data = processed_data["embedding"]
            if not embedding_data["texts"] or not embedding_data["embeddings"]:
                raise ValueError("Empty embedding data")
            if len(embedding_data["texts"]) != len(embedding_data["embeddings"]):
                raise ValueError("Mismatched embedding data lengths")
        
        if "finetuning" in processed_data:
            finetuning_data = processed_data["finetuning"]
            if not finetuning_data["examples"]:
                raise ValueError("Empty fine-tuning data")
            for example in finetuning_data["examples"]:
                if not all(field in example for field in ["input", "output", "metadata"]):
                    raise ValueError("Invalid fine-tuning example format")
    
    def _cache_processed_data(
        self,
        session_id: str,
        processed_data: Dict[str, Any]
    ) -> None:
        """Cache processed data for future use.
        
        Args:
            session_id: ID of the session
            processed_data: Processed data to cache
        """
        try:
            cache_file = self.cache_dir / f"{session_id}_processed.json"
            with open(cache_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache processed data: {str(e)}")
    
    def store_processed_data(
        self,
        session_id: str,
        processed_data: Dict[str, Any],
        model_name: str
    ) -> None:
        """Store processed data in the database.
        
        Args:
            session_id: ID of the session
            processed_data: Dictionary containing processed data
            model_name: Name of the model
        """
        try:
            for data_type, data in processed_data.items():
                self.db_manager.store_training_data(
                    session_id=session_id,
                    data_type=DataType(data_type),
                    model_name=model_name,
                    processed_data=data
                )
        except Exception as e:
            logger.error(f"Failed to store processed data: {str(e)}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        self.model.cpu()
        del self.model
        torch.cuda.empty_cache() 