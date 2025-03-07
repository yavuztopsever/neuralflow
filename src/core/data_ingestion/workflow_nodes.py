from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import uuid
import asyncio
from pathlib import Path
import shutil

from ...graph.workflow_nodes import WorkflowNode, NodeConfig
from ..database.manager import DatabaseManager
from ..database.models.training import DataType
from .training.pipeline import DataSciencePipeline
from ..tools.memory.memory_manager import MemoryManager
from .processors.data_processor import DataProcessor
from .training.workflow_nodes import ModelTrainingNode

logger = logging.getLogger(__name__)

class DataIngestionNode(WorkflowNode):
    """Node for ingesting and processing session data."""
    
    def __init__(self, node_id: str, config: NodeConfig):
        """Initialize the data ingestion node.
        
        Args:
            node_id: Unique identifier for the node
            config: Node configuration
        """
        super().__init__(node_id, config)
        self.memory_manager = MemoryManager()
        self.db_manager = DatabaseManager()
        self.data_processor = DataProcessor(self.db_manager)
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data ingestion logic.
        
        Args:
            context: Workflow context containing session data
            
        Returns:
            Updated context with processed data
        """
        if not self.validate_input(context):
            raise ValueError("Invalid input data for data ingestion")
        
        try:
            # Get session data from context
            session_data = context.get("session_data", {})
            
            # Store session data in database
            stored_session = self.db_manager.store_session_data(session_data)
            
            # Process session data into training data
            processed_data = self.data_processor.process_session_data(
                stored_session,
                data_types=[DataType.EMBEDDING, DataType.FINETUNING]
            )
            
            # Store processed data
            self.data_processor.store_processed_data(
                session_id=stored_session.session_id,
                processed_data=processed_data,
                model_name=context.get("model_name", "default")
            )
            
            # Update memory with processed data
            self._update_memory(processed_data)
            
            # Prepare output
            output = {
                "session_id": stored_session.session_id,
                "processed_data": processed_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if not self.validate_output(output):
                raise ValueError("Invalid output format for data ingestion")
            
            return output
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {str(e)}")
            raise
    
    def _update_memory(self, processed_data: Dict[str, Any]) -> None:
        """Update memory with processed data.
        
        Args:
            processed_data: Dictionary containing processed data
        """
        # Add embedding data to memory
        if "embedding" in processed_data:
            embedding_data = processed_data["embedding"]
            for text, embedding in zip(embedding_data["texts"], embedding_data["embeddings"]):
                self.memory_manager.add_to_memory(
                    content=text,
                    memory_type="long_term",
                    metadata={"embedding": embedding}
                )
        
        # Add finetuning data to memory
        if "finetuning" in processed_data:
            finetuning_data = processed_data["finetuning"]
            for example in finetuning_data["examples"]:
                self.memory_manager.add_to_memory(
                    content=f"{example['input']}\n{example['output']}",
                    memory_type="mid_term",
                    metadata={"type": "finetuning_example"}
                )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for data ingestion.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        required_fields = ["session_data", "model_name"]
        return all(field in input_data for field in required_fields)
    
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """Validate output data from data ingestion.
        
        Args:
            output_data: Output data to validate
            
        Returns:
            True if output is valid, False otherwise
        """
        required_fields = ["session_id", "processed_data", "timestamp"]
        return all(field in output_data for field in required_fields)
    
    def cleanup(self):
        """Clean up resources."""
        self.data_processor.cleanup()

class SessionEndNode(WorkflowNode):
    """Node for handling session end events and triggering data processing."""
    
    def __init__(self, node_id: str, config: NodeConfig):
        """Initialize the session end node.
        
        Args:
            node_id: Unique identifier for the node
            config: Node configuration
        """
        super().__init__(node_id, config)
        self.memory_manager = MemoryManager()
        self.db_manager = DatabaseManager()
        self.data_processor = DataProcessor(self.db_manager)
        
        # Initialize training node
        self.training_node = ModelTrainingNode(
            config=NodeConfig(
                model_dir=str(config.get("model_dir", "models")),
                embedding_model=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                llm_model=config.get("llm_model", "deepseek-1.5b")
            ),
            db_manager=self.db_manager
        )
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute session end logic.
        
        Args:
            context: Workflow context containing session data
            
        Returns:
            Updated context with processing results
        """
        if not self.validate_input(context):
            raise ValueError("Invalid input data for session end")
        
        try:
            # Get session data from context
            session_data = context.get("session_data", {})
            session_id = session_data.get("session_id")
            
            if not session_id:
                raise ValueError("Session ID is required")
            
            # Store final session data in database
            stored_session = self.db_manager.store_session_data(session_data)
            
            # Process session data into training data
            processed_data = self.data_processor.process_session_data(
                stored_session,
                data_types=[DataType.EMBEDDING, DataType.FINETUNING]
            )
            
            # Store processed data
            self.data_processor.store_processed_data(
                session_id=stored_session.session_id,
                processed_data=processed_data,
                model_name=context.get("model_name", "default")
            )
            
            # Trigger model training asynchronously
            asyncio.create_task(self._trigger_training(session_id))
            
            # Update memory with processed data
            self._update_memory(processed_data)
            
            # Prepare output
            output = {
                "session_id": session_id,
                "processed_data": processed_data,
                "timestamp": datetime.utcnow().isoformat(),
                "training_triggered": True
            }
            
            if not self.validate_output(output):
                raise ValueError("Invalid output format for session end")
            
            return output
            
        except Exception as e:
            logger.error(f"Session end processing failed: {str(e)}")
            raise
    
    async def _trigger_training(self, session_id: str) -> None:
        """Trigger model training for the session.
        
        Args:
            session_id: ID of the session to train on
        """
        try:
            # Prepare training context
            context = {
                "session_id": session_id,
                "model_type": "all"
            }
            
            # Execute training
            result = await self.training_node.execute(context)
            
            # Log training results
            logger.info(f"Training completed for session {session_id}")
            logger.info(f"Results: {result}")
            
        except Exception as e:
            logger.error(f"Training failed for session {session_id}: {str(e)}")
    
    def _update_memory(self, processed_data: Dict[str, Any]) -> None:
        """Update memory with processed data.
        
        Args:
            processed_data: Processed training data
        """
        # Store in long-term memory
        for data_type, data in processed_data.items():
            self.memory_manager.add_to_memory(
                content=str(data),
                memory_type="long_term",
                metadata={
                    "data_type": data_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for session end.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        required_fields = ["session_data"]
        return all(field in input_data for field in required_fields)
    
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """Validate output data from session end.
        
        Args:
            output_data: Output data to validate
            
        Returns:
            True if output is valid, False otherwise
        """
        required_fields = ["session_id", "processed_data", "timestamp", "training_triggered"]
        return all(field in output_data for field in required_fields)
    
    def cleanup(self):
        """Clean up resources."""
        self.data_processor.cleanup()
        self.training_node.cleanup()

class ModelTrainingNode(WorkflowNode):
    """Node for model training."""
    
    def __init__(
        self,
        config: NodeConfig,
        db_manager: DatabaseManager,
        output_dir: str = "training_output",
        models_root_dir: str = "/Volumes/HomeX/yavuztopsever/neuralflow/models"
    ):
        """Initialize the model training node.
        
        Args:
            config: Node configuration
            db_manager: Database manager
            output_dir: Directory to save outputs
            models_root_dir: Root directory for models
        """
        super().__init__(config)
        self.db_manager = db_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_root_dir = Path(models_root_dir)
        self.models_root_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data science pipeline
        self.pipeline = DataSciencePipeline(
            output_dir=str(self.output_dir),
            models_root_dir=str(self.models_root_dir)
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the model training node.
        
        Args:
            input_data: Input data
            
        Returns:
            Training results
        """
        try:
            # Validate input data
            if not self._validate_input(input_data):
                raise ValueError("Invalid input data")
            
            # Get training data
            training_data = self._get_training_data(
                input_data["session_id"],
                input_data["model_type"]
            )
            
            # Run data science pipeline
            pipeline_results = self.pipeline.run_pipeline(
                data=training_data,
                model_type=input_data["model_type"],
                session_id=input_data["session_id"]
            )
            
            # Prepare output
            output = {
                "session_id": input_data["session_id"],
                "model_type": input_data["model_type"],
                "timestamp": datetime.utcnow().isoformat(),
                "results": pipeline_results
            }
            
            # Validate output
            if not self._validate_output(output):
                raise ValueError("Invalid output data")
            
            return output
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def _validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data.
        
        Args:
            input_data: Input data
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["session_id", "model_type"]
        return all(field in input_data for field in required_fields)
    
    def _validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate output data.
        
        Args:
            output: Output data
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["session_id", "model_type", "timestamp", "results"]
        return all(field in output for field in required_fields)
    
    def _get_training_data(
        self,
        session_id: str,
        model_type: str
    ) -> Dict[str, Any]:
        """Get training data from database.
        
        Args:
            session_id: Training session ID
            model_type: Type of model to train
            
        Returns:
            Training data
        """
        training_data = {}
        
        if model_type in ["all", "embedding"]:
            # Get embedding data
            embedding_data = self.db_manager.get_data(
                session_id=session_id,
                data_type=DataType.EMBEDDING
            )
            if embedding_data:
                training_data["embedding"] = {
                    "texts": [item["text"] for item in embedding_data],
                    "embeddings": [item["embedding"] for item in embedding_data]
                }
        
        if model_type in ["all", "llm"]:
            # Get LLM data
            llm_data = self.db_manager.get_data(
                session_id=session_id,
                data_type=DataType.LLM
            )
            if llm_data:
                training_data["llm"] = {
                    "examples": [
                        {"input": item["input"], "output": item["output"]}
                        for item in llm_data
                    ]
                }
        
        return training_data
    
    def cleanup(self):
        """Clean up resources."""
        self.pipeline.cleanup() 