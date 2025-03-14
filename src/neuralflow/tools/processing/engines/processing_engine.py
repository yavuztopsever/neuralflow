"""
Processing engine for NeuralFlow.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import asyncio

from ..core.unified_processor import UnifiedProcessor, ProcessorConfig
from ...services.core.unified_service import UnifiedService

logger = logging.getLogger(__name__)

class ProcessingEngine:
    """Engine for coordinating processing tasks."""
    
    def __init__(
        self,
        config: Optional[ProcessorConfig] = None,
        service: Optional[UnifiedService] = None
    ):
        """Initialize processing engine.
        
        Args:
            config: Optional processor configuration
            service: Optional unified service
        """
        self.processor = UnifiedProcessor(config, service)
        self.tasks = {}
    
    async def submit_task(
        self,
        task_id: str,
        data: Dict[str, Any],
        processor_type: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a processing task.
        
        Args:
            task_id: Task identifier
            data: Data to process
            processor_type: Type of processor to use
            context: Optional processing context
            
        Returns:
            str: Task ID
        """
        task = asyncio.create_task(
            self.processor.process(data, processor_type, context)
        )
        self.tasks[task_id] = task
        return task_id
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Optional[Dict[str, Any]]: Task result if completed
        """
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        if task.done():
            result = await task
            del self.tasks[task_id]
            return result
        
        return {"status": "pending"}
    
    def cleanup(self):
        """Clean up resources."""
        for task in self.tasks.values():
            task.cancel()
        self.processor.cleanup()
        self.tasks = {}
