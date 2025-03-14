"""
Memory registry for NeuralFlow.
"""

from typing import Dict, Any, Optional
import logging
from .unified_memory import UnifiedMemory

logger = logging.getLogger(__name__)

class MemoryRegistry:
    """Registry for managing memory instances."""
    
    _instance = None
    _memories: Dict[str, UnifiedMemory] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryRegistry, cls).__new__(cls)
        return cls._instance
    
    def register_memory(
        self,
        name: str,
        memory: UnifiedMemory,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a memory instance.
        
        Args:
            name: Memory name
            memory: Memory instance
            config: Optional memory configuration
        """
        if name in self._memories:
            logger.warning(f"Memory already registered: {name}")
            return
        
        self._memories[name] = memory
        if config:
            memory.update_config(config)
        
        logger.info(f"Memory registered: {name}")
    
    def get_memory(self, name: str) -> Optional[UnifiedMemory]:
        """Get a memory instance by name.
        
        Args:
            name: Memory name
            
        Returns:
            Optional[UnifiedMemory]: Memory if found
        """
        return self._memories.get(name)
    
    def list_memories(self) -> Dict[str, Dict[str, Any]]:
        """List all registered memories.
        
        Returns:
            Dict[str, Dict[str, Any]]: Memory information
        """
        return {
            name: memory.get_memory_info()
            for name, memory in self._memories.items()
        }
    
    def cleanup(self):
        """Clean up all memories."""
        for memory in self._memories.values():
            memory.cleanup()
        self._memories = {}
