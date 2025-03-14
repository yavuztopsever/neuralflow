"""
Memory manager for NeuralFlow.
"""

from typing import Dict, Any, Optional, List
import logging
from .registry import MemoryRegistry
from .unified_memory import UnifiedMemory, MemoryConfig

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manager for coordinating memory instances."""
    
    def __init__(self):
        """Initialize memory manager."""
        self.registry = MemoryRegistry()
    
    def create_memory(
        self,
        name: str,
        config: Optional[MemoryConfig] = None
    ) -> UnifiedMemory:
        """Create and register a new memory instance.
        
        Args:
            name: Memory name
            config: Optional memory configuration
            
        Returns:
            UnifiedMemory: Created memory instance
        """
        memory = UnifiedMemory(config)
        self.registry.register_memory(name, memory)
        return memory
    
    def get_memory(self, name: str) -> Optional[UnifiedMemory]:
        """Get a memory instance by name.
        
        Args:
            name: Memory name
            
        Returns:
            Optional[UnifiedMemory]: Memory if found
        """
        return self.registry.get_memory(name)
    
    def list_memories(self) -> Dict[str, Dict[str, Any]]:
        """List all memory instances.
        
        Returns:
            Dict[str, Dict[str, Any]]: Memory information
        """
        return self.registry.list_memories()
    
    def cleanup(self):
        """Clean up all memory instances."""
        self.registry.cleanup()
