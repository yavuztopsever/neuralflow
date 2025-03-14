from typing import Dict, Any
from ..core.services.auth.auth_service import AuthService
from ..core.services.rate_limit.rate_limit_service import RateLimitService
from ..core.services.validation.validation_service import ValidationService
from ..core.services.context.context_service import ContextService
from ..core.services.response.response_service import ResponseService
from ..core.services.metrics.metrics_service import MetricsService
from ..core.services.events.event_service import EventService
from ..core.services.security.security_service import SecurityService
from ..core.services.engine.engine_service import EngineService
from ..core.services.workflow.workflow_service import WorkflowService
from ..core.services.graph.graph_service import GraphService
from ..core.services.context.context_handler import ContextHandler
from ..core.services.response.response_generation import ResponseGenerator
from ..core.tools.processing.task_processor import TaskProcessor
from ..core.tools.search.graph_search import GraphSearch
from ..core.tools.memory.memory_manager import MemoryManager
from ..core.tools.vector.vector_search import VectorSearch

class ServiceConnections:
    def __init__(self):
        # Core services
        self.auth_service = AuthService()
        self.rate_limit_service = RateLimitService()
        self.validation_service = ValidationService()
        self.context_service = ContextService()
        self.response_service = ResponseService()
        self.metrics_service = MetricsService()
        self.event_service = EventService()
        self.security_service = SecurityService()
        self.engine_service = EngineService()
        self.workflow_service = WorkflowService()
        self.graph_service = GraphService()
        
        # Specialized services
        self.context_handler = ContextHandler()
        self.response_generator = ResponseGenerator()
        self.task_processor = TaskProcessor()
        self.graph_search = GraphSearch()
        
        # Tools
        self.memory_manager = MemoryManager()
        self.vector_search = VectorSearch()
        
        # Initialize service connections
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize connections between services according to the workflow requirements."""
        # Authentication Node connections
        self.auth_service.connect(self.security_service)
        self.auth_service.connect(self.event_service)
        
        # Rate Limiting Node connections
        self.rate_limit_service.connect(self.metrics_service)
        self.rate_limit_service.connect(self.event_service)
        
        # Input Processor Node connections
        self.validation_service.connect(self.workflow_service)
        
        # Memory Manager Node connections
        self.memory_manager.connect(self.context_service)
        self.memory_manager.connect(self.context_handler)
        
        # Context Orchestrator Node connections
        self.context_service.connect(self.memory_manager)
        self.context_service.connect(self.vector_search)
        self.context_service.connect(self.context_handler)
        
        # Reasoning Engine Node connections
        self.engine_service.connect(self.task_processor)
        self.engine_service.connect(self.graph_search)
        
        # Model Manager Node connections (uses same connections as Reasoning Engine)
        self.engine_service.connect(self.task_processor)
        self.engine_service.connect(self.graph_search)
        
        # Response Assembler Node connections
        self.response_service.connect(self.response_generator)
        self.response_service.connect(self.metrics_service)
        
        # Response Manager Node connections (uses same connections as Response Assembler)
        self.response_service.connect(self.response_generator)
        self.response_service.connect(self.metrics_service)
        
        # Metrics Logging Node connections
        self.metrics_service.connect(self.event_service)
        self.metrics_service.connect(self.workflow_service)
        
        # Train Module Node connections (uses same connections as Reasoning Engine)
        self.engine_service.connect(self.task_processor)
        self.engine_service.connect(self.graph_search)
        
        # Search Nodes connections
        self.graph_search.connect(self.vector_search)
        self.graph_search.connect(self.memory_manager)
        
        # Context Processing Nodes connections
        self.context_handler.connect(self.memory_manager)
        self.context_handler.connect(self.vector_search)
        
        # Error Checking Node connections
        self.validation_service.connect(self.workflow_service)
        
        # Additional cross-service connections for better integration
        self.workflow_service.connect(self.event_service)
        self.workflow_service.connect(self.metrics_service)
        self.graph_service.connect(self.graph_search)
        self.graph_service.connect(self.vector_search)
    
    def get_service_for_node(self, node_type: str) -> Any:
        """Get the appropriate service for a given node type."""
        service_map = {
            # Core nodes
            "authentication": self.auth_service,
            "rate_limiting": self.rate_limit_service,
            "input_processor": self.validation_service,
            "memory_manager": self.memory_manager,
            "context_orchestrator": self.context_service,
            "reasoning_engine": self.engine_service,
            "model_manager": self.engine_service,
            "response_assembler": self.response_service,
            "response_manager": self.response_service,
            "metrics_logging": self.metrics_service,
            "train_module": self.engine_service,
            
            # Search nodes
            "local_document_search": self.graph_search,
            "web_search": self.graph_search,
            "rag_search": self.graph_search,
            
            # Memory nodes
            "memory_retrieval": self.memory_manager,
            
            # Context processing nodes
            "context_sufficiency": self.context_handler,
            "context_aggregation": self.context_handler,
            "context_summarization": self.context_handler,
            "emotion_context": self.context_handler,
            "context_pooling": self.context_handler,
            "dynamic_context_filtering": self.context_handler,
            
            # Error handling nodes
            "error_checking": self.validation_service,
            
            # Response nodes
            "response_delivery": self.response_service,
            
            # Memory update nodes
            "memory_update": self.memory_manager
        }
        
        if node_type not in service_map:
            raise ValueError(f"No service found for node type: {node_type}")
            
        return service_map[node_type]

# Create a singleton instance
service_connections = ServiceConnections() 