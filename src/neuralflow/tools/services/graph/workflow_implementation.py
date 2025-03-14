from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio
from .workflow_nodes import (
    NodeType, NodeConfig, EdgeConfig, WorkflowNode,
    AuthenticationNode, RateLimitingNode, InputProcessorNode,
    MemoryManagerNode, ContextOrchestratorNode, ReasoningEngineNode,
    ModelManagerNode, ResponseAssemblerNode, ResponseManagerNode,
    MetricsLoggingNode, TrainModuleNode, LocalDocumentSearchNode,
    WebSearchNode, RAGSearchNode, MemoryRetrievalNode,
    ContextSufficiencyNode, ContextAggregationNode, ContextSummarizationNode,
    EmotionContextNode, ContextPoolingNode, DynamicContextFilteringNode,
    ErrorCheckingNode, ResponseDeliveryNode, MemoryUpdateNode
)
from .service_connections import service_connections
from .tools.memory_manager import MemoryManager, MemoryThreshold
from .tools.task_queue import TaskQueue, TaskConfig

@dataclass
class WorkflowState:
    current_node: str
    context: Dict[str, Any]
    completed_nodes: List[str]
    error: Optional[str] = None
    memory_status: Optional[Dict[str, Any]] = None
    task_stats: Optional[Dict[str, Any]] = None

class WorkflowImplementation:
    def __init__(self):
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: Dict[tuple, EdgeConfig] = {}
        self.state = WorkflowState(
            current_node="",
            context={},
            completed_nodes=[]
        )
        
        # Initialize memory manager with custom thresholds
        self.memory_manager = MemoryManager(
            thresholds=MemoryThreshold(
                warning=1024,  # 1GB warning
                critical=2048,  # 2GB critical
                max_chunk_size=100  # 100MB chunks
            )
        )
        
        # Initialize task queue with custom config
        self.task_queue = TaskQueue(
            config=TaskConfig(
                max_concurrent=5,
                rate_limit=100,
                timeout=30.0,
                retry_count=3,
                retry_delay=1.0
            )
        )
        
        self._initialize_workflow()

    def _initialize_workflow(self):
        # Initialize nodes with their services
        self.nodes["auth"] = AuthenticationNode(
            "auth",
            NodeConfig(
                node_type=NodeType.AUTHENTICATION,
                config={},
                dependencies=[]
            )
        )
        
        self.nodes["rate_limit"] = RateLimitingNode(
            "rate_limit",
            NodeConfig(
                node_type=NodeType.RATE_LIMITING,
                config={},
                dependencies=["auth"]
            )
        )
        
        self.nodes["input_processor"] = InputProcessorNode(
            "input_processor",
            NodeConfig(
                node_type=NodeType.INPUT_PROCESSOR,
                config={},
                dependencies=["rate_limit"]
            )
        )
        
        self.nodes["memory_manager"] = MemoryManagerNode(
            "memory_manager",
            NodeConfig(
                node_type=NodeType.MEMORY_MANAGER,
                config={},
                dependencies=["input_processor"]
            )
        )
        
        self.nodes["context_orchestrator"] = ContextOrchestratorNode(
            "context_orchestrator",
            NodeConfig(
                node_type=NodeType.CONTEXT_ORCHESTRATOR,
                config={},
                dependencies=["memory_manager"]
            )
        )
        
        self.nodes["memory_retrieval"] = MemoryRetrievalNode(
            "memory_retrieval",
            NodeConfig(
                node_type=NodeType.MEMORY_RETRIEVAL,
                config={},
                dependencies=["context_orchestrator"]
            )
        )
        
        self.nodes["context_sufficiency"] = ContextSufficiencyNode(
            "context_sufficiency",
            NodeConfig(
                node_type=NodeType.CONTEXT_SUFFICIENCY,
                config={},
                dependencies=["memory_retrieval"]
            )
        )
        
        self.nodes["reasoning_engine"] = ReasoningEngineNode(
            "reasoning_engine",
            NodeConfig(
                node_type=NodeType.REASONING_ENGINE,
                config={},
                dependencies=["context_sufficiency"]
            )
        )
        
        self.nodes["rag_search"] = RAGSearchNode(
            "rag_search",
            NodeConfig(
                node_type=NodeType.RAG_SEARCH,
                config={},
                dependencies=["reasoning_engine"]
            )
        )
        
        self.nodes["local_document_search"] = LocalDocumentSearchNode(
            "local_document_search",
            NodeConfig(
                node_type=NodeType.LOCAL_DOCUMENT_SEARCH,
                config={},
                dependencies=["rag_search"]
            )
        )
        
        self.nodes["web_search"] = WebSearchNode(
            "web_search",
            NodeConfig(
                node_type=NodeType.WEB_SEARCH,
                config={},
                dependencies=["rag_search"]
            )
        )
        
        self.nodes["context_aggregation"] = ContextAggregationNode(
            "context_aggregation",
            NodeConfig(
                node_type=NodeType.CONTEXT_AGGREGATION,
                config={},
                dependencies=["local_document_search", "web_search", "memory_retrieval"]
            )
        )
        
        self.nodes["context_summarization"] = ContextSummarizationNode(
            "context_summarization",
            NodeConfig(
                node_type=NodeType.CONTEXT_SUMMARIZATION,
                config={},
                dependencies=["context_aggregation"]
            )
        )
        
        self.nodes["emotion_context"] = EmotionContextNode(
            "emotion_context",
            NodeConfig(
                node_type=NodeType.EMOTION_CONTEXT,
                config={},
                dependencies=["input_processor"]
            )
        )
        
        self.nodes["context_pooling"] = ContextPoolingNode(
            "context_pooling",
            NodeConfig(
                node_type=NodeType.CONTEXT_POOLING,
                config={},
                dependencies=["context_summarization", "emotion_context"]
            )
        )
        
        self.nodes["model_manager"] = ModelManagerNode(
            "model_manager",
            NodeConfig(
                node_type=NodeType.MODEL_MANAGER,
                config={},
                dependencies=["context_pooling"]
            )
        )
        
        self.nodes["dynamic_context_filtering"] = DynamicContextFilteringNode(
            "dynamic_context_filtering",
            NodeConfig(
                node_type=NodeType.DYNAMIC_CONTEXT_FILTERING,
                config={},
                dependencies=["model_manager"]
            )
        )
        
        self.nodes["error_checking"] = ErrorCheckingNode(
            "error_checking",
            NodeConfig(
                node_type=NodeType.ERROR_CHECKING,
                config={},
                dependencies=["dynamic_context_filtering"]
            )
        )
        
        self.nodes["response_assembler"] = ResponseAssemblerNode(
            "response_assembler",
            NodeConfig(
                node_type=NodeType.RESPONSE_ASSEMBLER,
                config={},
                dependencies=["error_checking"]
            )
        )
        
        self.nodes["response_manager"] = ResponseManagerNode(
            "response_manager",
            NodeConfig(
                node_type=NodeType.RESPONSE_MANAGER,
                config={},
                dependencies=["response_assembler"]
            )
        )
        
        self.nodes["memory_update"] = MemoryUpdateNode(
            "memory_update",
            NodeConfig(
                node_type=NodeType.MEMORY_UPDATE,
                config={},
                dependencies=["response_manager"]
            )
        )
        
        self.nodes["response_delivery"] = ResponseDeliveryNode(
            "response_delivery",
            NodeConfig(
                node_type=NodeType.RESPONSE_DELIVERY,
                config={},
                dependencies=["memory_update"]
            )
        )
        
        self.nodes["metrics_logging"] = MetricsLoggingNode(
            "metrics_logging",
            NodeConfig(
                node_type=NodeType.METRICS_LOGGING,
                config={},
                dependencies=["response_delivery"]
            )
        )
        
        self.nodes["train_module"] = TrainModuleNode(
            "train_module",
            NodeConfig(
                node_type=NodeType.TRAIN_MODULE,
                config={},
                dependencies=["metrics_logging"]
            )
        )
        
        # Initialize edges
        self._initialize_edges()
        
        # Initialize service connections
        service_connections._initialize_connections()

    def _initialize_edges(self):
        # Define edges between nodes
        self.edges[("auth", "rate_limit")] = EdgeConfig(
            source="auth",
            target="rate_limit",
            data_type="auth_result",
            required=True
        )
        
        self.edges[("rate_limit", "input_processor")] = EdgeConfig(
            source="rate_limit",
            target="input_processor",
            data_type="rate_limit_status",
            required=True
        )
        
        self.edges[("input_processor", "memory_manager")] = EdgeConfig(
            source="input_processor",
            target="memory_manager",
            data_type="session_data",
            required=True
        )
        
        self.edges[("memory_manager", "context_orchestrator")] = EdgeConfig(
            source="memory_manager",
            target="context_orchestrator",
            data_type="session_data_confirmation",
            required=True
        )
        
        self.edges[("context_orchestrator", "memory_retrieval")] = EdgeConfig(
            source="context_orchestrator",
            target="memory_retrieval",
            data_type="retrieval_request_parameters",
            required=True
        )
        
        self.edges[("memory_retrieval", "context_sufficiency")] = EdgeConfig(
            source="memory_retrieval",
            target="context_sufficiency",
            data_type="memory_results",
            required=True
        )
        
        self.edges[("context_sufficiency", "reasoning_engine")] = EdgeConfig(
            source="context_sufficiency",
            target="reasoning_engine",
            data_type="context_sufficiency_decision",
            required=True
        )
        
        self.edges[("reasoning_engine", "rag_search")] = EdgeConfig(
            source="reasoning_engine",
            target="rag_search",
            data_type="rag_search_decision",
            required=True
        )
        
        self.edges[("rag_search", "local_document_search")] = EdgeConfig(
            source="rag_search",
            target="local_document_search",
            data_type="search_parameters",
            required=True
        )
        
        self.edges[("rag_search", "web_search")] = EdgeConfig(
            source="rag_search",
            target="web_search",
            data_type="search_parameters",
            required=True
        )
        
        self.edges[("local_document_search", "context_aggregation")] = EdgeConfig(
            source="local_document_search",
            target="context_aggregation",
            data_type="local_document_search_results",
            required=True
        )
        
        self.edges[("web_search", "context_aggregation")] = EdgeConfig(
            source="web_search",
            target="context_aggregation",
            data_type="web_search_results",
            required=True
        )
        
        self.edges[("memory_retrieval", "context_aggregation")] = EdgeConfig(
            source="memory_retrieval",
            target="context_aggregation",
            data_type="memory_results",
            required=True
        )
        
        self.edges[("context_aggregation", "context_summarization")] = EdgeConfig(
            source="context_aggregation",
            target="context_summarization",
            data_type="aggregated_context",
            required=True
        )
        
        self.edges[("input_processor", "emotion_context")] = EdgeConfig(
            source="input_processor",
            target="emotion_context",
            data_type="user_input",
            required=True
        )
        
        self.edges[("context_summarization", "context_pooling")] = EdgeConfig(
            source="context_summarization",
            target="context_pooling",
            data_type="summarized_context",
            required=True
        )
        
        self.edges[("emotion_context", "context_pooling")] = EdgeConfig(
            source="emotion_context",
            target="context_pooling",
            data_type="emotion_context",
            required=True
        )
        
        self.edges[("context_pooling", "model_manager")] = EdgeConfig(
            source="context_pooling",
            target="model_manager",
            data_type="final_context_pool",
            required=True
        )
        
        self.edges[("model_manager", "dynamic_context_filtering")] = EdgeConfig(
            source="model_manager",
            target="dynamic_context_filtering",
            data_type="model_response",
            required=True
        )
        
        self.edges[("dynamic_context_filtering", "error_checking")] = EdgeConfig(
            source="dynamic_context_filtering",
            target="error_checking",
            data_type="filtered_context",
            required=True
        )
        
        self.edges[("error_checking", "response_assembler")] = EdgeConfig(
            source="error_checking",
            target="response_assembler",
            data_type="error_check_results",
            required=True
        )
        
        self.edges[("response_assembler", "response_manager")] = EdgeConfig(
            source="response_assembler",
            target="response_manager",
            data_type="response",
            required=True
        )
        
        self.edges[("response_manager", "memory_update")] = EdgeConfig(
            source="response_manager",
            target="memory_update",
            data_type="styled_response",
            required=True
        )
        
        self.edges[("memory_update", "response_delivery")] = EdgeConfig(
            source="memory_update",
            target="response_delivery",
            data_type="memory_update_confirmation",
            required=True
        )
        
        self.edges[("response_delivery", "metrics_logging")] = EdgeConfig(
            source="response_delivery",
            target="metrics_logging",
            data_type="user_response_delivered",
            required=True
        )
        
        self.edges[("metrics_logging", "train_module")] = EdgeConfig(
            source="metrics_logging",
            target="train_module",
            data_type="workflow_data",
            required=True
        )

    async def execute_node(self, node_id: str) -> Dict[str, Any]:
        """Execute a single node with memory management and task queuing."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        node = self.nodes[node_id]
        
        # Check memory status before execution
        memory_status = await self.memory_manager.check_memory_status()
        self.state.memory_status = memory_status
        
        if memory_status["status"] == "critical":
            # Wait for memory to be freed
            await asyncio.sleep(1)
            memory_status = await self.memory_manager.check_memory_status()
            if memory_status["status"] == "critical":
                raise MemoryError("Critical memory usage detected")

        # Execute node through task queue
        try:
            result = await self.task_queue.add_task(
                f"node_{node_id}",
                node.execute,
                self.state.context
            )
            
            # Update task stats
            self.state.task_stats = self.task_queue.get_stats()
            
            # Process result in chunks if needed
            if isinstance(result, (dict, list, str, bytes)):
                result = await self.memory_manager.process_in_chunks(
                    result,
                    lambda x: x  # Identity function for chunking only
                )
            
            return result
            
        except Exception as e:
            self.state.error = str(e)
            raise

    async def execute_workflow(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the entire workflow with memory management and async processing."""
        try:
            # Reset workflow state
            self.state = WorkflowState(
                current_node="",
                context=initial_data,
                completed_nodes=[]
            )
            
            # Process initial data in chunks if needed
            if isinstance(initial_data, (dict, list, str, bytes)):
                self.state.context = await self.memory_manager.process_in_chunks(
                    initial_data,
                    lambda x: x  # Identity function for chunking only
                )
            
            # Execute nodes in sequence with memory management
            for node_id in self.nodes:
                self.state.current_node = node_id
                
                try:
                    # Execute node with memory management and task queuing
                    result = await self.execute_node(node_id)
                    
                    # Update context with result
                    self.state.context[node_id] = result
                    self.state.completed_nodes.append(node_id)
                    
                    # Allow other tasks to run between nodes
                    await asyncio.sleep(0)
                    
                except Exception as e:
                    self.state.error = str(e)
                    raise
            
            # Clean up resources
            await self.memory_manager.cleanup()
            await self.task_queue.cleanup()
            
            return self.state.context
            
        except Exception as e:
            self.state.error = str(e)
            raise

    def get_workflow_state(self) -> WorkflowState:
        """Get current workflow state including memory and task stats."""
        return self.state

    async def cleanup(self):
        """Clean up resources."""
        await self.memory_manager.cleanup()
        await self.task_queue.cleanup()

def create_workflow() -> WorkflowImplementation:
    """Create a new workflow instance."""
    return WorkflowImplementation() 