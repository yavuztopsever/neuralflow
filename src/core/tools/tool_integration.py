from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from langchain.tools import Tool
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import VectorStore
from langchain.embeddings import OpenAIEmbeddings

from .memory_manager import MemoryManager
from .vector_search import VectorSearch
from .processing.task_processor import TaskProcessor
from .search.graph_search import GraphSearch

@dataclass
class ToolConfig:
    max_memory_items: int = 1000
    vector_dimension: int = 768
    search_k: int = 5
    graph_search_type: str = "hybrid"

class ToolIntegration:
    def __init__(self, llm, config: Optional[ToolConfig] = None):
        self.config = config or ToolConfig()
        self.llm = llm
        
        # Initialize vector store and embeddings
        self.vector_store = VectorStore()
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            max_items=self.config.max_memory_items,
            vector_store=self.vector_store
        )
        
        # Initialize vector search
        self.vector_search = VectorSearch(
            dimension=self.config.vector_dimension
        )
        
        # Initialize graph search
        self.graph_search = GraphSearch(
            vector_store=self.vector_store
        )
        
        # Initialize task processor
        self.task_processor = TaskProcessor(
            llm=self.llm,
            tools=self._get_tools(),
            memory=self.memory_manager.conversation_memory
        )

    def _get_tools(self) -> List[Tool]:
        """Get all available tools."""
        return [
            Tool(
                name="memory_search",
                func=self.memory_manager.get_from_memory,
                description="Search through memory using semantic similarity"
            ),
            Tool(
                name="vector_search",
                func=self.vector_search.search,
                description="Search through vector embeddings"
            ),
            Tool(
                name="graph_search",
                func=self.graph_search.search,
                description="Search through knowledge graph"
            ),
            Tool(
                name="add_to_memory",
                func=self.memory_manager.add_to_memory,
                description="Add information to memory"
            ),
            Tool(
                name="clear_memory",
                func=self.memory_manager.clear_memory,
                description="Clear all memory"
            ),
            Tool(
                name="get_conversation_history",
                func=self.memory_manager.get_conversation_history,
                description="Get the conversation history"
            ),
            Tool(
                name="add_graph_node",
                func=self.graph_search.add_node,
                description="Add a node to the knowledge graph"
            ),
            Tool(
                name="add_graph_edge",
                func=self.graph_search.add_edge,
                description="Add an edge to the knowledge graph"
            ),
            Tool(
                name="find_graph_path",
                func=self.graph_search.find_path,
                description="Find the shortest path between two nodes in the graph"
            )
        ]

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task using all available tools."""
        try:
            # Process the task using the task processor
            result = await self.task_processor.process_task(task)
            
            if result.success:
                # Update memory with the result
                self.memory_manager.add_to_memory(
                    content=str(result.output),
                    memory_type="short_term",
                    metadata={
                        "task_id": task.get("id"),
                        "task_type": task.get("type"),
                        "timestamp": time.time()
                    }
                )
                
                # Update graph if relevant
                if task.get("type") == "knowledge_update":
                    self.graph_search.add_node(
                        node_id=f"task_{task.get('id')}",
                        content=str(result.output),
                        metadata={
                            "task_id": task.get("id"),
                            "task_type": task.get("type"),
                            "timestamp": time.time()
                        }
                    )
            
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "metadata": result.metadata
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e)
            }

    async def process_batch(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple tasks in parallel."""
        return await self.task_processor.process_batch(tasks)

    def get_tool_usage_stats(self) -> Dict[str, int]:
        """Get statistics about tool usage."""
        return self.task_processor.get_tool_usage_stats()

    def reset_tool_stats(self):
        """Reset tool usage statistics."""
        self.task_processor.reset_tool_stats()

    def save_state(self, filepath: str):
        """Save the current state of all tools."""
        state = {
            "memory": self.memory_manager.get_conversation_history(),
            "graph": {
                "nodes": list(self.graph_search.node_cache.values()),
                "edges": list(self.graph_search.edge_cache.values())
            },
            "vector_store": self.vector_store
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f)

    def load_state(self, filepath: str):
        """Load the state of all tools."""
        with open(filepath, 'r') as f:
            state = json.load(f)
            
            # Restore memory
            self.memory_manager.clear_memory()
            for msg in state["memory"]:
                self.memory_manager.conversation_memory.add_message(
                    BaseMessage(
                        type=msg["type"],
                        content=msg["content"]
                    )
                )
            
            # Restore graph
            self.graph_search.clear()
            for node in state["graph"]["nodes"]:
                self.graph_search.add_node(
                    node_id=node["id"],
                    content=node["content"],
                    metadata=node["metadata"]
                )
            for edge in state["graph"]["edges"]:
                self.graph_search.add_edge(
                    source=edge["source"],
                    target=edge["target"],
                    weight=edge["weight"],
                    metadata=edge["metadata"]
                )
            
            # Restore vector store
            self.vector_store = state["vector_store"] 