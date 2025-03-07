"""
Core workflow management system for LangGraph.
This module provides a unified interface for managing workflows using LangChain components.
"""

from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple, Union
from dataclasses import dataclass
import asyncio
import logging
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from ...config.langchain_config import LangChainConfig, LangChainManager
from ..tools.langchain_tools import LangChainTools
from ..utils.logging_manager import get_logger
from ..utils.state_manager import StateManager

logger = get_logger(__name__)

@dataclass
class WorkflowConfig:
    """Configuration for the workflow system."""
    max_context_items: int = 5
    max_parallel_tasks: int = 3
    response_format: str = "text"
    include_sources: bool = True
    include_metadata: bool = False
    execution_mode: str = "safe"
    priority: int = 0
    add_thinking: bool = False
    langchain_config: Optional[LangChainConfig] = None
    model_name: str = "gpt-4"

class WorkflowState(BaseModel):
    """Schema for the workflow state using Pydantic for validation."""
    user_query: str = ""
    retrieved_context: Dict[str, Any] = Field(default_factory=dict)
    execution_result: Dict[str, Any] = Field(default_factory=dict)
    final_response: Optional[str] = None
    error: Optional[str] = None
    priority: int = 0
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    thinking: List[str] = Field(default_factory=list)
    conversation_id: Optional[str] = None
    context_processed: bool = False
    needs_more_context: bool = False

class WorkflowManager:
    """Main class for managing workflows in the LangGraph system."""
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.state_manager = StateManager()
        
        # Initialize LangChain components
        if self.config.langchain_config:
            self.langchain_manager = LangChainManager(self.config.langchain_config)
            self.tools = LangChainTools(
                self.langchain_manager.vector_store,
                self.langchain_manager.llm
            )
            
            # Initialize LLM and memory
            self.llm = ChatOpenAI(
                model_name=self.config.model_name,
                temperature=0.7
            )
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
            
            # Create structured chains using LCEL
            self.workflow_chain = self._create_workflow_chain()
            
            # Create agent with function calling
            self.agent = self._create_agent()
        else:
            raise ValueError("LangChain configuration is required")
        
        self.workflow_graph = self._create_workflow_graph()
    
    def _create_workflow_chain(self):
        """Create the main workflow chain using LCEL."""
        # Create context retrieval chain
        context_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant that retrieves relevant context for queries."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{query}")
        ])
        
        context_chain = context_prompt | self.llm | StrOutputParser()
        
        # Create task execution chain
        task_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant that executes tasks based on queries and context."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="Query: {query}\nContext: {context}")
        ])
        
        task_chain = task_prompt | self.llm | StrOutputParser()
        
        # Create response generation chain
        response_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant that generates final responses."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="Query: {query}\nContext: {context}\nTask Result: {task_result}")
        ])
        
        response_chain = response_prompt | self.llm | StrOutputParser()
        
        # Combine chains using LCEL
        return (
            RunnablePassthrough.assign(
                context=context_chain,
                task_result=task_chain
            )
            | response_chain
        )
    
    def _create_agent(self) -> AgentExecutor:
        """Create an agent with function calling capabilities."""
        # Define system message
        system_message = """You are a helpful assistant that can use various tools to help users.
        Use the available tools to accomplish tasks and provide detailed responses."""
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools.get_all_tools(),
            prompt=prompt
        )
        
        # Create agent executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools.get_all_tools(),
            memory=self.memory,
            verbose=True
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the main workflow graph."""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("user_input", self._user_input_node)
        workflow.add_node("context_retrieval", self._context_retrieval_node)
        workflow.add_node("task_execution", self._task_execution_node)
        workflow.add_node("response_generation", self._response_generation_node)
        
        # Add edges
        workflow.add_edge("user_input", "context_retrieval")
        workflow.add_edge("context_retrieval", "task_execution")
        workflow.add_edge("task_execution", "response_generation")
        workflow.add_edge("response_generation", END)
        
        return workflow
    
    async def _user_input_node(self, state: Union[Dict[str, Any], WorkflowState]) -> Dict[str, Any]:
        """Process user input and initialize workflow state."""
        try:
            if hasattr(state, "get") and callable(state.get):
                user_query = state.get("user_query", "")
            else:
                user_query = getattr(state, "user_query", "")
            
            logger.info(f"Processing user input: {user_query[:30]}...")
            
            # Add to memory
            self.memory.chat_memory.add_user_message(user_query)
            
            return {
                "retrieved_context": {},
                "execution_result": {},
                "final_response": None,
                "needs_more_context": False,
                "next": "context_retrieval"
            }
        except Exception as e:
            logger.error(f"Error in user_input_node: {str(e)}", exc_info=True)
            return {
                "error": f"Error processing user input: {str(e)}",
                "next": END
            }
    
    async def _context_retrieval_node(self, state: Union[Dict[str, Any], WorkflowState]) -> Dict[str, Any]:
        """Retrieve and process relevant context using LangChain."""
        try:
            # Use the search tool to retrieve context
            search_tool = self.tools.get_search_tool()
            context = await asyncio.to_thread(
                search_tool.run,
                state.user_query
            )
            
            # Add to memory
            self.memory.chat_memory.add_ai_message(f"Retrieved context: {context}")
            
            return {
                "retrieved_context": {"search_results": context},
                "context_processed": True,
                "next": "task_execution"
            }
        except Exception as e:
            logger.error(f"Error in context_retrieval_node: {str(e)}", exc_info=True)
            return {
                "error": f"Error retrieving context: {str(e)}",
                "next": END
            }
    
    async def _task_execution_node(self, state: Union[Dict[str, Any], WorkflowState]) -> Dict[str, Any]:
        """Execute tasks using LangChain agent."""
        try:
            # Use the agent to execute tasks
            result = await asyncio.to_thread(
                self.agent.run,
                input=state.user_query
            )
            
            # Add to memory
            self.memory.chat_memory.add_ai_message(f"Task execution result: {result}")
            
            return {
                "execution_result": {"agent_result": result},
                "next": "response_generation"
            }
        except Exception as e:
            logger.error(f"Error in task_execution_node: {str(e)}", exc_info=True)
            return {
                "error": f"Error executing tasks: {str(e)}",
                "next": END
            }
    
    async def _response_generation_node(self, state: Union[Dict[str, Any], WorkflowState]) -> Dict[str, Any]:
        """Generate the final response using LangChain chain."""
        try:
            # Use the workflow chain to generate response
            result = await asyncio.to_thread(
                self.workflow_chain.invoke,
                {
                    "query": state.user_query,
                    "context": state.retrieved_context.get("search_results", ""),
                    "task_result": state.execution_result.get("agent_result", ""),
                    "chat_history": self.memory.chat_memory.messages
                }
            )
            
            # Add to memory
            self.memory.chat_memory.add_ai_message(result)
            
            return {
                "final_response": result,
                "next": END
            }
        except Exception as e:
            logger.error(f"Error in response_generation_node: {str(e)}", exc_info=True)
            return {
                "error": f"Error generating response: {str(e)}",
                "next": END
            }
    
    async def run(self, user_query: str, **kwargs) -> str:
        """Run the workflow with the given query."""
        try:
            # Initialize state
            initial_state = WorkflowState(
                user_query=user_query,
                priority=self.config.priority,
                conversation_id=kwargs.get("conversation_id")
            )
            
            # Run the workflow
            final_state = await self.workflow_graph.arun(initial_state)
            
            return final_state.final_response or "No response generated"
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return f"Error executing workflow: {str(e)}"
    
    async def run_with_progress(
        self, 
        user_query: str, 
        **kwargs
    ) -> AsyncGenerator[Tuple[float, Optional[str]], None]:
        """Run the workflow with progress updates."""
        try:
            # Initialize state
            initial_state = WorkflowState(
                user_query=user_query,
                priority=self.config.priority,
                conversation_id=kwargs.get("conversation_id")
            )
            
            # Run the workflow with progress tracking
            async for progress, message in self.workflow_graph.arun_with_progress(initial_state):
                yield progress, message
                
        except Exception as e:
            logger.error(f"Workflow execution with progress failed: {str(e)}")
            yield 0.0, f"Error executing workflow: {str(e)}"
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current workflow state."""
        return self.state_manager.get_state()
    
    def set_state(self, state: Dict[str, Any]):
        """Set the current workflow state."""
        self.state_manager.set_state(state)
    
    def save_checkpoint(self, checkpoint_id: str):
        """Save the current workflow state as a checkpoint."""
        self.state_manager.save_checkpoint(checkpoint_id)
    
    def load_checkpoint(self, checkpoint_id: str):
        """Load a workflow state from a checkpoint."""
        self.state_manager.load_checkpoint(checkpoint_id) 