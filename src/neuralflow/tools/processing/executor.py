"""
Unified executor tool for NeuralFlow.
Provides task execution and reasoning capabilities.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from ..base import BaseTool, ToolConfig, ToolType, ToolResult
from ...storage.base import StorageConfig
from ...utils.logging import get_logger

logger = get_logger(__name__)

class TaskType(Enum):
    """Types of tasks that can be executed."""
    FUNCTION_CALL = "function_call"
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    API_REQUEST = "api_request"
    REASONING = "reasoning"

class ReasoningStage(Enum):
    """Stages in the reasoning process."""
    OUTLINE = "outline"
    DRAFT = "draft"
    SYNTHESIS = "synthesis"

@dataclass
class ExecutorConfig:
    """Configuration for the executor tool."""
    max_workers: int = 4
    max_iterations: int = 3
    temperature: float = 0.7
    model_name: str = "gpt-4"
    draft_refinement_threshold: float = 0.8

class ExecutorTool(BaseTool):
    """Unified tool for task execution and reasoning."""
    
    def __init__(
        self,
        config: ToolConfig,
        executor_config: Optional[ExecutorConfig] = None,
        storage_config: Optional[StorageConfig] = None
    ):
        """Initialize the executor tool.
        
        Args:
            config: Tool configuration
            executor_config: Optional executor-specific configuration
            storage_config: Optional storage configuration
        """
        super().__init__(config, storage_config)
        
        # Initialize executor config
        self.executor_config = executor_config or ExecutorConfig()
        
        # Initialize components
        self.thread_pool = ThreadPoolExecutor(max_workers=self.executor_config.max_workers)
        self.registered_functions: Dict[str, Callable] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model_name=self.executor_config.model_name,
            temperature=self.executor_config.temperature
        )
        self.memory = ConversationBufferMemory()
        
        # Initialize chains
        self._initialize_chains()
    
    async def __aenter__(self):
        """Initialize async resources."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async resources."""
        if self.session:
            await self.session.close()
    
    def register_function(self, name: str, func: Callable) -> None:
        """Register a function for function call tasks.
        
        Args:
            name: Function name
            func: Function to register
        """
        self.registered_functions[name] = func
    
    async def _execute_impl(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the tool implementation.
        
        Args:
            input_data: Input data containing task type and parameters
            context: Optional execution context
            
        Returns:
            Dict[str, Any]: Execution results
        """
        task_type = TaskType(input_data.get("task_type", "function_call"))
        task_data = input_data.get("task_data", {})
        
        try:
            if task_type == TaskType.FUNCTION_CALL:
                result = await self._execute_function_call(task_data)
            elif task_type == TaskType.WEB_SEARCH:
                result = await self._execute_web_search(task_data)
            elif task_type == TaskType.CODE_EXECUTION:
                result = await self._execute_code(task_data)
            elif task_type == TaskType.API_REQUEST:
                result = await self._execute_api_request(task_data)
            elif task_type == TaskType.REASONING:
                result = await self._execute_reasoning(task_data, context)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            return {
                "task_type": task_type.value,
                "output": result,
                "metadata": self._generate_metadata(task_type)
            }
            
        except Exception as e:
            self.error_handler.handle_error(
                "EXECUTION_ERROR",
                f"Task execution failed: {str(e)}",
                details={
                    "task_type": task_type.value,
                    "task_data": task_data
                }
            )
            raise
    
    async def _execute_function_call(self, task_data: Dict[str, Any]) -> Any:
        """Execute a registered function with given parameters."""
        func_name = task_data.get("function_name")
        if func_name not in self.registered_functions:
            raise ValueError(f"Function {func_name} not registered")
        
        func = self.registered_functions[func_name]
        args = task_data.get("args", [])
        kwargs = task_data.get("kwargs", {})
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: func(*args, **kwargs)
        )
    
    async def _execute_web_search(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a web search task."""
        query = task_data.get("query")
        if not query:
            raise ValueError("Search query is required")
        
        # TODO: Implement web search logic
        return {
            "query": query,
            "results": []
        }
    
    async def _execute_code(self, task_data: Dict[str, Any]) -> Any:
        """Execute code in a safe environment."""
        code = task_data.get("code")
        if not code:
            raise ValueError("Code is required")
        
        # TODO: Implement safe code execution
        return {"output": "Code execution not implemented"}
    
    async def _execute_api_request(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an API request."""
        if not self.session:
            raise RuntimeError("Client session not initialized")
        
        url = task_data.get("url")
        method = task_data.get("method", "GET")
        headers = task_data.get("headers", {})
        data = task_data.get("data")
        
        if not url:
            raise ValueError("URL is required")
        
        async with self.session.request(
            method=method,
            url=url,
            headers=headers,
            json=data
        ) as response:
            return {
                "status": response.status,
                "headers": dict(response.headers),
                "data": await response.json()
            }
    
    async def _execute_reasoning(
        self,
        task_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a reasoning task using Chain-of-Draft approach."""
        query = task_data.get("query")
        if not query:
            raise ValueError("Query is required for reasoning")
        
        context_str = str(context) if context else ""
        current_iteration = 0
        
        # Generate initial outline
        outline = await self.outline_chain.arun(
            query=query,
            context=context_str
        )
        
        # Generate and refine drafts
        drafts = []
        while current_iteration < self.executor_config.max_iterations:
            draft = await self.draft_chain.arun(
                outline=outline,
                query=query,
                context=context_str
            )
            drafts.append(draft)
            
            # Check if refinement needed
            if self._needs_refinement(draft):
                outline = await self.refinement_chain.arun(
                    current_outline=outline,
                    draft=draft,
                    query=query
                )
            else:
                break
            
            current_iteration += 1
        
        # Generate final synthesis
        final_answer = await self.synthesis_chain.arun(
            outline=outline,
            drafts="\n\n".join(drafts),
            query=query,
            context=context_str
        )
        
        return {
            "outline": outline,
            "drafts": drafts,
            "final_answer": final_answer,
            "iterations": current_iteration + 1
        }
    
    def _initialize_chains(self):
        """Initialize LangChain components."""
        # Outline chain
        outline_template = """Given the following query and context, create a detailed outline for the response.
        Query: {query}
        Context: {context}
        
        Create a structured outline that covers all aspects of the query.
        """
        self.outline_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["query", "context"],
                template=outline_template
            ),
            memory=self.memory
        )
        
        # Draft chain
        draft_template = """Using the following outline and query, generate a detailed draft response.
        Outline: {outline}
        Query: {query}
        Context: {context}
        
        Generate a comprehensive draft that follows the outline structure.
        """
        self.draft_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["outline", "query", "context"],
                template=draft_template
            ),
            memory=self.memory
        )
        
        # Synthesis chain
        synthesis_template = """Based on the following outline, drafts, and query, create a final synthesized response.
        Outline: {outline}
        Drafts: {drafts}
        Query: {query}
        Context: {context}
        
        Create a polished final response that incorporates the best elements from all drafts.
        """
        self.synthesis_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["outline", "drafts", "query", "context"],
                template=synthesis_template
            ),
            memory=self.memory
        )
        
        # Refinement chain
        refinement_template = """Review the current outline and draft, and refine the outline if needed.
        Current Outline: {current_outline}
        Draft: {draft}
        Query: {query}
        
        Provide an improved outline that addresses any gaps or issues identified in the draft.
        """
        self.refinement_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["current_outline", "draft", "query"],
                template=refinement_template
            ),
            memory=self.memory
        )
    
    def _needs_refinement(self, draft: str) -> bool:
        """Determine if a draft needs refinement."""
        # TODO: Implement refinement logic
        return False
    
    def _generate_metadata(self, task_type: TaskType) -> Dict[str, Any]:
        """Generate metadata about the execution."""
        return {
            "task_type": task_type.value,
            "config": {
                "max_workers": self.executor_config.max_workers,
                "max_iterations": self.executor_config.max_iterations,
                "temperature": self.executor_config.temperature,
                "model_name": self.executor_config.model_name,
                "draft_refinement_threshold": self.executor_config.draft_refinement_threshold
            }
        }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            super().cleanup()
            self.thread_pool.shutdown(wait=True)
            if hasattr(self, "llm"):
                del self.llm
            if hasattr(self, "memory"):
                del self.memory
        except Exception as e:
            self.error_handler.handle_error(
                "CLEANUP_ERROR",
                f"Failed to clean up resources: {e}"
            ) 