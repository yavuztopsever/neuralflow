from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import aiohttp
from ..utils.logging import get_logger
from .cod_reasoning import ChainOfDraftReasoning, CoDConfig, CoDResult

logger = get_logger(__name__)

class TaskType(Enum):
    FUNCTION_CALL = "function_call"
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    API_REQUEST = "api_request"
    COD_REASONING = "cod_reasoning"

@dataclass
class TaskResult:
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

class TaskExecutor:
    def __init__(self, max_workers: int = 4):
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.registered_functions: Dict[str, Callable] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.cod_reasoning: Optional[ChainOfDraftReasoning] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    def register_function(self, name: str, func: Callable) -> None:
        """Register a function for function call tasks."""
        self.registered_functions[name] = func
        
    def initialize_cod_reasoning(self, config: Optional[CoDConfig] = None) -> None:
        """Initialize the Chain-of-Draft reasoning module."""
        self.cod_reasoning = ChainOfDraftReasoning(config)
        
    async def execute_task(self, task_type: TaskType, 
                         task_data: Dict[str, Any]) -> TaskResult:
        """Execute a task based on its type."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if task_type == TaskType.FUNCTION_CALL:
                result = await self._execute_function_call(task_data)
            elif task_type == TaskType.WEB_SEARCH:
                result = await self._execute_web_search(task_data)
            elif task_type == TaskType.CODE_EXECUTION:
                result = await self._execute_code(task_data)
            elif task_type == TaskType.API_REQUEST:
                result = await self._execute_api_request(task_data)
            elif task_type == TaskType.COD_REASONING:
                result = await self._execute_cod_reasoning(task_data)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
                
            execution_time = asyncio.get_event_loop().time() - start_time
            return TaskResult(
                success=True,
                output=result,
                execution_time=execution_time,
                metadata={"task_type": task_type.value}
            )
            
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            execution_time = asyncio.get_event_loop().time() - start_time
            return TaskResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=execution_time,
                metadata={"task_type": task_type.value}
            )
    
    async def _execute_function_call(self, task_data: Dict[str, Any]) -> Any:
        """Execute a registered function with given parameters."""
        func_name = task_data.get("function_name")
        if func_name not in self.registered_functions:
            raise ValueError(f"Function {func_name} not registered")
            
        func = self.registered_functions[func_name]
        args = task_data.get("args", [])
        kwargs = task_data.get("kwargs", {})
        
        # Run function in thread pool to avoid blocking
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
            
        # Implement web search logic here
        # This is a placeholder implementation
        return {
            "query": query,
            "results": []
        }
    
    async def _execute_code(self, task_data: Dict[str, Any]) -> Any:
        """Execute code in a safe environment."""
        code = task_data.get("code")
        if not code:
            raise ValueError("Code is required")
            
        # Implement safe code execution logic here
        # This is a placeholder implementation
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
            
        try:
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
        except Exception as e:
            raise RuntimeError(f"API request failed: {str(e)}")
    
    async def _execute_cod_reasoning(self, task_data: Dict[str, Any]) -> CoDResult:
        """Execute Chain-of-Draft reasoning task."""
        if not self.cod_reasoning:
            raise RuntimeError("Chain-of-Draft reasoning not initialized")
            
        query = task_data.get("query")
        context = task_data.get("context")
        
        if not query:
            raise ValueError("Query is required for CoD reasoning")
            
        return await self.cod_reasoning.process_query(query, context)
    
    def get_registered_functions(self) -> List[str]:
        """Get list of registered function names."""
        return list(self.registered_functions.keys()) 