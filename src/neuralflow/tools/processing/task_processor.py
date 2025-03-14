import logging
import importlib
import inspect
import subprocess
import re
import resource
import time
import psutil
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Callable, List, Union
from config.config import Config
from tools.web_search import WebSearch
from tools.document_handler import DocumentHandler
from tools.memory_manager import MemoryManager
from models.model_manager import ModelManager
from tools.response_generation import ResponseGenerator
from tools.context_handler import ContextHandler
from dataclasses import dataclass
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

@dataclass
class TaskResult:
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TaskHandler:
    """Handles task execution, distributed processing, and tool management."""

    def __init__(self, config=None, logger=None, psutil_module=None, web_search=None, model_manager=None):
        """Initializes the TaskHandler with configuration and dependencies."""
        self.config = config or Config
        self.logger = logger or logging
        self.psutil = psutil_module or psutil
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.distributed_mode = False
        self.workers = []
        self.num_workers = 1
        self.web_search = web_search or WebSearch()
        self.model_manager = model_manager
        self.task_queue = asyncio.Queue()
        
        # Initialize available tools
        self.available_tools = {
            "system": "tools.system_info",
            "time": "tools.time_info",
            "search": "tools.web_search",
            "document": "tools.document_handler",
            "memory": "tools.memory_manager",
            "graph": "tools.graph_search",
            "rag": "tools.rag_manager",
            "response": "tools.response_generation",
            "context": "tools.context_handler"
        }
        self.function_map = self._load_functions()
        self._initialize_components()

    def _initialize_components(self):
        """Initializes all necessary components."""
        try:
            # Initialize distributed backend if specified and available
            if (hasattr(self.config, 'DISTRIBUTED_BACKEND') and 
                self.config.DISTRIBUTED_BACKEND == "ray"):
                try:
                    import ray
                    ray.init(
                        address=self.config.DISTRIBUTED_ADDRESS, 
                        ignore_reinit_error=True,
                        _redis_password=getattr(self.config, 'REDIS_PASSWORD', None),
                        _node_ip_address=getattr(self.config, 'NODE_IP_ADDRESS', None),
                        include_dashboard=False,
                        logging_level=logging.ERROR,
                        _timeout_ms=3000
                    )
                    self.distributed_mode = True
                    self.logger.info("Ray initialized successfully")
                except (ImportError, ConnectionError, TimeoutError) as e:
                    self.logger.warning(f"Ray initialization failed: {e}. Using local execution.")
                    self.distributed_mode = False
                    
            # Initialize memory manager
            try:
                self.memory_manager = MemoryManager(self.config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize memory manager: {e}")
                self.memory_manager = None

            # Initialize response generator
            try:
                self.response_generator = ResponseGenerator(
                    model_manager=self.model_manager,
                    config=self.config
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize response generator: {e}")
                self.response_generator = None

            # Initialize context handler
            try:
                self.context_handler = ContextHandler(
                    memory_manager=self.memory_manager,
                    model_manager=self.model_manager,
                    config=self.config
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize context handler: {e}")
                self.context_handler = None
                
        except Exception as e:
            self.logger.warning(f"Error initializing components: {str(e)}")
            self.distributed_mode = False
            self.memory_manager = None
            self.response_generator = None
            self.context_handler = None

    def _load_functions(self) -> Dict[str, Callable]:
        """Loads functions from available tools and maps them to their respective keywords."""
        function_map = {}
        for key, module_name in self.available_tools.items():
            try:
                module = importlib.import_module(module_name)
                for name, func in inspect.getmembers(module, inspect.isfunction):
                    if key in name.lower():
                        function_map[key] = func
            except ImportError as e:
                self.logger.warning(f"Module {module_name} could not be imported: {e}")
        return function_map

    def is_task_call(self, user_query: str) -> Optional[str]:
        """Determines if a task execution is required based on keywords in the user query."""
        keywords = ["system info", "time", "search", "execute code", "help", "document", "memory", "graph", "rag"]
        for keyword in keywords:
            if keyword in user_query.lower():
                return keyword.split()[0]
        return None

    def execute_task(self, func: Callable, *args, **kwargs) -> Any:
        """Executes a task either locally or distributed based on configuration."""
        if self.distributed_mode:
            try:
                import ray
                remote_func = ray.remote(func)
                future = remote_func.remote(*args, **kwargs)
                return ray.get(future, timeout=5.0)
            except Exception as e:
                self.logger.warning(f"Distributed execution failed: {e}. Falling back to local execution.")
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def execute_code(self, code: str, language: str = "python", mode: str = "safe") -> str:
        """Executes code in the specified language and returns the output."""
        try:
            self.logger.info(f"Code execution request: {code} (language: {language}, mode: {mode})")

            if not self._validate_code(code, language, mode):
                self.logger.warning("Code validation failed!")
                return "Code validation failed!"

            self._set_resource_limits(mode)
            start_time = time.time()
            process = self._start_process(code, language)
            
            if not process:
                return "Unsupported language!"

            self._monitor_resources(process)
            output, error = process.communicate()
            end_time = time.time()

            self.logger.info(f"Code execution time: {end_time - start_time} seconds")
            return error.decode() if error else output.decode()
            
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            return f"An unexpected error occurred: {e}"

    def _validate_code(self, code: str, language: str, mode: str) -> bool:
        """Validates the code to prevent malicious commands."""
        patterns = self.config.CODE_VALIDATION_PATTERNS.get(language, [])
        for pattern in patterns:
            if re.search(pattern, code):
                return False
        return True

    def _set_resource_limits(self, mode: str):
        """Sets resource limits based on the execution mode."""
        limits = self.config.RESOURCE_LIMITS.get(mode, {"cpu": 5, "memory": 50})
        resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu"], limits["cpu"]))
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * limits["memory"], 1024 * 1024 * limits["memory"]))

    def _start_process(self, code: str, language: str) -> Optional[subprocess.Popen]:
        """Starts the process for code execution based on the language."""
        commands = {
            "python": ["python", "-c", code],
            "javascript": ["node", "-e", code],
            "bash": ["bash", "-c", code],
            "r": ["Rscript", "-e", code],
            "go": ["go", "run", "-"]
        }
        command = commands.get(language)
        if command:
            if language == "go":
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                process.stdin.write(code.encode())
                process.stdin.close()
            else:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return process
        return None

    def _monitor_resources(self, process: subprocess.Popen):
        """Monitors CPU and memory usage during code execution."""
        while process.poll() is None:
            cpu_percent = self.psutil.cpu_percent()
            memory_percent = self.psutil.virtual_memory().percent
            self.logger.info(f"CPU usage: {cpu_percent}%, Memory usage: {memory_percent}%")
            if cpu_percent > 80 or memory_percent > 80:
                self.logger.warning("High resource usage detected!")
            time.sleep(1)

    def handle_task(self, user_query: str) -> Any:
        """Handles task execution based on user query."""
        task_category = self.is_task_call(user_query)
        
        if "help" in user_query.lower():
            return self.help_function()
        
        if task_category and task_category in self.function_map:
            try:
                func = self.function_map[task_category]
                result = self.execute_task(func)
                return result
            except Exception as e:
                self.logger.error(f"Error executing task '{task_category}': {e}")
                return f"Error: An error occurred while executing the task '{task_category}'. Details: {str(e)}"
        
        if "time" in user_query.lower():
            return self.get_time()
            
        return "No relevant task found for execution."

    def help_function(self) -> str:
        """Returns a list of available tasks and their descriptions."""
        help_text = "Available tasks:\n"
        for key in self.available_tools.keys():
            help_text += f"- {key}: Execute {key}-related operations\n"
        help_text += "- help: Show this help message\n"
        return help_text

    def get_time(self) -> str:
        """Returns the current time and date."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Current time: {current_time}"

    def shutdown(self):
        """Shutdown the executor and cleanup resources."""
        self.executor.shutdown(wait=True)
        if self.distributed_mode:
            try:
                import ray
                ray.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down Ray: {e}") 

    async def calculate_task_priority(self, query: str, context: Dict[str, Any]) -> int:
        """Calculate the priority of a task based on query and context."""
        try:
            # Check for urgency indicators
            urgency_indicators = ["urgent", "asap", "immediately", "now"]
            if any(indicator in query.lower() for indicator in urgency_indicators):
                return 0  # Highest priority for urgent requests
            
            # Check context relevance
            if context and context.get('documents'):
                return 1
            
            # Default priorities by task type
            base_priorities = {
                "function_call": 2,
                "document_retrieval": 3,
                "memory_access": 4
            }
            
            # Get base priority with fallback to lowest priority
            priority = base_priorities.get(context.get("task_type", ""), 10)
            
            # Adjust based on query length (shorter queries might be more urgent)
            if len(query.split()) < 5:
                priority -= 1
                
            # Adjust based on context if available
            if context and isinstance(context, dict):
                # If this query is related to previous interactions, might be more important
                if context.get('chat_logs') and len(context.get('chat_logs', [])) > 0:
                    priority -= 1
            
            return max(0, priority)  # Ensure priority doesn't go below 0
        except Exception as e:
            self.logger.warning(f"Error calculating task priority: {e}")
            return 5  # Default to medium priority on error

    async def determine_task_type(self, query: str, context: Dict[str, Any]) -> str:
        """Determine the type of task to execute."""
        try:
            if self.model_manager:
                # Format prompt for task determination
                prompt = f"""Based on the following query and context, determine the appropriate task type:
                Query: {query}
                Context: {context}
                
                Choose one of: web_search, function_call, llm_task
                """
                
                # Get task type from model
                task_type = await self.model_manager.generate_response(prompt)
                return task_type.strip().lower()
            else:
                # Fallback to keyword-based determination
                if any(keyword in query.lower() for keyword in ["search", "find", "look up"]):
                    return "web_search"
                elif any(keyword in query.lower() for keyword in ["execute", "run", "call"]):
                    return "function_call"
                else:
                    return "llm_task"
        except Exception as e:
            self.logger.error(f"Error determining task type: {str(e)}")
            return "llm_task"  # Default to LLM task on error

    async def execute_task(self, query: str, context: Dict[str, Any], search_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a task based on the query and context."""
        try:
            # For general informational queries, provide a default response
            if any(phrase in query.lower() for phrase in ["what can you do", "what can you help", "capabilities", "how can you help", "what are you capable"]):
                return {
                    "status": "success",
                    "data": {
                        "general_info": self._get_capabilities_info()
                    }
                }
            
            # Determine task type
            task_type = await self.determine_task_type(query, context)
            
            # Execute appropriate task
            if task_type == "web_search":
                return await self._execute_web_search(query, context)
            elif task_type == "function_call":
                return await self._execute_function_call(query, context)
            else:
                return await self._execute_llm_task(query, context)
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def _execute_function_call(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function call task."""
        try:
            if self.model_manager:
                # Determine function and arguments using model
                prompt = f"""Based on the query and context, determine the function to call and its arguments:
                Query: {query}
                Context: {context}
                """
                
                function_info = await self.model_manager.generate_response(prompt)
            else:
                # Fallback to keyword-based function selection
                function_info = self._determine_function_from_keywords(query)
            
            # Execute function
            result = self.execute_task(function_info["function_name"], function_info.get("arguments", {}))
            
            return {
                "status": "success",
                "data": {
                    "task_type": "function_call",
                    "function_result": result
                }
            }
        except Exception as e:
            self.logger.error(f"Error executing function call: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def _execute_web_search(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a web search task."""
        try:
            results = await self.web_search.search(query)
            return {
                "status": "success",
                "data": {
                    "task_type": "web_search",
                    "web_results": results
                }
            }
        except Exception as e:
            self.logger.error(f"Error executing web search: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def _execute_llm_task(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a general LLM task."""
        try:
            if self.model_manager:
                response = await self.model_manager.generate_response(query)
            else:
                response = "LLM functionality is not available."
                
            return {
                "status": "success",
                "data": {
                    "task_type": "llm_task",
                    "response": response
                }
            }
        except Exception as e:
            self.logger.error(f"Error executing LLM task: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def execute(self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None, search_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the task with the given state and context."""
        try:
            # Extract query from state
            query = state.get("user_query", "")
            if not query:
                raise ValueError("No query provided in state")
            
            # Process context if context handler is available
            if self.context_handler:
                context = await self.context_handler.process_context(query, context or {})
            
            # Add task to queue
            await self.task_queue.put((query, context))
            
            # Process task
            result = await self.execute_task(query, context or {}, search_params)
            
            # Generate response if response generator is available
            if self.response_generator and result.get("status") == "success":
                response = await self.response_generator.generate(
                    execution_result=result,
                    retrieved_context=context
                )
                result["response"] = response
            
            # Save to memory if successful
            if result.get("status") == "success" and self.memory_manager:
                await self.memory_manager.add_to_memory({
                    'query': query,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
            
            return result
        except Exception as e:
            self.logger.error(f"Error in task execution: {e}")
            return {
                "status": "error",
                "error": "I apologize, but I encountered an error while processing your request."
            }

    def _get_capabilities_info(self) -> str:
        """Get information about available capabilities."""
        capabilities = [
            "Execute code in multiple languages",
            "Perform web searches",
            "Handle document operations",
            "Manage memory and context",
            "Execute distributed tasks",
            "Process natural language queries"
        ]
        return "\n".join(f"- {cap}" for cap in capabilities)

    def _determine_function_from_keywords(self, query: str) -> Dict[str, Any]:
        """Determine function and arguments from keywords when model is not available."""
        # Map keywords to functions
        keyword_to_function = {
            "system": "get_system_info",
            "time": "get_time",
            "search": "web_search",
            "document": "handle_document",
            "memory": "access_memory",
            "graph": "graph_search"
        }
        
        # Find matching function
        for keyword, function_name in keyword_to_function.items():
            if keyword in query.lower():
                return {
                    "function_name": function_name,
                    "arguments": {"query": query}
                }
        
        # Default to general query handling
        return {
            "function_name": "handle_general_query",
            "arguments": {"query": query}
        }

class TaskProcessor:
    def __init__(self, llm, tools: List[Tool], memory: Optional[ConversationBufferMemory] = None):
        self.llm = llm
        self.tools = tools
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize agent
        self.agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            verbose=True
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

    async def process_task(self, task: Dict[str, Any]) -> TaskResult:
        """Process a task using the agent executor."""
        try:
            with get_openai_callback() as cb:
                result = await self.agent_executor.arun(
                    input=task["input"],
                    context=task.get("context", {})
                )
                
                return TaskResult(
                    success=True,
                    output=result,
                    metadata={
                        "tokens_used": cb.total_tokens,
                        "cost": cb.total_cost
                    }
                )
                
        except Exception as e:
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )

    async def process_batch(self, tasks: List[Dict[str, Any]]) -> List[TaskResult]:
        """Process multiple tasks in parallel."""
        return await asyncio.gather(
            *[self.process_task(task) for task in tasks]
        )

    def add_tool(self, tool: Tool):
        """Add a new tool to the agent."""
        self.tools.append(tool)
        self.agent_executor.tools = self.tools

    def remove_tool(self, tool_name: str):
        """Remove a tool from the agent."""
        self.tools = [tool for tool in self.tools if tool.name != tool_name]
        self.agent_executor.tools = self.tools

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()

    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the conversation history."""
        return self.memory.chat_memory.messages

    def save_conversation(self, filepath: str):
        """Save the conversation history to a file."""
        import json
        history = [
            {
                "type": msg.type,
                "content": msg.content
            }
            for msg in self.get_conversation_history()
        ]
        with open(filepath, 'w') as f:
            json.dump(history, f)

    def load_conversation(self, filepath: str):
        """Load the conversation history from a file."""
        import json
        with open(filepath, 'r') as f:
            history = json.load(f)
            self.clear_memory()
            for msg in history:
                self.memory.chat_memory.add_message(
                    BaseMessage(
                        type=msg["type"],
                        content=msg["content"]
                    )
                )

    def get_tool_usage_stats(self) -> Dict[str, int]:
        """Get statistics about tool usage."""
        stats = {}
        for tool in self.tools:
            stats[tool.name] = getattr(tool, "call_count", 0)
        return stats

    def reset_tool_stats(self):
        """Reset tool usage statistics."""
        for tool in self.tools:
            if hasattr(tool, "call_count"):
                tool.call_count = 0 