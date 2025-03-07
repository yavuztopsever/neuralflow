# NeuralFlow Workflow Architecture

This document outlines the core workflow architecture of NeuralFlow. It serves as a technical reference for developers working with the codebase.

## Core Workflow Components

The NeuralFlow application implements a dynamic workflow with the following key components:

### 1. Workflow Manager

The `WorkflowManager` class is the central component that orchestrates the entire workflow process. It manages:
- Workflow configuration
- State management
- LangChain integration
- Tool management
- Memory management

```python
class WorkflowManager:
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
```

### 2. Workflow Configuration

The workflow configuration is managed through the `WorkflowConfig` class:

```python
@dataclass
class WorkflowConfig:
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
```

### 3. Workflow State

The workflow state is managed using a Pydantic model that tracks:
- User queries
- Retrieved context
- Execution results
- Memory
- Thinking process
- Conversation state

```python
class WorkflowState(BaseModel):
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
```

### 4. Workflow Graph

The workflow is implemented as a directed graph with the following nodes:

1. **User Input Node**
   - Processes user queries
   - Initializes workflow state
   - Adds input to conversation memory

2. **Context Retrieval Node**
   - Uses LangChain tools to retrieve relevant context
   - Processes and structures context
   - Updates state with retrieved information

3. **Task Execution Node**
   - Uses LangChain agent to execute tasks
   - Handles function calling
   - Processes task results

4. **Response Generation Node**
   - Generates final responses
   - Integrates context and task results
   - Updates conversation memory

```python
def _create_workflow_graph(self) -> StateGraph:
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
```

### 5. LangChain Integration

The workflow integrates with LangChain for:
- LLM interactions
- Tool management
- Memory management
- Agent execution

```python
def _create_workflow_chain(self):
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
```

## Execution Flow

The workflow execution follows these steps:

1. **Initialization**
   - Create WorkflowManager with configuration
   - Initialize LangChain components
   - Set up workflow graph

2. **User Input Processing**
   - Receive user query
   - Initialize state
   - Update conversation memory

3. **Context Retrieval**
   - Search for relevant context
   - Process and structure context
   - Update state with context

4. **Task Execution**
   - Execute tasks using LangChain agent
   - Process task results
   - Update state with results

5. **Response Generation**
   - Generate final response
   - Update conversation memory
   - Return response to user

## State Management

The workflow maintains state throughout execution:
- Conversation history
- Retrieved context
- Task results
- Error states
- Processing flags

## Error Handling

The workflow includes comprehensive error handling:
- Node-level error catching
- State validation
- Graceful degradation
- Error reporting

## Memory Management

Memory is managed through:
- Conversation buffer memory
- Context storage
- State persistence
- Checkpoint management

## Configuration

The workflow is configurable through:
- WorkflowConfig for general settings
- LangChainConfig for LLM settings
- Tool configuration
- Memory settings

## Future Improvements

Planned enhancements include:
- Enhanced error recovery
- Advanced state management
- Improved context retrieval
- Extended tool support
- Better monitoring and logging