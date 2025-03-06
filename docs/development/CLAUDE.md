# LangGraph Project Guidelines

## Core Workflow Architecture
- **Main Flow**: user_input → context_manager → task_execution (conditional) → response_manager → memory_manager
- **Context Path**: Memory Manager → Vector DB → Graph DB → Document RAG → Context Evaluation
- **Workflow Logic**: See `WORKFLOW.md` for detailed architecture documentation
- **Intelligent Routing**: Context sufficiency determines execution path
- **Dynamic Reasoning**: Self-querying system for missing information using LangGraph

## LangGraph Features
- **State Management**: Uses LangGraph's native state persistence with checkpointing
- **Memory Integration**: Memory is tracked in LangGraph state using MemoryStore
- **Checkpoint System**: Each conversation gets unique checkpoint IDs
- **Streaming Support**: Using `.astream()` for real-time output progress
- **Pydantic Models**: State schemas use Pydantic for validation

## Launch Commands
- Standard launch: `./run.sh`
- Launch with public URL: `./run.sh --share`
- Launch with memory optimization: `./run.sh --minimal`
- Direct launch: `python direct_gradio.py`

## Build & Test Commands
- Install dependencies: `pip install -r requirements.txt`
- Run all tests: `pytest tests/`
- Run specific test file: `pytest tests/test_graph_workflow.py`
- Run single test: `pytest tests/test_graph_workflow.py::test_start_function`
- Run async tests: `pytest -m asyncio`
- Run e2e tests: `pytest tests/e2e/`

## Code Style Guidelines
- **Imports**: Standard library first, third-party second, project imports last
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPERCASE for constants
- **Type Hints**: Use throughout for function parameters and return values
- **Documentation**: Docstrings for all functions, classes using triple quotes
- **Error Handling**: Try/except blocks with specific error types, descriptive messages
- **Formatting**: 4 spaces for indentation, reasonable line length
- **Testing**: Write unit tests with pytest, use fixtures for test setup
- **Architecture**: Modular design with dependency injection pattern
- **State Management**: Use LangGraph for workflow state management
- **Configuration**: Use Config class for managing environment/settings

## Distributed Support
- **State Management**: Use Redis for distributed state management
- **State Save Interval**: Configure `STATE_SAVE_INTERVAL` in `config/config.py`
- **Memory Stores**: Split between short, mid, and long-term memory

## UI Framework
- **Gradio**: The application uses Gradio for the UI (direct_gradio.py)
- **Public URL Sharing**: Enable with `--share` flag or set `SHARE=true` environment variable
- **Memory Optimization**: Use `--minimal` flag or set `MINIMAL_MODE=true` environment variable
- **Customization**: UI settings can be adjusted in direct_gradio.py