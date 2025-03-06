# LangGraph UI Guide

This guide explains how to access and use the LangGraph user interface on various systems.

## Core Workflow

The LangGraph application follows an advanced workflow:

1. **User Input → Context Manager**: Your query is processed and enriched with various context sources
2. **Context Manager**: Gathers conversation history, relevant documents, and knowledge
3. **Task Execution**: For complex queries, performs additional research when needed
4. **Response Generation**: Creates personalized responses based on all available context
5. **Memory Manager**: Saves interaction history for future reference

See `WORKFLOW.md` for detailed architecture information.

## Accessing the UI

There are multiple ways to access the user interface, depending on your system and network setup:

### Option 1: Public Share URL (Recommended)

The simplest way to access the UI, especially on macOS with firewall restrictions, is using a Gradio public share URL:

```bash
python direct_gradio.py
```

This will generate a temporary public URL like `https://xxx-xxx-xxx.gradio.live` that you can open in any web browser. This URL remains active for 72 hours.

### Option 2: Local Access

To access the UI on your local machine:

```bash
./run.sh
```

This starts the server at `http://localhost:7860` by default.

### Option 3: Network Access

To access from other devices on your network:

```bash
./run.sh --host=0.0.0.0
```

Then visit `http://<your-computer-ip>:7860` from other devices.

### Option 4: Minimal UI for Performance

For systems with limited resources:

```bash
./run.sh --minimal
```

This runs a simplified UI with reduced memory usage.

## Troubleshooting UI Access

If you're having trouble accessing the UI:

1. **Firewall Issues**: 
   - macOS firewall may block incoming connections
   - Use `--share` flag to create a public URL
   - Or allow incoming connections to Python in Security settings

2. **Port Already in Use**:
   - Run with a different port: `./run.sh --port=8080`
   - Alternatively, `kill_port.sh 7860` to free the default port

3. **Memory Issues**:
   - Use `--minimal` flag for reduced memory usage
   - Check the system monitor for memory usage

## Using the Chat Interface

1. **Chat History**: 
   - The main panel shows your conversation history
   - Previous messages are retained during your session

2. **Input Message**:
   - Type your message in the input box at the bottom
   - Press Enter or click "Send" to submit

3. **LangGraph Workflow Status**:
   - The footer shows the current LangGraph workflow steps
   - Watch as your message progresses through:
     - Context Retrieval
     - Task Execution
     - Response Generation

4. **Clear Conversation**:
   - Use the "Clear" button to reset the conversation

## Advanced Options

- **Share URL**: Create a public URL with `./run.sh --share`
- **Custom Port**: Change the port with `./run.sh --port=8080`
- **No Memory Optimizations**: Disable memory limits with `./run.sh --no-memory-opt`

## Technical Information

- The UI is built with Gradio, a Python library for creating web interfaces
- It communicates with the LangGraph workflow in the backend
- Memory usage is monitored and optimized for resource-constrained environments
- The interface supports both local and shared access methods