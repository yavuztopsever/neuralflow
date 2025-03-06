# LangGraph Debugging Guide

This document provides troubleshooting steps for common issues you might encounter when running the LangGraph application with Gradio UI.

## Startup Issues

### Application fails to start

If the application fails to start, check the following:

1. **Check logs**:
   ```
   cat logs/app.log
   ```

2. **Check Python environment**:
   - Make sure all dependencies are installed:
   ```
   pip install -r requirements.txt
   ```

3. **Gradio issues**:
   - If Gradio fails to start, try running it directly:
   ```
   python direct_gradio.py
   ```
   
   - Check if Gradio is properly installed:
   ```
   pip install gradio
   python -c "import gradio; print(gradio.__version__)"
   ```

### Redis Connection Issues

The application will use a mock Redis implementation if Redis is unavailable. If you want to use real Redis:

1. **Install Redis**:
   - On macOS: `brew install redis && brew services start redis`
   - On Linux: `sudo apt install redis-server && sudo systemctl start redis`

2. **Check Redis connection**:
   ```
   redis-cli ping
   ```

3. **Update .env file**:
   ```
   USE_REDIS=true
   REDIS_HOST=localhost
   REDIS_PORT=6379
   ```

### LLM Issues

1. **Mock LLM is being used**:
   - If you see a warning about using mock LLM, install llama-cpp-python:
   ```
   pip install llama-cpp-python
   ```

2. **GGUF model not found**:
   - Make sure you have a GGUF model file in `models/gguf_llm/`
   - You can download models from https://huggingface.co/TheBloke

## Import Issues

If you encounter `ModuleNotFoundError` when running Streamlit, it's likely due to Python path issues:

1. **Add parent directory to Python path**:
   ```python
   import sys
   import os
   sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
   ```

2. **Run with the correct Python path**:
   ```bash
   PYTHONPATH=/Volumes/HomeX/yavuztopsever/LangGraph_Temp streamlit run ui/streamlit_chat_interface.py
   ```

3. **Path handling issues**:
   When combining Path objects with strings, convert Path objects to strings first:
   ```python
   # Correct:
   path_str = str(Path("some/path"))
   
   # Incorrect:
   path_str = Path("some/path") + "/extension"  # TypeError
   ```

## Runtime Issues

### Memory Issues

If the application is using too much memory:

1. **Enable low memory mode**:
   ```
   export LOW_MEMORY_MODE=true
   ```

2. **Reduce context window size**:
   - Edit config/config.py:
   ```python
   GGUF_CONTEXT_WINDOW = 2048  # Reduced from default
   ```

### Gradio UI Issues

If the Gradio UI is not loading properly:

1. **Check for port conflicts**:
   ```
   lsof -i :7860
   ```

2. **Try different browser** or use incognito/private browsing

3. **Force restart the Gradio server**:
   ```
   pkill -f gradio
   ./run.sh
   ```
   
4. **Network access issues**:
   - For Mac with firewall, use the `--share` flag to create a public URL:
   ```
   ./run.sh --share
   ```
   
   - Check firewall settings if you can't access the UI from other devices:
   ```
   # On macOS
   sudo defaults read /Library/Preferences/com.apple.alf globalstate
   ```

## Advanced Debugging

### Thread Safety Issues

If you experience random crashes or race conditions:

1. Check thread synchronization in memory_manager.py and graph_workflow.py

2. Enable detailed logging:
   ```
   export LOG_LEVEL=DEBUG
   ```

3. Monitor process activity:
   ```
   ps aux | grep python
   ```

### Component Initialization

If specific components fail to initialize:

1. Initialize and test them individually:
   ```python
   from tools.memory_manager import MemoryManager
   mm = MemoryManager()
   ```

2. Check file permissions for storage directories:
   ```
   ls -la data/ memory/ logs/ models/
   ```

## Command Summary

- **Run the application**: `./run.sh`
- **Run with public URL**: `./run.sh --share`
- **Run in minimal mode**: `./run.sh --minimal`
- **Direct Gradio launch**: `python direct_gradio.py`
- **Custom port**: `PORT=8000 ./run.sh`
- **Test Mode**: `TEST_MODE=true python main.py`
- **Monitor logs**: `tail -f logs/app.log`
- **Kill running instances**: `pkill -f gradio`