#!/bin/bash
# Simple wrapper script to run the LangGraph application with Gradio UI

# Default values
PORT=7860
SHARE=false
MINIMAL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --share)
            SHARE=true
            shift
            ;;
        --minimal)
            MINIMAL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="python main.py --port $PORT"
if [ "$SHARE" = true ]; then
    CMD="$CMD --share"
fi
if [ "$MINIMAL" = true ]; then
    CMD="$CMD --minimal"
fi

# Run the application
echo "Starting LangGraph application..."
$CMD