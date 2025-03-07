#!/bin/bash

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Create necessary directories
mkdir -p src/storage/{data,models,vector_store,cache}
mkdir -p src/logs
mkdir -p src/config

# Copy example environment file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
fi

echo "Development environment setup complete!"
echo "Please activate the virtual environment with: source .venv/bin/activate" 