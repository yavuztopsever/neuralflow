#!/bin/bash

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Create necessary directories
mkdir -p src/neuralflow/tools/{data,monitoring,state,services,processing,search,memory}/{core,providers,handlers,utils}
mkdir -p src/neuralflow/infrastructure/{config,storage,logging}/{core,providers,handlers}
mkdir -p src/neuralflow/frontend/{api,ui}/{core,components,routes,services}
mkdir -p storage/{data,models,vector_store,cache,state}
mkdir -p logs/{app,access,error}
mkdir -p config/{environments,providers}

# Initialize empty __init__.py files
find src/neuralflow -type d -exec touch {}/__init__.py \;

# Copy example environment file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
fi

# Create example configuration files
cat > config/environments/development.yaml << EOL
environment: development
debug: true
log_level: DEBUG
storage:
  data_dir: storage/data
  models_dir: storage/models
  vector_store_dir: storage/vector_store
  cache_dir: storage/cache
  state_dir: storage/state
logging:
  app_log: logs/app/app.log
  access_log: logs/access/access.log
  error_log: logs/error/error.log
monitoring:
  enabled: true
  interval: 30
  metrics_enabled: true
  tracing_enabled: true
memory:
  max_items: 1000
  ttl: 3600
  cleanup_interval: 300
  cache_enabled: true
  persistence_enabled: true
EOL

echo "Development environment setup complete!"
echo "Please activate the virtual environment with: source .venv/bin/activate" 