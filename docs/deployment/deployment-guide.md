# NeuralFlow Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- CPU: 4+ cores recommended
- RAM: 8GB minimum, 16GB recommended
- Storage: SSD with 20GB+ free space
- GPU: CUDA 11.0+ (optional, for GPU support)

### Software Requirements
- Python 3.8 or higher
- Redis 6.0 or higher
- Docker 20.10 or higher
- NVIDIA drivers (if using GPU)

### Environment Setup
1. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    redis-server \
    build-essential \
    git

# macOS
brew install python3 redis
```

2. Install CUDA (if using GPU):
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA Toolkit
# Follow instructions at https://developer.nvidia.com/cuda-downloads
```

## Local Development

### Virtual Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### Configuration
1. Copy example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your settings:
```ini
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379/0

# Model Settings
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS=4096
```

### Running the Application
```bash
# Development mode
python -m neuralflow.main --dev

# Debug mode
python -m neuralflow.main --debug

# Production mode
python -m neuralflow.main
```

## Docker Deployment

### Building the Image
```bash
# Build image
docker build -t neuralflow .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.9 -t neuralflow .
```

### Running Containers
```bash
# Basic run
docker run -p 8000:8000 neuralflow

# With environment variables
docker run -p 8000:8000 \
    -e OPENAI_API_KEY=your_key \
    -e DATABASE_URL=your_url \
    neuralflow

# With GPU support
docker run --gpus all -p 8000:8000 neuralflow
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://postgres:password@db:5432/neuralflow
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=neuralflow
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Cloud Deployment

### AWS Deployment

#### Prerequisites
- AWS CLI installed and configured
- ECR repository created
- ECS cluster configured

#### Steps
1. Build and push Docker image:
```bash
# Login to ECR
aws ecr get-login-password --region region | docker login --username AWS --password-stdin account.dkr.ecr.region.amazonaws.com

# Build and tag image
docker build -t neuralflow .
docker tag neuralflow:latest account.dkr.ecr.region.amazonaws.com/neuralflow:latest

# Push image
docker push account.dkr.ecr.region.amazonaws.com/neuralflow:latest
```

2. Deploy to ECS:
```bash
# Create task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Update service
aws ecs update-service --cluster your-cluster --service your-service --task-definition neuralflow:latest
```

### Kubernetes Deployment

#### Prerequisites
- kubectl installed and configured
- Kubernetes cluster running

#### Steps
1. Create Kubernetes manifests:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuralflow
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neuralflow
  template:
    metadata:
      labels:
        app: neuralflow
    spec:
      containers:
      - name: neuralflow
        image: neuralflow:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

2. Apply manifests:
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## Monitoring

### Prometheus Integration

1. Add Prometheus metrics:
```python
from prometheus_client import Counter, Histogram

requests_total = Counter('neuralflow_requests_total', 'Total requests')
response_time = Histogram('neuralflow_response_time_seconds', 'Response time')
```

2. Configure Prometheus:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'neuralflow'
    static_configs:
      - targets: ['localhost:8000']
```

### Logging

1. Configure logging:
```python
import logging

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'neuralflow.log',
            'level': 'DEBUG',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
})
```

## Troubleshooting

### Common Issues

1. Memory Issues
```bash
# Check memory usage
docker stats

# Increase container memory
docker run -m 4g neuralflow
```

2. Database Connection Issues
```bash
# Check database connection
psql $DATABASE_URL

# Check logs
docker logs container_id
```

3. GPU Issues
```bash
# Check GPU status
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Health Checks

1. Application health:
```bash
curl http://localhost:8000/health
```

2. Database health:
```bash
curl http://localhost:8000/health/db
```

3. Redis health:
```bash
curl http://localhost:8000/health/redis
``` 