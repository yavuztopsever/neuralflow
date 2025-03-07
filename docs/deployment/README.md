# NeuralFlow Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying NeuralFlow in various environments, from local development to production.

## Deployment Options

### 1. Local Development
- Docker Compose
- Direct Python installation
- Virtual environment

### 2. Production
- Kubernetes cluster
- Docker Swarm
- Cloud platforms (AWS, GCP, Azure)

## Prerequisites

### System Requirements
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+
- OS: Linux (Ubuntu 20.04+ recommended)

### Software Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.8+
- Git

## Local Development Deployment

### 1. Using Docker Compose

```bash
# Clone repository
git clone https://github.com/yavuztopsever/neuralflow.git
cd neuralflow

# Build and start services
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 2. Direct Python Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python src/main.py
```

## Production Deployment

### 1. Docker Deployment

#### Build Docker Image
```bash
# Build image
docker build -t neuralflow:latest .

# Push to registry
docker tag neuralflow:latest your-registry/neuralflow:latest
docker push your-registry/neuralflow:latest
```

#### Docker Compose Production
```yaml
version: '3.8'

services:
  neuralflow:
    image: neuralflow:latest
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://user:pass@db:5432/neuralflow
      - REDIS_URL=redis://redis:6379/0
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
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

### 2. Kubernetes Deployment

#### Deployment Configuration
```yaml
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
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: neuralflow-secrets
              key: database-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

#### Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: neuralflow
spec:
  selector:
    app: neuralflow
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 3. Cloud Platform Deployment

#### AWS ECS Deployment
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name neuralflow-cluster

# Create task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service --cluster neuralflow-cluster --service-name neuralflow --task-definition neuralflow
```

#### GCP Cloud Run Deployment
```bash
# Build and push container
gcloud builds submit --tag gcr.io/your-project/neuralflow

# Deploy to Cloud Run
gcloud run deploy neuralflow \
  --image gcr.io/your-project/neuralflow \
  --platform managed \
  --region your-region
```

## Configuration

### Environment Variables
```bash
# Required
ENVIRONMENT=production
DATABASE_URL=postgresql://user:pass@db:5432/neuralflow
REDIS_URL=redis://redis:6379/0

# Optional
LOG_LEVEL=INFO
API_KEY=your-api-key
MAX_WORKERS=4
```

### Security Configuration

#### SSL/TLS Setup
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Authentication
```python
# JWT Configuration
JWT_SECRET_KEY = "your-secret-key"
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
```

## Monitoring

### 1. Logging Setup
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neuralflow.log'),
        logging.StreamHandler()
    ]
)
```

### 2. Metrics Collection
```python
from prometheus_client import Counter, Histogram

# Define metrics
request_counter = Counter('http_requests_total', 'Total HTTP requests')
request_latency = Histogram('http_request_duration_seconds', 'HTTP request latency')
```

### 3. Health Checks
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": check_database(),
        "redis": check_redis(),
        "version": "1.0.0"
    }
```

## Backup and Recovery

### 1. Database Backup
```bash
# Backup PostgreSQL database
pg_dump -U user -d neuralflow > backup.sql

# Restore from backup
psql -U user -d neuralflow < backup.sql
```

### 2. Configuration Backup
```bash
# Backup configuration files
tar -czf config_backup.tar.gz config/

# Restore configuration
tar -xzf config_backup.tar.gz
```

## Scaling

### 1. Horizontal Scaling
```bash
# Scale Docker Compose services
docker-compose up -d --scale neuralflow=3

# Scale Kubernetes deployment
kubectl scale deployment neuralflow --replicas=3
```

### 2. Load Balancing
```yaml
# Nginx load balancer configuration
upstream neuralflow {
    least_conn;
    server neuralflow1:8000;
    server neuralflow2:8000;
    server neuralflow3:8000;
}
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   - Check database credentials
   - Verify network connectivity
   - Check database logs

2. **Memory Issues**
   - Monitor memory usage
   - Adjust worker count
   - Check for memory leaks

3. **Performance Issues**
   - Monitor response times
   - Check resource usage
   - Optimize database queries

### Debug Tools

1. **Log Analysis**
```bash
# View application logs
docker-compose logs -f neuralflow

# View database logs
docker-compose logs -f db
```

2. **Resource Monitoring**
```bash
# Monitor container resources
docker stats

# Monitor Kubernetes resources
kubectl top pods
```

## Support

For deployment support:
- Email: deploy@neuralflow.com
- Documentation: https://docs.neuralflow.com/deployment
- GitHub Issues: https://github.com/neuralflow/neuralflow/issues 