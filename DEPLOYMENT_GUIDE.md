# Saafe Fire Detection System - Enterprise Deployment Guide

## Overview

This guide provides comprehensive deployment instructions for the Saafe Fire Detection System across multiple environments, following enterprise DevOps best practices with 20+ years of operational experience.

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores, 2.5GHz
- **Memory**: 8GB RAM
- **Storage**: 50GB SSD
- **Network**: 100Mbps bandwidth
- **OS**: Ubuntu 20.04 LTS, CentOS 8, or Amazon Linux 2

#### Recommended Requirements
- **CPU**: 8 cores, 3.0GHz
- **Memory**: 16GB RAM
- **Storage**: 100GB NVMe SSD
- **Network**: 1Gbps bandwidth
- **GPU**: NVIDIA GPU with CUDA support (optional)

#### Software Dependencies
```bash
# Core dependencies
Python 3.9+
Docker 20.10+
Docker Compose 2.0+
Git 2.30+
AWS CLI 2.0+

# Optional dependencies
NVIDIA Docker (for GPU support)
Kubernetes 1.21+ (for container orchestration)
Terraform 1.0+ (for infrastructure as code)
```

## Environment Setup

### 1. Development Environment

#### Local Development Setup
```bash
# Clone the repository
git clone https://github.com/AAA6666799/saafe.git
cd saafe

# Create and activate virtual environment
python3 -m venv saafe_env
source saafe_env/bin/activate  # Linux/macOS
# saafe_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp config/app_config.json.example config/app_config.json
cp config/user_config.json.example config/user_config.json

# Initialize the application
python main.py --setup

# Run the application
streamlit run app.py
```

#### Docker Development Setup
```bash
# Build the development image
docker build -f Dockerfile -t saafe:dev .

# Run with Docker Compose
docker-compose -f docker-compose.dev.yml up -d

# Access the application
open http://localhost:8501
```

### 2. Staging Environment

#### AWS Staging Deployment
```bash
# Set up AWS credentials
aws configure

# Deploy infrastructure
cd infrastructure/staging
terraform init
terraform plan
terraform apply

# Deploy application
cd ../../
./scripts/deploy-staging.sh

# Verify deployment
./scripts/health-check.sh staging
```

#### Staging Configuration
```yaml
# docker-compose.staging.yml
version: '3.8'
services:
  saafe-app:
    image: saafe:staging
    environment:
      - ENV=staging
      - LOG_LEVEL=INFO
      - METRICS_ENABLED=true
    ports:
      - "8501:8501"
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 3. Production Environment

#### Production Deployment Checklist
- [ ] Infrastructure provisioned and tested
- [ ] Security groups configured
- [ ] SSL certificates installed
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery tested
- [ ] Load balancing configured
- [ ] Auto-scaling policies set
- [ ] Disaster recovery plan validated

#### Production Infrastructure
```bash
# Production deployment
cd infrastructure/production
terraform init
terraform plan -var-file="production.tfvars"
terraform apply -var-file="production.tfvars"

# Deploy application with blue-green strategy
./scripts/deploy-production.sh --strategy=blue-green

# Validate deployment
./scripts/production-validation.sh
```

## Container Orchestration

### Docker Configuration

#### Multi-stage Dockerfile
```dockerfile
# Production Dockerfile
FROM python:3.9-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash saafe
USER saafe
WORKDIR /home/saafe/app

# Copy requirements and install Python dependencies
COPY --chown=saafe:saafe requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=saafe:saafe . .

# Set environment variables
ENV PYTHONPATH=/home/saafe/app
ENV PATH=/home/saafe/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Expose port
EXPOSE 8501

# Start application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Docker Compose Production
```yaml
version: '3.8'

services:
  saafe-app:
    build:
      context: .
      dockerfile: Dockerfile
    image: saafe:latest
    container_name: saafe-production
    environment:
      - ENV=production
      - LOG_LEVEL=WARNING
      - METRICS_ENABLED=true
      - SENTRY_DSN=${SENTRY_DSN}
    ports:
      - "8501:8501"
    volumes:
      - ./config:/home/saafe/app/config:ro
      - ./logs:/home/saafe/app/logs
      - ./models:/home/saafe/app/models:ro
    restart: unless-stopped
    networks:
      - saafe-network
    depends_on:
      - redis
      - prometheus

  redis:
    image: redis:7-alpine
    container_name: saafe-redis
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - saafe-network
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: saafe-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - saafe-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: saafe-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    networks:
      - saafe-network
    restart: unless-stopped

networks:
  saafe-network:
    driver: bridge

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
```

## AWS Cloud Deployment

### Infrastructure as Code (Terraform)

#### Main Infrastructure
```hcl
# infrastructure/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "saafe-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "Saafe"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "DevOps Team"
    }
  }
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "saafe-${var.environment}"
  cidr = var.vpc_cidr
  
  azs             = var.availability_zones
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true
  
  tags = {
    Name = "saafe-${var.environment}-vpc"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "saafe" {
  name = "saafe-${var.environment}"
  
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  
  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight           = 1
  }
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Application Load Balancer
resource "aws_lb" "saafe" {
  name               = "saafe-${var.environment}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = var.environment == "production"
  
  access_logs {
    bucket  = aws_s3_bucket.logs.bucket
    prefix  = "alb"
    enabled = true
  }
}

# ECS Service
resource "aws_ecs_service" "saafe" {
  name            = "saafe-${var.environment}"
  cluster         = aws_ecs_cluster.saafe.id
  task_definition = aws_ecs_task_definition.saafe.arn
  desired_count   = var.desired_count
  
  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight           = 100
  }
  
  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets         = module.vpc.private_subnets
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.saafe.arn
    container_name   = "saafe-app"
    container_port   = 8501
  }
  
  depends_on = [aws_lb_listener.saafe]
}
```

### CI/CD Pipeline (AWS CodePipeline)

#### BuildSpec Configuration
```yaml
# buildspec.yml
version: 0.2

phases:
  pre_build:
    commands:
      - echo Logging in to Amazon ECR...
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
      - REPOSITORY_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME
      - COMMIT_HASH=$(echo $CODEBUILD_RESOLVED_SOURCE_VERSION | cut -c 1-7)
      - IMAGE_TAG=${COMMIT_HASH:=latest}
  
  build:
    commands:
      - echo Build started on `date`
      - echo Building the Docker image...
      - docker build -t $IMAGE_REPO_NAME:$IMAGE_TAG .
      - docker tag $IMAGE_REPO_NAME:$IMAGE_TAG $REPOSITORY_URI:$IMAGE_TAG
      - docker tag $IMAGE_REPO_NAME:$IMAGE_TAG $REPOSITORY_URI:latest
      
      - echo Running tests...
      - docker run --rm $IMAGE_REPO_NAME:$IMAGE_TAG python -m pytest tests/ -v
      
      - echo Running security scan...
      - docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image $IMAGE_REPO_NAME:$IMAGE_TAG
  
  post_build:
    commands:
      - echo Build completed on `date`
      - echo Pushing the Docker images...
      - docker push $REPOSITORY_URI:$IMAGE_TAG
      - docker push $REPOSITORY_URI:latest
      - echo Writing image definitions file...
      - printf '[{"name":"saafe-app","imageUri":"%s"}]' $REPOSITORY_URI:$IMAGE_TAG > imagedefinitions.json

artifacts:
  files:
    - imagedefinitions.json
    - infrastructure/**/*
    - scripts/**/*
```

## Monitoring and Observability

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'saafe-app'
    static_configs:
      - targets: ['saafe-app:8501']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Saafe Fire Detection System",
    "panels": [
      {
        "title": "Fire Detection Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(saafe_fire_detections_total[5m])"
          }
        ]
      },
      {
        "title": "Alert Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, saafe_alert_response_time_seconds_bucket)"
          }
        ]
      }
    ]
  }
}
```

## Security Configuration

### Security Groups
```hcl
# Security group for ALB
resource "aws_security_group" "alb" {
  name_prefix = "saafe-alb-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Security group for ECS tasks
resource "aws_security_group" "ecs_tasks" {
  name_prefix = "saafe-ecs-tasks-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 8501
    to_port         = 8501
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

## Backup and Recovery

### Automated Backup Strategy
```bash
#!/bin/bash
# scripts/backup.sh

set -euo pipefail

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_BUCKET="saafe-backups-${ENVIRONMENT}"

# Backup configuration files
echo "Backing up configuration..."
aws s3 sync config/ s3://${BACKUP_BUCKET}/config/${BACKUP_DATE}/

# Backup models
echo "Backing up models..."
aws s3 sync models/ s3://${BACKUP_BUCKET}/models/${BACKUP_DATE}/

# Backup logs (last 7 days)
echo "Backing up recent logs..."
find logs/ -name "*.log" -mtime -7 -exec aws s3 cp {} s3://${BACKUP_BUCKET}/logs/${BACKUP_DATE}/ \;

# Create backup manifest
cat > backup_manifest.json << EOF
{
  "backup_date": "${BACKUP_DATE}",
  "environment": "${ENVIRONMENT}",
  "components": ["config", "models", "logs"],
  "retention_days": 90
}
EOF

aws s3 cp backup_manifest.json s3://${BACKUP_BUCKET}/manifests/${BACKUP_DATE}.json

echo "Backup completed successfully"
```

## Performance Tuning

### Application Optimization
```python
# config/performance.py
PERFORMANCE_CONFIG = {
    "streamlit": {
        "server.maxUploadSize": 200,
        "server.maxMessageSize": 200,
        "server.enableCORS": False,
        "server.enableXsrfProtection": True,
        "browser.gatherUsageStats": False
    },
    "pytorch": {
        "torch.set_num_threads": 4,
        "torch.set_num_interop_threads": 2
    },
    "caching": {
        "ttl": 300,
        "max_entries": 1000
    }
}
```

### Resource Limits
```yaml
# ECS Task Definition Resource Limits
resources:
  limits:
    cpu: "2000m"      # 2 vCPU
    memory: "4096Mi"   # 4GB RAM
  requests:
    cpu: "1000m"      # 1 vCPU
    memory: "2048Mi"   # 2GB RAM
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: High Memory Usage
```bash
# Check memory usage
docker stats saafe-production

# Solution: Increase memory limits or optimize code
# Update docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 8G
```

#### Issue: Slow Model Inference
```bash
# Check GPU availability
nvidia-smi

# Solution: Enable GPU support
# Update Dockerfile:
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
```

#### Issue: Connection Timeouts
```bash
# Check network connectivity
curl -I http://localhost:8501/health

# Solution: Adjust timeout settings
# Update load balancer configuration
```

## Maintenance Procedures

### Regular Maintenance Tasks
```bash
#!/bin/bash
# scripts/maintenance.sh

# Weekly maintenance routine
echo "Starting weekly maintenance..."

# Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete

# Update dependencies
pip list --outdated

# Check disk usage
df -h

# Validate backups
./scripts/validate-backups.sh

# Run health checks
./scripts/health-check.sh production

echo "Maintenance completed"
```

### Update Procedures
```bash
#!/bin/bash
# scripts/update.sh

# Zero-downtime update procedure
echo "Starting application update..."

# Build new image
docker build -t saafe:new .

# Run tests
docker run --rm saafe:new python -m pytest

# Blue-green deployment
./scripts/blue-green-deploy.sh saafe:new

echo "Update completed successfully"
```

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Classification**: Operations Manual  
**Owner**: DevOps Engineering Team