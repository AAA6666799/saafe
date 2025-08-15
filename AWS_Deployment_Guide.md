# Saafe Fire Detection - AWS Deployment Guide

## Overview
Complete step-by-step guide to deploy your Saafe AI system to AWS cloud infrastructure.

## Current System Analysis
- **Standalone Application**: Streamlit-based with PyTorch AI models
- **Architecture**: Spatio-Temporal Transformer with anti-hallucination
- **Size**: ~7.18M parameters, 28.7MB model files
- **Performance**: <50ms inference, 98.7% accuracy
- **Dependencies**: PyTorch, Streamlit, Plotly, Twilio, SendGrid

## AWS Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS Cloud                               │
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │   AWS IoT       │───▶│  Kinesis Data    │───▶│ Lambda      │ │
│  │   Core          │    │  Streams         │    │ Functions   │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
│           │                       │                      │      │
│           ▼                       ▼                      ▼      │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │  Device         │    │   SageMaker      │    │ DynamoDB    │ │
│  │ Management      │    │   Endpoints      │    │             │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
│                                   │                      │      │
│                                   ▼                      ▼      │
│                          ┌──────────────────┐    ┌─────────────┐ │
│                          │  ECS Fargate     │    │ CloudWatch  │ │
│                          │  Containers      │    │ Monitoring  │ │
│                          └──────────────────┘    └─────────────┘ │
│                                   │                              │
│                                   ▼                              │
│                          ┌──────────────────┐                   │
│                          │  Application     │                   │
│                          │  Load Balancer   │                   │
│                          └──────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: AWS Account Setup & Prerequisites

### 1.1 AWS Account Preparation
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
# AWS Access Key ID: [Your Access Key]
# AWS Secret Access Key: [Your Secret Key]
# Default region name: eu-west-1
# Default output format: json
```

### 1.2 AWS CodeArtifact Setup
```bash
# Login to CodeArtifact (your existing command)
aws codeartifact login --tool pip --repository saafe --domain saafeai --domain-owner 691595239825 --region eu-west-1

# Run automated setup script
python setup_codeartifact.py

# Verify CodeArtifact access
pip install --dry-run boto3
```#
## 1.2 Required AWS Services Setup
```bash
# Enable required AWS services
aws iam create-role --role-name SaafeExecutionRole --assume-role-policy-document file://trust-policy.json
aws iam attach-role-policy --role-name SaafeExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam attach-role-policy --role-name SaafeExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

### 1.3 Create Trust Policy File
```json
# trust-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": [
          "ecs-tasks.amazonaws.com",
          "sagemaker.amazonaws.com",
          "lambda.amazonaws.com"
        ]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

## Phase 2: Containerize Your Application

### 2.1 Create Dockerfile with CodeArtifact
```dockerfile
# Dockerfile-codeartifact (generated by setup script)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Copy requirements first for better caching
COPY requirements-codeartifact.txt .

# Setup CodeArtifact authentication
ARG AWS_ACCOUNT_ID=691595239825
ARG AWS_REGION=eu-west-1
ARG CODEARTIFACT_DOMAIN=saafeai
ARG CODEARTIFACT_REPO=saafe

# Login to CodeArtifact and install packages
RUN aws codeartifact login --tool pip \
    --repository $CODEARTIFACT_REPO \
    --domain $CODEARTIFACT_DOMAIN \
    --domain-owner $AWS_ACCOUNT_ID \
    --region $AWS_REGION \
    && pip install --no-cache-dir -r requirements-codeartifact.txt

# Copy application code
COPY saafe_mvp/ ./saafe_mvp/
COPY models/ ./models/
COPY config/ ./config/
COPY app.py .
COPY main.py .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2.2 Update Requirements for AWS CodeArtifact
```txt
# requirements-codeartifact.txt (generated by setup script)
# Saafe MVP - CodeArtifact Requirements

torch>=2.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
pandas>=1.3.0
scipy>=1.7.0
streamlit>=1.28.0
plotly>=5.0.0
twilio>=8.0.0
sendgrid>=6.0.0
psutil>=5.8.0

# AWS-specific packages
boto3>=1.26.0
botocore>=1.29.0
awscli>=2.0.0
sagemaker>=2.0.0

# Additional packages for AWS deployment
gunicorn>=20.1.0
uvicorn>=0.18.0
fastapi>=0.95.0
```

### 2.3 Build and Test Container Locally
```bash
# Build Docker image
docker build -t saafe-mvp:latest .

# Test locally
docker run -p 8501:8501 saafe-mvp:latest

# Test health endpoint
curl http://localhost:8501/_stcore/health
```##
 Phase 3: AWS ECR (Container Registry) Setup

### 3.1 Create ECR Repository
```bash
# Create ECR repository
aws ecr create-repository --repository-name saafe-mvp --region us-east-1

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
```

### 3.2 Push Container to ECR
```bash
# Tag image for ECR
docker tag saafe-mvp:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/saafe-mvp:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/saafe-mvp:latest
```

## Phase 4: AWS SageMaker Model Deployment

### 4.1 Prepare Model for SageMaker
```python
# sagemaker_deploy.py
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
import tarfile
import os

def prepare_model_artifacts():
    """Package model files for SageMaker"""
    
    # Create model.tar.gz
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        tar.add('models/transformer_model.pth', arcname='transformer_model.pth')
        tar.add('models/anti_hallucination.pkl', arcname='anti_hallucination.pkl')
        tar.add('models/model_metadata.json', arcname='model_metadata.json')
        tar.add('saafe_mvp/models/', arcname='code/')
    
    # Upload to S3
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    model_artifacts = sagemaker_session.upload_data(
        path='model.tar.gz',
        bucket=bucket,
        key_prefix='saafe-models'
    )
    
    return model_artifacts

def deploy_sagemaker_endpoint():
    """Deploy model to SageMaker endpoint"""
    
    model_artifacts = prepare_model_artifacts()
    
    # Create PyTorch model
    pytorch_model = PyTorchModel(
        model_data=model_artifacts,
        role='arn:aws:iam::<account-id>:role/SaafeExecutionRole',
        entry_point='inference.py',
        source_dir='sagemaker_code',
        framework_version='2.0.0',
        py_version='py39'
    )
    
    # Deploy endpoint
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name='saafe-endpoint'
    )
    
    return predictor

if __name__ == "__main__":
    predictor = deploy_sagemaker_endpoint()
    print(f"Endpoint deployed: {predictor.endpoint_name}")
```

### 4.2 Create SageMaker Inference Script
```python
# sagemaker_code/inference.py
import torch
import json
import numpy as np
from saafe_mvp.models.transformer import SpatioTemporalTransformer, ModelConfig
from saafe_mvp.models.anti_hallucination import AntiHallucinationEngine

def model_fn(model_dir):
    """Load model for SageMaker"""
    
    # Load transformer model
    config = ModelConfig()
    model = SpatioTemporalTransformer(config)
    model.load_state_dict(torch.load(f'{model_dir}/transformer_model.pth', map_location='cpu'))
    model.eval()
    
    # Load anti-hallucination engine
    import pickle
    with open(f'{model_dir}/anti_hallucination.pkl', 'rb') as f:
        anti_hallucination = pickle.load(f)
    
    return {
        'transformer': model,
        'anti_hallucination': anti_hallucination
    }

def input_fn(request_body, request_content_type):
    """Parse input data"""
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Convert sensor readings to tensor
        sensor_data = np.array(input_data['sensor_readings'])
        return torch.FloatTensor(sensor_data).unsqueeze(0)  # Add batch dimension
    
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Run inference"""
    
    transformer = model_dict['transformer']
    anti_hallucination = model_dict['anti_hallucination']
    
    with torch.no_grad():
        # Primary model inference
        outputs = transformer(input_data)
        risk_score = float(outputs['risk_score'].squeeze())
        
        # Anti-hallucination validation
        validation_result = anti_hallucination.validate_fire_prediction(
            risk_score, input_data.squeeze(0)
        )
        
        # Apply confidence adjustment
        final_risk_score = risk_score * validation_result.confidence_adjustment
        
        return {
            'risk_score': final_risk_score,
            'confidence': float(outputs.get('confidence', 0.0)),
            'predicted_class': 'normal' if final_risk_score < 30 else 'elevated' if final_risk_score < 85 else 'critical',
            'validation_result': {
                'is_valid': validation_result.is_valid,
                'reasoning': validation_result.reasoning,
                'cooking_detected': validation_result.cooking_detected
            }
        }

def output_fn(prediction, content_type):
    """Format output"""
    
    if content_type == 'application/json':
        return json.dumps(prediction)
    
    raise ValueError(f"Unsupported content type: {content_type}")
```## Phase 5
: AWS ECS Fargate Deployment

### 5.1 Create ECS Task Definition
```json
{
  "family": "saafe-mvp-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "3072",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/SaafeExecutionRole",
  "taskRoleArn": "arn:aws:iam::<account-id>:role/SaafeExecutionRole",
  "containerDefinitions": [
    {
      "name": "saafe-container",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/saafe-mvp:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/saafe-mvp",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "environment": [
        {
          "name": "AWS_DEFAULT_REGION",
          "value": "us-east-1"
        },
        {
          "name": "SAGEMAKER_ENDPOINT",
          "value": "saafe-endpoint"
        }
      ],
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8501/_stcore/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### 5.2 Create ECS Cluster and Service
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name saafe-cluster --capacity-providers FARGATE

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create ECS service
aws ecs create-service \
  --cluster saafe-cluster \
  --service-name saafe-service \
  --task-definition saafe-mvp-task:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

## Phase 6: AWS IoT Core Integration

### 6.1 Create IoT Thing Types and Policies
```bash
# Create IoT thing type for fire sensors
aws iot create-thing-type \
  --thing-type-name FireSensor \
  --thing-type-description "Fire detection sensors"

# Create IoT policy
aws iot create-policy \
  --policy-name SaafeSensorPolicy \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "iot:Connect",
          "iot:Publish",
          "iot:Subscribe",
          "iot:Receive"
        ],
        "Resource": "*"
      }
    ]
  }'
```

### 6.2 Create IoT Rule for Data Processing
```bash
# Create IoT rule to process sensor data
aws iot create-topic-rule \
  --rule-name SaafeDataProcessing \
  --topic-rule-payload '{
    "sql": "SELECT * FROM \"topic/saafe/sensors\"",
    "description": "Process fire sensor data",
    "actions": [
      {
        "kinesis": {
          "streamName": "saafe-sensor-stream",
          "partitionKey": "${deviceId}",
          "roleArn": "arn:aws:iam::<account-id>:role/SaafeExecutionRole"
        }
      }
    ]
  }'
```

## Phase 7: AWS Kinesis Data Streams

### 7.1 Create Kinesis Stream
```bash
# Create Kinesis data stream
aws kinesis create-stream \
  --stream-name saafe-sensor-stream \
  --shard-count 2

# Wait for stream to be active
aws kinesis wait stream-exists --stream-name saafe-sensor-stream
```

### 7.2 Create Lambda Function for Stream Processing
```python
# lambda_processor.py
import json
import boto3
import base64
from datetime import datetime

sagemaker_runtime = boto3.client('sagemaker-runtime')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('SaafePredictions')

def lambda_handler(event, context):
    """Process Kinesis stream records"""
    
    for record in event['Records']:
        # Decode Kinesis data
        payload = base64.b64decode(record['kinesis']['data'])
        sensor_data = json.loads(payload)
        
        # Prepare data for SageMaker
        model_input = {
            'sensor_readings': [
                [
                    sensor_data['temperature'],
                    sensor_data['pm25'],
                    sensor_data['co2'],
                    sensor_data['audio_level']
                ]
            ] * 60  # Repeat for 60 timesteps (simplified)
        }
        
        # Call SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName='saafe-endpoint',
            ContentType='application/json',
            Body=json.dumps(model_input)
        )
        
        # Parse prediction
        prediction = json.loads(response['Body'].read().decode())
        
        # Store in DynamoDB
        table.put_item(
            Item={
                'device_id': sensor_data['device_id'],
                'timestamp': datetime.now().isoformat(),
                'risk_score': prediction['risk_score'],
                'predicted_class': prediction['predicted_class'],
                'sensor_data': sensor_data,
                'validation_result': prediction['validation_result']
            }
        )
        
        # Trigger alerts if necessary
        if prediction['risk_score'] > 85:
            send_critical_alert(sensor_data, prediction)
    
    return {'statusCode': 200, 'body': 'Processed successfully'}

def send_critical_alert(sensor_data, prediction):
    """Send critical fire alert"""
    
    sns = boto3.client('sns')
    
    message = f"""
    🚨 CRITICAL FIRE ALERT 🚨
    
    Device: {sensor_data['device_id']}
    Risk Score: {prediction['risk_score']:.1f}/100
    Classification: {prediction['predicted_class']}
    
    Sensor Readings:
    - Temperature: {sensor_data['temperature']}°C
    - PM2.5: {sensor_data['pm25']} μg/m³
    - CO₂: {sensor_data['co2']} ppm
    - Audio: {sensor_data['audio_level']} dB
    
    Validation: {prediction['validation_result']['reasoning']}
    """
    
    sns.publish(
        TopicArn='arn:aws:sns:us-east-1:<account-id>:SaafeAlerts',
        Message=message,
        Subject='🚨 CRITICAL FIRE ALERT'
    )
```## Ph
ase 8: AWS DynamoDB Setup

### 8.1 Create DynamoDB Tables
```bash
# Create predictions table
aws dynamodb create-table \
  --table-name SaafePredictions \
  --attribute-definitions \
    AttributeName=device_id,AttributeType=S \
    AttributeName=timestamp,AttributeType=S \
  --key-schema \
    AttributeName=device_id,KeyType=HASH \
    AttributeName=timestamp,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST

# Create device management table
aws dynamodb create-table \
  --table-name SaafeDevices \
  --attribute-definitions \
    AttributeName=device_id,AttributeType=S \
  --key-schema \
    AttributeName=device_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

### 8.2 Create Global Secondary Indexes
```bash
# Add GSI for time-based queries
aws dynamodb update-table \
  --table-name SaafePredictions \
  --attribute-definitions \
    AttributeName=risk_level,AttributeType=S \
    AttributeName=timestamp,AttributeType=S \
  --global-secondary-index-updates \
    '[{
      "Create": {
        "IndexName": "RiskLevelIndex",
        "KeySchema": [
          {"AttributeName": "risk_level", "KeyType": "HASH"},
          {"AttributeName": "timestamp", "KeyType": "RANGE"}
        ],
        "Projection": {"ProjectionType": "ALL"},
        "BillingMode": "PAY_PER_REQUEST"
      }
    }]'
```

## Phase 9: Application Load Balancer & Auto Scaling

### 9.1 Create Application Load Balancer
```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name saafe-alb \
  --subnets subnet-12345 subnet-67890 \
  --security-groups sg-12345

# Create target group
aws elbv2 create-target-group \
  --name saafe-targets \
  --protocol HTTP \
  --port 8501 \
  --vpc-id vpc-12345 \
  --target-type ip \
  --health-check-path /_stcore/health

# Create listener
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:us-east-1:<account-id>:loadbalancer/app/saafe-alb/1234567890 \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:us-east-1:<account-id>:targetgroup/saafe-targets/1234567890
```

### 9.2 Configure Auto Scaling
```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/saafe-cluster/saafe-service \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10

# Create scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id service/saafe-cluster/saafe-service \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name saafe-scaling-policy \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    }
  }'
```

## Phase 10: Monitoring & Logging

### 10.1 CloudWatch Setup
```bash
# Create log group
aws logs create-log-group --log-group-name /ecs/saafe-mvp

# Create custom metrics
aws cloudwatch put-metric-alarm \
  --alarm-name "SaafeHighRiskScore" \
  --alarm-description "Alert when risk score is high" \
  --metric-name RiskScore \
  --namespace Saafe \
  --statistic Average \
  --period 300 \
  --threshold 85 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1
```

### 10.2 Custom Metrics Lambda
```python
# cloudwatch_metrics.py
import boto3
import json
from datetime import datetime

cloudwatch = boto3.client('cloudwatch')

def lambda_handler(event, context):
    """Send custom metrics to CloudWatch"""
    
    for record in event['Records']:
        # Parse DynamoDB stream record
        if record['eventName'] in ['INSERT', 'MODIFY']:
            item = record['dynamodb']['NewImage']
            
            risk_score = float(item['risk_score']['N'])
            device_id = item['device_id']['S']
            
            # Send custom metrics
            cloudwatch.put_metric_data(
                Namespace='Saafe',
                MetricData=[
                    {
                        'MetricName': 'RiskScore',
                        'Dimensions': [
                            {
                                'Name': 'DeviceId',
                                'Value': device_id
                            }
                        ],
                        'Value': risk_score,
                        'Unit': 'None',
                        'Timestamp': datetime.now()
                    },
                    {
                        'MetricName': 'PredictionCount',
                        'Value': 1,
                        'Unit': 'Count',
                        'Timestamp': datetime.now()
                    }
                ]
            )
    
    return {'statusCode': 200}
```

## Phase 11: Security & Compliance

### 11.1 VPC Security Groups
```bash
# Create security group for ECS tasks
aws ec2 create-security-group \
  --group-name saafe-ecs-sg \
  --description "Security group for Saafe ECS tasks"

# Allow inbound HTTP traffic
aws ec2 authorize-security-group-ingress \
  --group-id sg-12345 \
  --protocol tcp \
  --port 8501 \
  --source-group sg-67890  # ALB security group
```

### 11.2 IAM Policies
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:InvokeEndpoint",
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query",
        "sns:Publish",
        "cloudwatch:PutMetricData",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

## Phase 12: Deployment Scripts

### 12.1 Complete Deployment Script
```bash
#!/bin/bash
# deploy.sh

set -e

echo "🚀 Starting Saafe AWS Deployment..."

# Variables
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION="us-east-1"
ECR_REPO="saafe-mvp"

echo "Account ID: $ACCOUNT_ID"
echo "Region: $REGION"

# Step 1: Build and push Docker image
echo "📦 Building Docker image..."
docker build -t $ECR_REPO:latest .

echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

echo "🏷️ Tagging image..."
docker tag $ECR_REPO:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest

echo "⬆️ Pushing to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest

# Step 2: Deploy SageMaker endpoint
echo "🤖 Deploying SageMaker endpoint..."
python sagemaker_deploy.py

# Step 3: Create ECS resources
echo "🐳 Creating ECS cluster..."
aws ecs create-cluster --cluster-name saafe-cluster --capacity-providers FARGATE

echo "📋 Registering task definition..."
aws ecs register-task-definition --cli-input-json file://task-definition.json

echo "🚀 Creating ECS service..."
aws ecs create-service \
  --cluster saafe-cluster \
  --service-name saafe-service \
  --task-definition saafe-mvp-task:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-12345],assignPublicIp=ENABLED}"

# Step 4: Wait for service to be stable
echo "⏳ Waiting for service to stabilize..."
aws ecs wait services-stable --cluster saafe-cluster --services saafe-service

echo "✅ Deployment completed successfully!"
echo "🌐 Application will be available at the ALB endpoint"
```

### 12.2 Environment Configuration
```bash
# .env
AWS_ACCOUNT_ID=123456789012
AWS_REGION=us-east-1
ECR_REPOSITORY=saafe-mvp
ECS_CLUSTER=saafe-cluster
SAGEMAKER_ENDPOINT=saafe-endpoint
DYNAMODB_TABLE=SaafePredictions
SNS_TOPIC_ARN=arn:aws:sns:us-east-1:123456789012:SaafeAlerts
```

## Phase 13: Testing & Validation

### 13.1 End-to-End Testing
```python
# test_aws_deployment.py
import boto3
import json
import requests
import time

def test_sagemaker_endpoint():
    """Test SageMaker endpoint"""
    
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    
    test_data = {
        'sensor_readings': [[24.5, 15.2, 410.0, 45.0]] * 60
    }
    
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName='saafe-endpoint',
        ContentType='application/json',
        Body=json.dumps(test_data)
    )
    
    result = json.loads(response['Body'].read().decode())
    print(f"SageMaker prediction: {result}")
    
    assert 'risk_score' in result
    assert 0 <= result['risk_score'] <= 100

def test_iot_pipeline():
    """Test IoT data pipeline"""
    
    iot_data = boto3.client('iot-data')
    
    test_message = {
        'device_id': 'test-sensor-001',
        'timestamp': time.time(),
        'temperature': 24.5,
        'pm25': 15.2,
        'co2': 410.0,
        'audio_level': 45.0
    }
    
    iot_data.publish(
        topic='topic/saafe/sensors',
        payload=json.dumps(test_message)
    )
    
    print("IoT message published successfully")

def test_web_application():
    """Test web application endpoint"""
    
    # Get ALB DNS name
    elbv2 = boto3.client('elbv2')
    response = elbv2.describe_load_balancers(Names=['saafe-alb'])
    alb_dns = response['LoadBalancers'][0]['DNSName']
    
    # Test health endpoint
    health_response = requests.get(f'http://{alb_dns}/_stcore/health')
    assert health_response.status_code == 200
    
    print(f"Web application healthy at: http://{alb_dns}")

if __name__ == "__main__":
    test_sagemaker_endpoint()
    test_iot_pipeline()
    test_web_application()
    print("✅ All tests passed!")
```

## Summary

Your Saafe system is now deployed on AWS with:

- **Containerized Application**: ECS Fargate with auto-scaling
- **AI Model Serving**: SageMaker endpoints for inference
- **IoT Integration**: AWS IoT Core for sensor data
- **Real-time Processing**: Kinesis + Lambda for stream processing
- **Data Storage**: DynamoDB for predictions and device management
- **Monitoring**: CloudWatch metrics and alarms
- **Security**: VPC, IAM roles, and security groups
- **High Availability**: Multi-AZ deployment with load balancing

**Estimated Monthly Cost**: $800-1200 for moderate usage
**Deployment Time**: 2-3 hours with the provided scripts
**Scalability**: Auto-scales from 2 to 10 instances based on demand