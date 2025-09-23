# AWS Agent Deployment Plan for FLIR+SCD41 Fire Detection System

## Current Status

The FLIR+SCD41 Fire Detection System is currently designed with a hybrid architecture where:
- **ML Models**: Deployed to AWS SageMaker endpoints (✅ Implemented)
- **Agents**: Run locally or on IoT devices (❌ Not deployed to AWS)

## Required Changes for AWS Agent Deployment

### 1. Containerize Agent Framework
Create Docker images for each agent type to run on AWS ECS/Fargate:

```dockerfile
# agents/Dockerfile.analysis
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY src/agents/analysis/ ./src/agents/analysis/
COPY src/agents/base.py ./src/agents/base.py
COPY src/ml/ ./src/ml/

# Install the package
COPY setup.py .
RUN pip install --no-cache-dir -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.agents.analysis.fire_pattern_analysis import FirePatternAnalysisAgent; print('OK')"

CMD ["python", "run_analysis_agent.py"]
```

### 2. Create AWS Lambda Functions for Event-Driven Agents

```python
# src/aws/lambda/monitoring_agent.py
import json
import boto3
from src.agents.monitoring.system_health import SystemHealthMonitor

def lambda_handler(event, context):
    """AWS Lambda function for Monitoring Agent"""
    
    # Initialize monitoring agent
    config = {
        'system_health_thresholds': {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0
        }
    }
    
    monitor = SystemHealthMonitor("lambda_monitoring_agent", config)
    
    # Process event data
    sensor_data = event.get('sensor_data', {})
    
    # Perform monitoring
    result = monitor.process(sensor_data)
    
    # Send results to SNS topic
    sns = boto3.client('sns')
    sns.publish(
        TopicArn='arn:aws:sns:us-east-1:691595239825:fire-detection-alerts',
        Message=json.dumps(result),
        Subject='System Health Monitoring Result'
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Monitoring completed',
            'result': result
        })
    }
```

### 3. Create ECS Task Definitions for Stateful Agents

```json
{
  "family": "fire-analysis-agent",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::691595239825:role/SaafeECSExecutionRole",
  "containerDefinitions": [
    {
      "name": "analysis-agent",
      "image": "691595239825.dkr.ecr.us-east-1.amazonaws.com/saafe-analysis-agent:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [
        {
          "name": "AWS_REGION",
          "value": "us-east-1"
        },
        {
          "name": "SAGEMAKER_ENDPOINT",
          "value": "flir-scd41-fire-detection-corrected-v3-20250901-121555"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fire-analysis-agent",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### 4. Implement Agent Communication with SQS/SNS

```python
# src/agents/aws_communication.py
import boto3
import json
from typing import Dict, Any

class AWSAgentCommunicator:
    """Handles communication between agents using AWS SQS/SNS"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.sqs = boto3.client('sqs', region_name=region)
        self.sns = boto3.client('sns', region_name=region)
        self.region = region
        
    def send_message_to_queue(self, queue_url: str, message: Dict[str, Any]) -> str:
        """Send message to SQS queue"""
        response = self.sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message)
        )
        return response['MessageId']
    
    def receive_messages_from_queue(self, queue_url: str, max_messages: int = 10) -> list:
        """Receive messages from SQS queue"""
        response = self.sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=20
        )
        return response.get('Messages', [])
    
    def publish_to_topic(self, topic_arn: str, message: Dict[str, Any], subject: str) -> str:
        """Publish message to SNS topic"""
        response = self.sns.publish(
            TopicArn=topic_arn,
            Message=json.dumps(message),
            Subject=subject
        )
        return response['MessageId']
```

### 5. Create CloudFormation Template for Agent Infrastructure

```yaml
# cloudformation/agents-infrastructure.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Infrastructure for FLIR+SCD41 Fire Detection Agents'

Parameters:
  EnvironmentName:
    Type: String
    Default: 'production'
    Description: 'Environment name'

Resources:
  # SQS Queues for agent communication
  AnalysisQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub '${EnvironmentName}-analysis-queue'
      VisibilityTimeoutSeconds: 300
      MessageRetentionPeriod: 1209600

  ResponseQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub '${EnvironmentName}-response-queue'
      VisibilityTimeoutSeconds: 300
      MessageRetentionPeriod: 1209600

  # SNS Topics for notifications
  AlertTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: !Sub '${EnvironmentName}-fire-alerts'
      DisplayName: 'Fire Detection Alerts'

  # ECS Cluster for containerized agents
  AgentCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub '${EnvironmentName}-agent-cluster'
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT
      DefaultCapacityProviderStrategy:
        - CapacityProvider: FARGATE
          Weight: 1

  # CloudWatch Log Groups
  AnalysisAgentLogs:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub '/ecs/${EnvironmentName}/analysis-agent'
      RetentionInDays: 14

  ResponseAgentLogs:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub '/ecs/${EnvironmentName}/response-agent'
      RetentionInDays: 14

Outputs:
  AnalysisQueueUrl:
    Description: 'URL of the analysis queue'
    Value: !Ref AnalysisQueue
    Export:
      Name: !Sub '${AWS::StackName}-AnalysisQueueUrl'

  ResponseQueueUrl:
    Description: 'URL of the response queue'
    Value: !Ref ResponseQueue
    Export:
      Name: !Sub '${AWS::StackName}-ResponseQueueUrl'

  AlertTopicArn:
    Description: 'ARN of the alert topic'
    Value: !Ref AlertTopic
    Export:
      Name: !Sub '${AWS::StackName}-AlertTopicArn'

  AgentClusterName:
    Description: 'Name of the agent cluster'
    Value: !Ref AgentCluster
    Export:
      Name: !Sub '${AWS::StackName}-AgentClusterName'
```

## Deployment Steps

### Phase 1: Containerization (1-2 days)
1. Create Docker images for each agent type
2. Push images to Amazon ECR
3. Test containerized agents locally

### Phase 2: Infrastructure Setup (2-3 days)
1. Deploy CloudFormation template for agent infrastructure
2. Create ECS task definitions
3. Set up IAM roles and policies
4. Configure CloudWatch logging

### Phase 3: Lambda Functions (2-3 days)
1. Create Lambda functions for event-driven agents
2. Configure event sources and triggers
3. Set up SNS topics and SQS queues
4. Test Lambda function integrations

### Phase 4: ECS Services (1-2 days)
1. Deploy ECS services for stateful agents
2. Configure auto-scaling policies
3. Set up service discovery
4. Test agent communication

### Phase 5: Integration Testing (2-3 days)
1. Test end-to-end agent workflows
2. Validate communication between agents
3. Verify integration with SageMaker endpoints
4. Performance and load testing

## Benefits of AWS Agent Deployment

1. **Scalability**: Automatic scaling based on demand
2. **Reliability**: High availability with multi-AZ deployment
3. **Cost-Effectiveness**: Pay only for resources used
4. **Monitoring**: Centralized logging and monitoring with CloudWatch
5. **Security**: IAM roles and policies for secure access
6. **Maintenance**: Managed services reduce operational overhead

## Estimated Timeline
- **Total Implementation Time**: 8-13 days
- **Infrastructure Cost**: ~$500-800/month (depending on usage)

## Next Steps
1. Create container images for agent components
2. Deploy CloudFormation template for infrastructure
3. Begin Lambda function development
4. Set up CI/CD pipeline for agent deployments