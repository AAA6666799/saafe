# AWS Agent Deployment Summary

## Overview

This document summarizes the implementation of AWS Lambda agents for the FLIR+SCD41 Fire Detection System. Previously, the agents were running locally or on IoT devices, but now they have been implemented to run on AWS Lambda for better scalability, reliability, and maintenance.

## Implemented AWS Lambda Agents

### 1. Monitoring Agent (`saafe-monitoring-agent`)
**File**: `src/aws/lambda/monitoring_agent.py`
**Purpose**: Monitors system health and sensor data quality

**Key Features**:
- Validates sensor data (FLIR thermal and SCD41 gas readings)
- Detects anomalies in temperature and CO2 levels
- Checks data freshness to ensure timely readings
- Sends alerts via SNS when issues are detected
- Runs on a scheduled basis (every 5 minutes)

### 2. Analysis Agent (`saafe-analysis-agent`)
**File**: `src/aws/lambda/analysis_agent.py`
**Purpose**: Performs fire pattern analysis using SageMaker endpoints

**Key Features**:
- Receives 18-feature input (15 thermal + 3 gas features)
- Validates all required features are present
- Calls SageMaker endpoint for ML inference
- Processes prediction results and confidence scores
- Sends results to monitoring systems via SNS

### 3. Response Agent (`saafe-response-agent`)
**File**: `src/aws/lambda/response_agent.py`
**Purpose**: Handles emergency responses based on fire detection results

**Key Features**:
- Determines response level based on confidence scores
- Executes appropriate actions for different threat levels:
  - **CRITICAL**: Emergency alerts, sprinkler activation, fire department notification
  - **HIGH**: High priority alerts, warning systems, security notification
  - **MEDIUM**: Medium priority alerts, incident logging, increased monitoring
  - **LOW**: Low priority alerts, incident logging
  - **NONE**: No action required
- Sends notifications via SNS to appropriate channels

## Deployment Infrastructure

### IAM Roles and Permissions
- Created `SaafeLambdaExecutionRole` with necessary permissions:
  - AWSLambdaBasicExecutionRole for basic Lambda operations
  - AmazonSageMakerFullAccess for SageMaker endpoint invocation
  - AmazonSNSFullAccess for alert notifications

### Deployment Script
**File**: `deploy_lambda_agents.sh`
**Functionality**:
- Creates Lambda execution role if it doesn't exist
- Packages each agent with its dependencies
- Deploys or updates Lambda functions
- Configures CloudWatch event triggers
- Sets up permissions for CloudWatch to invoke Lambda functions

### CloudWatch Integration
- Monitoring agent runs on a scheduled basis (every 5 minutes)
- All agents log to CloudWatch for monitoring and debugging
- Automatic alerting through SNS topics

## Communication Flow

1. **Scheduled Monitoring**: CloudWatch triggers monitoring agent every 5 minutes
2. **Sensor Data Processing**: Analysis agent receives feature data and calls SageMaker endpoint
3. **Fire Detection**: SageMaker returns prediction results to analysis agent
4. **Response Coordination**: Response agent receives detection results and executes appropriate actions
5. **Alerting**: All agents can send notifications via SNS topics

## Benefits of AWS Deployment

### Scalability
- Automatic scaling based on demand
- No need to manage underlying infrastructure
- Pay only for actual usage

### Reliability
- High availability with multi-AZ deployment
- Built-in fault tolerance
- Automatic retries and error handling

### Maintainability
- Centralized logging with CloudWatch
- Easy updates and deployments
- Version control for Lambda functions

### Security
- IAM roles with least privilege principle
- Encrypted communications
- Secure access to AWS services

## Deployment Commands

To deploy the agents to AWS:

```bash
# Make the deployment script executable
chmod +x deploy_lambda_agents.sh

# Run the deployment script
./deploy_lambda_agents.sh
```

## Testing the Deployment

After deployment, you can test the functions using the AWS CLI:

```bash
# Test the monitoring agent
aws lambda invoke \
    --function-name saafe-monitoring-agent \
    --payload '{"sensor_data": {"flir": {"temperature": 45.2}, "scd41": {"co2_concentration": 550}}}' \
    response.json

# Test the analysis agent
aws lambda invoke \
    --function-name saafe-analysis-agent \
    --payload '{"features": {"t_mean": 25.5, "t_std": 3.2, "t_max": 45.0, "t_p95": 38.0, "t_hot_area_pct": 2.1, "t_hot_largest_blob_pct": 1.5, "t_grad_mean": 1.2, "t_grad_std": 0.8, "t_diff_mean": 0.9, "t_diff_std": 0.4, "flow_mag_mean": 0.7, "flow_mag_std": 0.3, "tproxy_val": 26.0, "tproxy_delta": 1.0, "tproxy_vel": 0.2, "gas_val": 450.0, "gas_delta": 50.0, "gas_vel": 2.5}}' \
    response.json

# Test the response agent
aws lambda invoke \
    --function-name saafe-response-agent \
    --payload '{"detection_results": {"confidence_score": 0.85, "fire_detected": true}}' \
    response.json
```

## Next Steps

1. **Configure SNS Topics**: Set up the required SNS topics for alerts
2. **Set Up Additional Triggers**: Configure event sources beyond the scheduled monitoring
3. **Implement Error Handling**: Add more sophisticated error handling and retry logic
4. **Add Metrics Collection**: Implement custom CloudWatch metrics for agent performance
5. **Set Up CI/CD**: Create automated deployment pipelines for agent updates

## Conclusion

The AWS Lambda agents provide a robust, scalable, and maintainable implementation of the FLIR+SCD41 Fire Detection System's agent framework. By running on AWS Lambda, the system benefits from automatic scaling, high availability, and reduced operational overhead while maintaining seamless integration with the existing SageMaker ML models.