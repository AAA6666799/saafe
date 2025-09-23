#!/usr/bin/env python3
"""
SageMaker Model Deployment Example
This script demonstrates how to deploy a trained model using SageMaker hosting services.
"""

import boto3
import json
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET = 'fire-detection-training-691595239825'
S3_PREFIX = 'flir_scd41_training'

def deploy_model(model_name, model_data_url, role_arn):
    """Deploy a trained model to SageMaker hosting services."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Generate unique model name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    deployment_name = f"flir-scd41-model-{timestamp}"
    
    try:
        # Create model
        model_params = {
            'ModelName': deployment_name,
            'PrimaryContainer': {
                'Image': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3',
                'ModelDataUrl': model_data_url,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': f's3://{S3_BUCKET}/{S3_PREFIX}/code/sagemaker_submit_directory.tar.gz'
                }
            },
            'ExecutionRoleArn': role_arn
        }
        
        response = sagemaker.create_model(**model_params)
        print(f"Model created successfully: {deployment_name}")
        
        # Create endpoint configuration
        endpoint_config_name = f"{deployment_name}-config"
        endpoint_config_params = {
            'EndpointConfigName': endpoint_config_name,
            'ProductionVariants': [
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': deployment_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large'
                }
            ]
        }
        
        response = sagemaker.create_endpoint_config(**endpoint_config_params)
        print(f"Endpoint configuration created: {endpoint_config_name}")
        
        # Create endpoint
        endpoint_name = f"{deployment_name}-endpoint"
        endpoint_params = {
            'EndpointName': endpoint_name,
            'EndpointConfigName': endpoint_config_name
        }
        
        response = sagemaker.create_endpoint(**endpoint_params)
        print(f"Endpoint creation initiated: {endpoint_name}")
        print("Note: Endpoint deployment may take several minutes to complete")
        
        return endpoint_name
        
    except Exception as e:
        print(f"Error deploying model: {e}")
        return None

def invoke_endpoint(endpoint_name, payload):
    """Invoke a deployed endpoint for inference."""
    
    # Initialize SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
    
    try:
        # Invoke endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        result = json.loads(response['Body'].read().decode())
        print(f"Inference result: {result}")
        return result
        
    except Exception as e:
        print(f"Error invoking endpoint: {e}")
        return None

def delete_endpoint(endpoint_name):
    """Delete a deployed endpoint and its configuration."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    try:
        # Delete endpoint
        sagemaker.delete_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint deleted: {endpoint_name}")
        
        # Delete endpoint configuration
        endpoint_config_name = f"{endpoint_name}-config"
        sagemaker.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"Endpoint configuration deleted: {endpoint_config_name}")
        
        # Delete model
        model_name = endpoint_name.replace('-endpoint', '')
        sagemaker.delete_model(ModelName=model_name)
        print(f"Model deleted: {model_name}")
        
    except Exception as e:
        print(f"Error deleting endpoint resources: {e}")

def list_endpoints():
    """List all SageMaker endpoints."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    try:
        # List endpoints
        response = sagemaker.list_endpoints(MaxResults=10)
        
        print("SageMaker endpoints:")
        for endpoint in response['Endpoints']:
            print(f"  - {endpoint['EndpointName']}: {endpoint['EndpointStatus']}")
            
    except Exception as e:
        print(f"Error listing endpoints: {e}")

if __name__ == "__main__":
    print("FLIR+SCD41 Fire Detection - SageMaker Model Deployment Example")
    print("=" * 60)
    
    # List existing endpoints
    list_endpoints()
    
    # Example usage (commented out to prevent accidental deployment)
    """
    # Deploy a model (replace with actual values)
    model_data_url = f"s3://{S3_BUCKET}/{S3_PREFIX}/models/model.tar.gz"
    role_arn = "arn:aws:iam::691595239825:role/SageMakerExecutionRole"
    
    endpoint_name = deploy_model("example-model", model_data_url, role_arn)
    
    if endpoint_name:
        # Wait for endpoint to be in service (this can take 5-15 minutes)
        print(f"Waiting for endpoint {endpoint_name} to be in service...")
        # In practice, you would use a waiter or poll the endpoint status
        
        # Example inference payload (replace with actual data)
        payload = {
            "features": {
                "t_mean": 30.5,
                "t_std": 2.1,
                "t_max": 55.2,
                # ... other features
                "gas_val": 450.0,
                "gas_delta": 25.0,
                "gas_vel": 1.2
            }
        }
        
        # Invoke endpoint
        result = invoke_endpoint(endpoint_name, payload)
    """
    
    print("\nTo deploy a model, uncomment the deployment code and provide:")
    print("1. Model data URL (S3 path to trained model artifacts)")
    print("2. IAM role ARN with SageMaker execution permissions")
    print("3. Inference payload matching your model's input format")
    
    print("\nFor detailed monitoring, use the AWS SageMaker console or AWS CLI:")
    print("aws sagemaker list-endpoints --region us-east-1")