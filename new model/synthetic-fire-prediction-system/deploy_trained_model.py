#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Model Deployment
This script deploys the trained model to SageMaker hosting services.
"""

import boto3
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET = 'fire-detection-training-691595239825'
S3_PREFIX = 'flir_scd41_training'

def deploy_model(model_artifacts_uri):
    """Deploy the trained model to SageMaker hosting services."""
    print("Deploying trained model...")
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Generate unique names
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"flir-scd41-xgboost-model-{timestamp}"
    endpoint_config_name = f"{model_name}-config"
    endpoint_name = f"{model_name}-endpoint"
    
    try:
        # Create model
        print(f"Creating model: {model_name}")
        model_params = {
            'ModelName': model_name,
            'PrimaryContainer': {
                'Image': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-xgboost:1.5-1-cpu-py3',
                'ModelDataUrl': model_artifacts_uri,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': f's3://{S3_BUCKET}/{S3_PREFIX}/code/sagemaker_submit_directory.tar.gz'
                }
            },
            'ExecutionRoleArn': 'arn:aws:iam::691595239825:role/SageMakerExecutionRole'
        }
        
        response = sagemaker.create_model(**model_params)
        print(f"Model created successfully: {model_name}")
        
        # Create endpoint configuration
        print(f"Creating endpoint configuration: {endpoint_config_name}")
        endpoint_config_params = {
            'EndpointConfigName': endpoint_config_name,
            'ProductionVariants': [
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large'
                }
            ]
        }
        
        response = sagemaker.create_endpoint_config(**endpoint_config_params)
        print(f"Endpoint configuration created successfully: {endpoint_config_name}")
        
        # Create endpoint
        print(f"Creating endpoint: {endpoint_name}")
        endpoint_params = {
            'EndpointName': endpoint_name,
            'EndpointConfigName': endpoint_config_name
        }
        
        response = sagemaker.create_endpoint(**endpoint_params)
        print(f"Endpoint creation initiated successfully: {endpoint_name}")
        
        print("\n" + "=" * 50)
        print("MODEL DEPLOYMENT INITIATED SUCCESSFULLY")
        print("=" * 50)
        print(f"Model name: {model_name}")
        print(f"Endpoint configuration: {endpoint_config_name}")
        print(f"Endpoint name: {endpoint_name}")
        print("\nNote: Endpoint deployment may take 5-15 minutes to complete")
        print("\nTo check endpoint status, use:")
        print(f"aws sagemaker describe-endpoint --endpoint-name {endpoint_name} --region {AWS_REGION}")
        
        return endpoint_name
        
    except Exception as e:
        print(f"Error deploying model: {e}")
        return None

def test_endpoint(endpoint_name):
    """Test the deployed endpoint with sample data."""
    print(f"\nTesting endpoint {endpoint_name} with sample data...")
    
    # Initialize SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
    
    # Sample test data (potential fire conditions)
    # For XGBoost, we need to provide data in CSV format
    test_data = "1,45.2,8.7,78.5,72.1,25.3,18.7,3.2,1.8,2.9,1.5,4.2,2.1,52.0,15.0,3.2,850.0,120.0,8.5"
    
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=test_data
        )
        
        result = response['Body'].read().decode()
        print(f"Prediction result: {result}")
        
        return result
        
    except Exception as e:
        print(f"Error testing endpoint: {e}")
        return None

def main():
    """Main function to deploy the trained model."""
    print("FLIR+SCD41 Fire Detection - Model Deployment")
    print("=" * 45)
    
    # Model artifacts URI from our successful training job
    model_artifacts_uri = "s3://fire-detection-training-691595239825/flir_scd41_training/models/flir-scd41-xgboost-simple-20250828-154649/output/model.tar.gz"
    
    # Deploy model
    endpoint_name = deploy_model(model_artifacts_uri)
    
    if endpoint_name:
        print(f"\nModel deployment initiated for endpoint: {endpoint_name}")
        print("Please wait 5-15 minutes for the endpoint to be ready for use.")
        
        # Optionally test the endpoint (will only work once it's InService)
        # test_endpoint(endpoint_name)
    else:
        print("\nModel deployment failed.")

if __name__ == "__main__":
    main()