#!/usr/bin/env python3
"""
Model Deployment Script for SageMaker
This script deploys the trained models to SageMaker hosting services.
"""

import boto3
import time
import json
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET = 'fire-detection-training-691595239825'
S3_PREFIX = 'flir_scd41_training'

def deploy_model(model_name, model_data_url, algorithm_type='sklearn'):
    """Deploy a trained model to SageMaker hosting services."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Generate unique names
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    deployment_name = f"flir-scd41-{model_name}-{timestamp}"
    
    try:
        # Create model
        if algorithm_type == 'xgboost':
            image_uri = f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-xgboost:1.5-1-cpu-py3'
        else:
            image_uri = f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3'
        
        model_params = {
            'ModelName': deployment_name,
            'PrimaryContainer': {
                'Image': image_uri,
                'ModelDataUrl': model_data_url,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'flir_scd41_inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': f's3://{S3_BUCKET}/{S3_PREFIX}/code/sagemaker_submit_directory.tar.gz'
                }
            },
            'ExecutionRoleArn': 'arn:aws:iam::691595239825:role/SageMakerExecutionRole'
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
        print("Note: Endpoint deployment may take 5-15 minutes to complete")
        
        return endpoint_name, deployment_name, endpoint_config_name
        
    except Exception as e:
        print(f"Error deploying model: {e}")
        return None, None, None

def create_inference_script():
    """Create an inference script for the deployed models."""
    
    inference_script = '''
import json
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

def model_fn(model_dir):
    """Load the model and scaler."""
    print(f"Loading model from {model_dir}")
    
    # Load model
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    
    # Load scaler if it exists
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    return model, scaler

def input_fn(request_body, request_content_type):
    """Parse input data."""
    print(f"Processing input: {request_body}")
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Handle different input formats
        if 'features' in input_data:
            # Single sample
            features = input_data['features']
            if isinstance(features, dict):
                # Convert dict to list in correct order
                feature_order = [
                    't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
                    't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
                    't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
                    'tproxy_val', 'tproxy_delta', 'tproxy_vel',
                    'gas_val', 'gas_delta', 'gas_vel'
                ]
                feature_values = [features[f] for f in feature_order]
                data = np.array([feature_values])
            else:
                # Assume it's already a list
                data = np.array([features])
        elif 'samples' in input_data:
            # Multiple samples
            samples = input_data['samples']
            data_list = []
            for sample in samples:
                if isinstance(sample, dict):
                    feature_order = [
                        't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
                        't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
                        't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
                        'tproxy_val', 'tproxy_delta', 'tproxy_vel',
                        'gas_val', 'gas_delta', 'gas_vel'
                    ]
                    feature_values = [sample[f] for f in feature_order]
                    data_list.append(feature_values)
                else:
                    data_list.append(sample)
            data = np.array(data_list)
        else:
            # Direct array input
            data = np.array(input_data)
        
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_and_scaler):
    """Make predictions."""
    model, scaler = model_and_scaler
    
    print(f"Making predictions for data shape: {input_data.shape}")
    
    # Scale data if scaler is available
    if scaler is not None:
        input_data = scaler.transform(input_data)
    
    # Make predictions
    predictions = model.predict(input_data)
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_data)
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
    else:
        return {
            'predictions': predictions.tolist()
        }

def output_fn(prediction, content_type):
    """Format the output."""
    print(f"Formatting output: {prediction}")
    
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
'''
    
    # Save the inference script
    script_path = "/tmp/flir_scd41_inference.py"
    with open(script_path, 'w') as f:
        f.write(inference_script)
    
    print(f"Inference script saved to {script_path}")
    
    # Upload to S3
    s3 = boto3.client('s3', region_name=AWS_REGION)
    s3_key = f"{S3_PREFIX}/code/flir_scd41_inference.py"
    
    try:
        s3.upload_file(script_path, S3_BUCKET, s3_key)
        print(f"Inference script uploaded to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"Error uploading inference script: {e}")
    
    return script_path

def wait_for_endpoint(endpoint_name):
    """Wait for endpoint to be in service."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    print(f"Waiting for endpoint {endpoint_name} to be in service...")
    
    while True:
        try:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            print(f"Endpoint status: {status}")
            
            if status == 'InService':
                print("Endpoint is ready for use!")
                break
            elif status in ['Failed', 'OutOfService']:
                print(f"Endpoint failed with status: {status}")
                if 'FailureReason' in response:
                    print(f"Failure reason: {response['FailureReason']}")
                break
                
            # Wait before checking again
            time.sleep(30)
            
        except Exception as e:
            print(f"Error checking endpoint status: {e}")
            break
        except KeyboardInterrupt:
            print("Waiting interrupted by user")
            break

def test_endpoint(endpoint_name):
    """Test the deployed endpoint with sample data."""
    
    # Initialize SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
    
    # Sample test data (normal conditions)
    test_data_normal = {
        "features": {
            "t_mean": 22.5,
            "t_std": 1.2,
            "t_max": 25.1,
            "t_p95": 24.8,
            "t_hot_area_pct": 0.5,
            "t_hot_largest_blob_pct": 0.3,
            "t_grad_mean": 0.1,
            "t_grad_std": 0.05,
            "t_diff_mean": 0.2,
            "t_diff_std": 0.1,
            "flow_mag_mean": 0.3,
            "flow_mag_std": 0.1,
            "tproxy_val": 23.0,
            "tproxy_delta": 0.5,
            "tproxy_vel": 0.1,
            "gas_val": 410.0,
            "gas_delta": 5.0,
            "gas_vel": 1.0
        }
    }
    
    # Sample test data (potential fire conditions)
    test_data_fire = {
        "features": {
            "t_mean": 45.2,
            "t_std": 8.7,
            "t_max": 78.5,
            "t_p95": 72.1,
            "t_hot_area_pct": 25.3,
            "t_hot_largest_blob_pct": 18.7,
            "t_grad_mean": 3.2,
            "t_grad_std": 1.8,
            "t_diff_mean": 2.9,
            "t_diff_std": 1.5,
            "flow_mag_mean": 4.2,
            "flow_mag_std": 2.1,
            "tproxy_val": 52.0,
            "tproxy_delta": 15.0,
            "tproxy_vel": 3.2,
            "gas_val": 850.0,
            "gas_delta": 120.0,
            "gas_vel": 8.5
        }
    }
    
    print(f"Testing endpoint {endpoint_name} with sample data...")
    
    for i, (data, label) in enumerate([(test_data_normal, "Normal"), (test_data_fire, "Fire")]):
        try:
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(data)
            )
            
            result = json.loads(response['Body'].read().decode())
            print(f"{label} conditions test {i+1}:")
            print(f"  Predictions: {result.get('predictions', 'N/A')}")
            if 'probabilities' in result:
                print(f"  Probabilities: {result['probabilities']}")
                
        except Exception as e:
            print(f"Error testing {label} conditions: {e}")

def list_endpoints():
    """List all SageMaker endpoints."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    try:
        # List endpoints
        response = sagemaker.list_endpoints(MaxResults=20)
        
        print("SageMaker endpoints:")
        for endpoint in response['Endpoints']:
            print(f"  - {endpoint['EndpointName']}: {endpoint['EndpointStatus']}")
            
    except Exception as e:
        print(f"Error listing endpoints: {e}")

if __name__ == "__main__":
    print("FLIR+SCD41 Fire Detection - Model Deployment")
    print("=" * 45)
    
    # List existing endpoints
    list_endpoints()
    
    # Create inference script
    print("\nCreating inference script...")
    create_inference_script()
    
    # Deploy models
    print("\nDeploying models to SageMaker...")
    
    # Deploy XGBoost model (using existing trained model)
    xgb_model_url = f"s3://{S3_BUCKET}/{S3_PREFIX}/models/flir-scd41-xgboost-simple-20250828-154649/output/model.tar.gz"
    xgb_endpoint, xgb_model, xgb_config = deploy_model(
        "xgboost-model", 
        xgb_model_url, 
        "xgboost"
    )
    
    # Deploy ensemble debug model (using existing trained model)
    ensemble_debug_model_url = f"s3://{S3_BUCKET}/{S3_PREFIX}/models/flir-scd41-ensemble-debug-20250901-110038/output/model.tar.gz"
    ensemble_debug_endpoint, ensemble_debug_model, ensemble_debug_config = deploy_model(
        "ensemble-debug-model", 
        ensemble_debug_model_url, 
        "sklearn"
    )
    
    # Deploy full ensemble model (using existing trained model)
    full_ensemble_model_url = f"s3://{S3_BUCKET}/{S3_PREFIX}/models/flir-scd41-full-ensemble-fixed2-20250901-103338/output/model.tar.gz"
    full_ensemble_endpoint, full_ensemble_model, full_ensemble_config = deploy_model(
        "full-ensemble-model", 
        full_ensemble_model_url, 
        "sklearn"
    )
    
    # Wait for endpoints to be ready and test them
    if xgb_endpoint:
        print(f"\nWaiting for XGBoost model endpoint: {xgb_endpoint}")
        wait_for_endpoint(xgb_endpoint)
        test_endpoint(xgb_endpoint)
        
    if ensemble_debug_endpoint:
        print(f"\nWaiting for ensemble debug model endpoint: {ensemble_debug_endpoint}")
        wait_for_endpoint(ensemble_debug_endpoint)
        test_endpoint(ensemble_debug_endpoint)
        
    if full_ensemble_endpoint:
        print(f"\nWaiting for full ensemble model endpoint: {full_ensemble_endpoint}")
        wait_for_endpoint(full_ensemble_endpoint)
        test_endpoint(full_ensemble_endpoint)
    
    print("\nDeployment process completed!")
    print("For detailed monitoring, use the AWS SageMaker console or AWS CLI:")
    print("aws sagemaker list-endpoints --region us-east-1")