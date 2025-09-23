#!/usr/bin/env python3
"""
Comprehensive Training Monitor and Deployment Script
This script monitors training jobs and automatically deploys models when training completes.
"""

import boto3
import time
import json
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET = 'fire-detection-training-691595239825'
S3_PREFIX = 'flir_scd41_training'

class TrainingMonitor:
    def __init__(self):
        self.sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
        self.s3 = boto3.client('s3', region_name=AWS_REGION)
    
    def monitor_training_job(self, job_name, check_interval=60):
        """Monitor a training job until completion."""
        print(f"Monitoring training job: {job_name}")
        print("=" * 50)
        
        last_status = None
        last_message = None
        
        while True:
            try:
                # Get training job status
                response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
                status = response['TrainingJobStatus']
                secondary_status = response.get('SecondaryStatus', 'N/A')
                
                # Print status updates
                if status != last_status or secondary_status != last_message:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] Status: {status} ({secondary_status})")
                    last_status = status
                    last_message = secondary_status
                
                # Check for completion
                if status == 'Completed':
                    print(f"\nTraining job completed successfully!")
                    return True, response
                    
                # Check for failure
                elif status in ['Failed', 'Stopped']:
                    failure_reason = response.get('FailureReason', 'No reason provided')
                    print(f"\nTraining job failed: {failure_reason}")
                    return False, response
                
                # Wait before next check
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"Error monitoring training job: {e}")
                return False, None
    
    def deploy_model(self, job_name, model_artifacts):
        """Deploy a trained model to SageMaker hosting services."""
        print(f"\nDeploying model from training job: {job_name}")
        print("=" * 50)
        
        try:
            # Generate unique names
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name = f"flir-scd41-model-{timestamp}"
            endpoint_config_name = f"{model_name}-config"
            endpoint_name = f"{model_name}-endpoint"
            
            # Create model
            model_params = {
                'ModelName': model_name,
                'PrimaryContainer': {
                    'Image': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3',
                    'ModelDataUrl': model_artifacts,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'serve',  # Updated to use our fixed inference script
                        'SAGEMAKER_SUBMIT_DIRECTORY': f's3://{S3_BUCKET}/{S3_PREFIX}/code/code_fixed_20250828-153948.tar.gz'  # Updated to use our fixed code package
                    }
                },
                'ExecutionRoleArn': 'arn:aws:iam::691595239825:role/SageMakerExecutionRole'
            }
            
            response = self.sagemaker.create_model(**model_params)
            print(f"Model created: {model_name}")
            
            # Create endpoint configuration
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
            
            response = self.sagemaker.create_endpoint_config(**endpoint_config_params)
            print(f"Endpoint configuration created: {endpoint_config_name}")
            
            # Create endpoint
            endpoint_params = {
                'EndpointName': endpoint_name,
                'EndpointConfigName': endpoint_config_name
            }
            
            response = self.sagemaker.create_endpoint(**endpoint_params)
            print(f"Endpoint creation initiated: {endpoint_name}")
            
            return endpoint_name
            
        except Exception as e:
            print(f"Error deploying model: {e}")
            return None
    
    def wait_for_endpoint(self, endpoint_name, check_interval=30):
        """Wait for endpoint to be in service."""
        print(f"\nWaiting for endpoint {endpoint_name} to be in service...")
        print("=" * 50)
        
        while True:
            try:
                response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Endpoint status: {status}")
                
                if status == 'InService':
                    print("Endpoint is ready for use!")
                    return True
                elif status in ['Failed', 'OutOfService']:
                    failure_reason = response.get('FailureReason', 'No reason provided')
                    print(f"Endpoint failed: {failure_reason}")
                    return False
                    
                # Wait before checking again
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"Error checking endpoint status: {e}")
                return False
    
    def test_endpoint(self, endpoint_name):
        """Test the deployed endpoint with sample data."""
        print(f"\nTesting endpoint {endpoint_name} with sample data...")
        print("=" * 50)
        
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

def main():
    # Training job names (updated to use our new fixed jobs)
    training_job_names = [
        "flir-scd41-rf-training-fixed-20250828-153947",  # Random Forest (fixed)
        "flir-scd41-xgb-training-fixed-20250828-153947"   # XGBoost (fixed)
    ]
    
    print("FLIR+SCD41 Fire Detection - Training Monitor and Deployment")
    print("=" * 60)
    
    # Initialize monitor
    monitor = TrainingMonitor()
    
    # Monitor training jobs
    completed_jobs = []
    for job_name in training_job_names:
        print(f"\nMonitoring training job: {job_name}")
        success, training_response = monitor.monitor_training_job(job_name, check_interval=30)
        
        if success:
            completed_jobs.append((job_name, training_response))
            print(f"Training job {job_name} completed successfully!")
        else:
            print(f"Training job {job_name} failed!")
    
    # Deploy models from completed jobs
    if completed_jobs:
        print(f"\nDeploying {len(completed_jobs)} completed models...")
        for job_name, training_response in completed_jobs:
            # Get model artifacts
            model_artifacts = training_response['ModelArtifacts']['S3ModelArtifacts']
            print(f"\nModel artifacts for {job_name}: {model_artifacts}")
            
            # Deploy model
            endpoint_name = monitor.deploy_model(job_name, model_artifacts)
            
            if endpoint_name:
                # Wait for endpoint to be ready
                if monitor.wait_for_endpoint(endpoint_name):
                    # Test endpoint
                    monitor.test_endpoint(endpoint_name)
                    
                    print(f"\nDeployment completed successfully for {job_name}!")
                    print(f"Endpoint name: {endpoint_name}")
                    print(f"You can now use this endpoint for real-time fire detection predictions.")
                else:
                    print(f"\nEndpoint deployment failed for {job_name}.")
            else:
                print(f"\nModel deployment failed for {job_name}.")
    else:
        print("\nNo training jobs completed successfully. Skipping deployment.")

if __name__ == "__main__":
    main()