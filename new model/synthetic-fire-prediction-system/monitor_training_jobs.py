#!/usr/bin/env python3
"""
Training Job Monitoring Script
This script monitors the progress of SageMaker training jobs for the FLIR+SCD41 fire detection system.
"""

import boto3
import time
import json
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'

def monitor_training_jobs(job_names):
    """Monitor the status of multiple training jobs."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Track job statuses
    job_statuses = {job_name: None for job_name in job_names}
    completed_jobs = set()
    
    print("Monitoring training jobs...")
    print("=" * 50)
    
    while len(completed_jobs) < len(job_names):
        for job_name in job_names:
            if job_name in completed_jobs:
                continue
                
            try:
                # Get training job status
                response = sagemaker.describe_training_job(TrainingJobName=job_name)
                status = response['TrainingJobStatus']
                
                # Update status if changed
                if job_statuses[job_name] != status:
                    job_statuses[job_name] = status
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] {job_name}: {status}")
                    
                    # Check for failure
                    if status in ['Failed', 'Stopped']:
                        if 'FailureReason' in response:
                            print(f"  Failure reason: {response['FailureReason']}")
                        completed_jobs.add(job_name)
                        
                    # Check for completion
                    elif status == 'Completed':
                        completed_jobs.add(job_name)
                        # Print training metrics if available
                        if 'FinalMetricDataList' in response:
                            print("  Final metrics:")
                            for metric in response['FinalMetricDataList']:
                                print(f"    {metric['MetricName']}: {metric['Value']}")
                        
            except Exception as e:
                print(f"Error monitoring {job_name}: {e}")
                
        # Wait before next check
        if len(completed_jobs) < len(job_names):
            time.sleep(30)  # Check every 30 seconds
    
    print("\nAll training jobs completed!")
    print("=" * 50)
    
    # Print final summary
    for job_name, status in job_statuses.items():
        print(f"{job_name}: {status}")

def get_training_job_details(job_name):
    """Get detailed information about a training job."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    try:
        # Get training job details
        response = sagemaker.describe_training_job(TrainingJobName=job_name)
        
        print(f"Training Job: {job_name}")
        print("-" * 30)
        print(f"Status: {response['TrainingJobStatus']}")
        print(f"Creation Time: {response['CreationTime']}")
        
        if 'TrainingStartTime' in response:
            print(f"Start Time: {response['TrainingStartTime']}")
            
        if 'TrainingEndTime' in response:
            print(f"End Time: {response['TrainingEndTime']}")
            
        print(f"Instance Type: {response['ResourceConfig']['InstanceType']}")
        print(f"Instance Count: {response['ResourceConfig']['InstanceCount']}")
        
        if 'FinalMetricDataList' in response:
            print("Final Metrics:")
            for metric in response['FinalMetricDataList']:
                print(f"  {metric['MetricName']}: {metric['Value']}")
                
        if 'FailureReason' in response:
            print(f"Failure Reason: {response['FailureReason']}")
            
        print(f"Model Artifacts: {response['ModelArtifacts']['S3ModelArtifacts']}")
        
    except Exception as e:
        print(f"Error getting details for {job_name}: {e}")

def list_model_artifacts(job_names):
    """List model artifacts produced by training jobs."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    s3 = boto3.client('s3', region_name=AWS_REGION)
    
    print("Model Artifacts:")
    print("-" * 30)
    
    for job_name in job_names:
        try:
            # Get training job details
            response = sagemaker.describe_training_job(TrainingJobName=job_name)
            
            if 'ModelArtifacts' in response:
                model_artifacts = response['ModelArtifacts']['S3ModelArtifacts']
                print(f"{job_name}:")
                print(f"  {model_artifacts}")
                
                # Parse S3 URI to list contents
                if model_artifacts.startswith('s3://'):
                    # Extract bucket and key
                    s3_uri = model_artifacts[5:]  # Remove 's3://'
                    bucket, key = s3_uri.split('/', 1)
                    
                    # List objects with this prefix
                    try:
                        list_response = s3.list_objects_v2(
                            Bucket=bucket,
                            Prefix=key
                        )
                        
                        if 'Contents' in list_response:
                            for obj in list_response['Contents']:
                                print(f"    - {obj['Key']} ({obj['Size']} bytes)")
                    except Exception as e:
                        print(f"    Error listing artifacts: {e}")
                        
        except Exception as e:
            print(f"Error getting artifacts for {job_name}: {e}")

if __name__ == "__main__":
    # Training job names (replace with your actual job names)
    training_jobs = [
        "flir-scd41-improved-training-20250828-152706",
        "flir-scd41-xgboost-training-20250828-152707"
    ]
    
    print("FLIR+SCD41 Fire Detection - Training Job Monitoring")
    print("=" * 55)
    
    # Option 1: Monitor training jobs in real-time
    print("1. Monitoring training jobs...")
    monitor_training_jobs(training_jobs)
    
    # Option 2: Get detailed information about each job
    print("\n2. Detailed job information:")
    for job_name in training_jobs:
        print()
        get_training_job_details(job_name)
    
    # Option 3: List model artifacts
    print("\n3. Model artifacts:")
    list_model_artifacts(training_jobs)