#!/usr/bin/env python3
"""
Simple SageMaker Training Job Example
This script demonstrates how to create a SageMaker training job for the FLIR+SCD41 fire detection system.
"""

import boto3
import json
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET = 'fire-detection-training-691595239825'
S3_PREFIX = 'flir_scd41_training'

def create_sagemaker_training_job():
    """Create a SageMaker training job for fire detection."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Generate unique job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"flir-scd41-fire-detection-{timestamp}"
    
    # Define training job parameters
    training_params = {
        'TrainingJobName': job_name,
        'RoleArn': 'arn:aws:iam::691595239825:role/SageMakerExecutionRole',  # Replace with your actual role ARN
        'AlgorithmSpecification': {
            'TrainingInputMode': 'File',
            'TrainingImage': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3'
        },
        'InputDataConfig': [
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://{S3_BUCKET}/{S3_PREFIX}/data/',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                }
            }
        ],
        'OutputDataConfig': {
            'S3OutputPath': f's3://{S3_BUCKET}/{S3_PREFIX}/models/'
        },
        'ResourceConfig': {
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1,
            'VolumeSizeInGB': 30
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 3600  # 1 hour
        }
    }
    
    try:
        # Create training job
        response = sagemaker.create_training_job(**training_params)
        print(f"Training job created successfully: {job_name}")
        print(f"Training job ARN: {response['TrainingJobArn']}")
        return job_name
    except Exception as e:
        print(f"Error creating training job: {e}")
        return None

def monitor_training_job(job_name):
    """Monitor the status of a training job."""
    
    if not job_name:
        print("No job name provided")
        return
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    try:
        # Get training job status
        response = sagemaker.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        print(f"Training job status: {status}")
        
        if 'SecondaryStatus' in response:
            print(f"Secondary status: {response['SecondaryStatus']}")
            
        if 'FailureReason' in response:
            print(f"Failure reason: {response['FailureReason']}")
            
        return status
    except Exception as e:
        print(f"Error monitoring training job: {e}")
        return None

def list_recent_training_jobs():
    """List recent training jobs."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    try:
        # List training jobs
        response = sagemaker.list_training_jobs(
            MaxResults=10,
            SortBy='CreationTime',
            SortOrder='Descending'
        )
        
        print("Recent training jobs:")
        for job in response['TrainingJobSummaries']:
            print(f"  - {job['TrainingJobName']}: {job['TrainingJobStatus']}")
            
    except Exception as e:
        print(f"Error listing training jobs: {e}")

if __name__ == "__main__":
    print("FLIR+SCD41 Fire Detection - SageMaker Training Example")
    print("=" * 50)
    
    # List recent training jobs
    list_recent_training_jobs()
    
    # Create a new training job
    print("\nCreating new training job...")
    job_name = create_sagemaker_training_job()
    
    # Monitor the job if created successfully
    if job_name:
        print(f"\nMonitoring training job: {job_name}")
        status = monitor_training_job(job_name)
        
    print("\nFor detailed monitoring, use the AWS SageMaker console or AWS CLI:")
    print(f"aws sagemaker describe-training-job --training-job-name {job_name} --region {AWS_REGION}")