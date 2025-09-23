#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Model Evaluation using SageMaker
This script evaluates the trained model's performance by creating a batch transform job.
"""

import boto3
import pandas as pd
import json
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET = 'fire-detection-training-691595239825'
S3_PREFIX = 'flir_scd41_training'

def create_batch_transform_job():
    """Create a batch transform job to evaluate the model."""
    print("Creating batch transform job for model evaluation...")
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Generate unique job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    transform_job_name = f"flir-scd41-evaluation-{timestamp}"
    
    # Model name from our training job
    model_artifacts_uri = "s3://fire-detection-training-691595239825/flir_scd41_training/models/flir-scd41-xgboost-simple-20250828-154649/output/model.tar.gz"
    
    try:
        # Create model
        model_name = f"flir-scd41-xgboost-model-{timestamp}"
        model_params = {
            'ModelName': model_name,
            'PrimaryContainer': {
                'Image': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-xgboost:1.5-1-cpu-py3',
                'ModelDataUrl': model_artifacts_uri
            },
            'ExecutionRoleArn': 'arn:aws:iam::691595239825:role/SageMakerExecutionRole'
        }
        
        sagemaker.create_model(**model_params)
        print(f"Model created: {model_name}")
        
        # Create transform job
        transform_params = {
            'TransformJobName': transform_job_name,
            'ModelName': model_name,
            'TransformInput': {
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://{S3_BUCKET}/{S3_PREFIX}/data/flir_scd41_data.csv'
                    }
                },
                'ContentType': 'text/csv',
                'SplitType': 'Line'
            },
            'TransformOutput': {
                'S3OutputPath': f's3://{S3_BUCKET}/{S3_PREFIX}/evaluation/results-{timestamp}/'
            },
            'TransformResources': {
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1
            }
        }
        
        response = sagemaker.create_transform_job(**transform_params)
        print(f"Batch transform job created: {transform_job_name}")
        print(f"Transform job ARN: {response['TransformJobArn']}")
        
        return transform_job_name
        
    except Exception as e:
        print(f"Error creating batch transform job: {e}")
        return None

def monitor_transform_job(transform_job_name, check_interval=30):
    """Monitor the batch transform job until completion."""
    print(f"Monitoring transform job: {transform_job_name}")
    print("=" * 50)
    
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    last_status = None
    
    while True:
        try:
            # Get transform job status
            response = sagemaker.describe_transform_job(TransformJobName=transform_job_name)
            status = response['TransformJobStatus']
            
            # Print status updates
            if status != last_status:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Transform job status: {status}")
                last_status = status
            
            # Check if transform job is complete
            if status == 'Completed':
                print(f"\nTransform job completed successfully!")
                return True
            elif status in ['Failed', 'Stopped']:
                failure_reason = response.get('FailureReason', 'No reason provided')
                print(f"\nTransform job failed: {failure_reason}")
                return False
            
            # Wait before next check
            import time
            time.sleep(check_interval)
            
        except Exception as e:
            print(f"Error monitoring transform job: {e}")
            return False

def get_transform_job_metrics(transform_job_name):
    """Get metrics from the completed transform job."""
    print(f"Getting metrics for transform job: {transform_job_name}")
    
    try:
        sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
        response = sagemaker.describe_transform_job(TransformJobName=transform_job_name)
        
        # Print basic information
        print("\n" + "=" * 50)
        print("TRANSFORM JOB METRICS")
        print("=" * 50)
        print(f"Transform job name: {response['TransformJobName']}")
        print(f"Creation time: {response['CreationTime']}")
        print(f"Completion time: {response.get('TransformEndTime', 'N/A')}")
        print(f"Input records: {response.get('TransformInput', {}).get('RecordWrapperType', 'N/A')}")
        print(f"Output records: {response.get('TransformOutput', {}).get('Accept', 'N/A')}")
        
        # Check for any failure reasons
        if 'FailureReason' in response:
            print(f"Failure reason: {response['FailureReason']}")
        
        print("=" * 50)
        
        return response
        
    except Exception as e:
        print(f"Error getting transform job metrics: {e}")
        return None

def main():
    """Main function to evaluate the trained model using batch transform."""
    print("FLIR+SCD41 Fire Detection - Model Evaluation using SageMaker")
    print("=" * 60)
    
    # Create batch transform job
    transform_job_name = create_batch_transform_job()
    
    if transform_job_name:
        # Monitor transform job
        success = monitor_transform_job(transform_job_name)
        
        if success:
            # Get metrics
            metrics = get_transform_job_metrics(transform_job_name)
            if metrics:
                print("\n✅ Model evaluation completed successfully!")
                print("\nThe batch transform job has processed the data and generated predictions.")
                print("You can now review the results in the S3 output location.")
            else:
                print("\n⚠️ Transform job completed but metrics could not be retrieved.")
        else:
            print("\n❌ Model evaluation failed.")
    else:
        print("\n❌ Failed to create batch transform job.")

if __name__ == "__main__":
    main()