#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Simple Training Approach
This script creates training jobs using SageMaker's built-in algorithms with proper data format.
"""

import boto3
import json
import pandas as pd
from datetime import datetime
import tempfile
import os

# AWS Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET = 'fire-detection-training-691595239825'
S3_PREFIX = 'flir_scd41_training'

def convert_json_to_csv():
    """Convert our JSON data to CSV format required by SageMaker built-in algorithms."""
    print("Converting JSON data to CSV format...")
    
    # Download JSON data from S3
    s3 = boto3.client('s3', region_name=AWS_REGION)
    
    # Get the 50,000 sample data file
    json_key = f"{S3_PREFIX}/data/demo_data_50000.json"
    local_json_path = "/tmp/demo_data_50000.json"
    
    try:
        s3.download_file(S3_BUCKET, json_key, local_json_path)
        print(f"Downloaded JSON data from s3://{S3_BUCKET}/{json_key}")
    except Exception as e:
        print(f"Error downloading JSON data: {e}")
        return None
    
    # Load JSON data
    with open(local_json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    samples = data["samples"]
    df = pd.DataFrame([{
        **sample["features"],
        "label": sample["label"]
    } for sample in samples])
    
    # For XGBoost, the label should be the first column
    columns = list(df.columns)
    label_col = columns.pop()  # Remove 'label' from the end
    columns.insert(0, label_col)  # Insert 'label' at the beginning
    df = df[columns]
    
    # Save as CSV
    csv_path = "/tmp/flir_scd41_data.csv"
    df.to_csv(csv_path, index=False, header=False)  # SageMaker XGBoost expects no header
    print(f"Converted data to CSV format: {csv_path}")
    
    # Upload CSV to S3
    csv_key = f"{S3_PREFIX}/data/flir_scd41_data.csv"
    s3.upload_file(csv_path, S3_BUCKET, csv_key)
    print(f"Uploaded CSV data to s3://{S3_BUCKET}/{csv_key}")
    
    return f"s3://{S3_BUCKET}/{csv_key}"

def create_xgboost_training_job():
    """Create a training job using SageMaker's built-in XGBoost algorithm."""
    print("Creating XGBoost training job...")
    
    # Convert data to CSV format
    csv_data_uri = convert_json_to_csv()
    if not csv_data_uri:
        print("Failed to convert data to CSV format")
        return None
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Generate unique job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"flir-scd41-xgboost-simple-{timestamp}"
    
    try:
        # Create training job using built-in XGBoost algorithm
        response = sagemaker.create_training_job(
            TrainingJobName=job_name,
            RoleArn="arn:aws:iam::691595239825:role/SageMakerExecutionRole",
            AlgorithmSpecification={
                'TrainingInputMode': 'File',
                'TrainingImage': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-xgboost:1.5-1-cpu-py3'
            },
            InputDataConfig=[
                {
                    'ChannelName': 'train',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': csv_data_uri,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv'
                }
            ],
            OutputDataConfig={
                'S3OutputPath': f's3://{S3_BUCKET}/{S3_PREFIX}/models/'
            },
            ResourceConfig={
                'InstanceType': 'ml.m5.2xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 50
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': 7200  # 2 hours
            },
            HyperParameters={
                'max_depth': '5',
                'eta': '0.2',
                'objective': 'binary:logistic',
                'num_round': '100',
                'subsample': '0.8',
                'eval_metric': 'auc'
            }
        )
        
        print(f"XGBoost training job created successfully: {job_name}")
        print(f"Training job ARN: {response['TrainingJobArn']}")
        return job_name
        
    except Exception as e:
        print(f"Error creating XGBoost training job: {e}")
        return None

def main():
    """Main function to start simple training approach."""
    print("FLIR+SCD41 Fire Detection - Simple Training Approach")
    print("=" * 55)
    
    # Create XGBoost training job
    job_name = create_xgboost_training_job()
    
    if job_name:
        print("\n" + "=" * 55)
        print("TRAINING JOB CREATED SUCCESSFULLY")
        print("=" * 55)
        print(f"Job name: {job_name}")
        print("\nFor monitoring, use:")
        print(f"aws sagemaker describe-training-job --training-job-name {job_name} --region {AWS_REGION}")
        print("\nOr use the monitoring script:")
        print("python check_training_progress.py")
    else:
        print("\nFailed to create training job")

if __name__ == "__main__":
    main()