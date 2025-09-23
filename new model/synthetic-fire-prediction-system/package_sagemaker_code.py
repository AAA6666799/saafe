#!/usr/bin/env python3
"""
Package Code for SageMaker Training
This script packages the training code into a tar.gz file for SageMaker.
"""

import os
import tarfile
import boto3
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET = 'fire-detection-training-691595239825'
S3_PREFIX = 'flir_scd41_training'

def create_code_package():
    """Create a code package for SageMaker training."""
    
    # Create a temporary directory for packaging
    package_dir = "/tmp/sagemaker_code_package"
    os.makedirs(package_dir, exist_ok=True)
    
    # Copy required files
    files_to_package = [
        "flir_scd41_sagemaker_training.py",
        "flir_scd41_inference.py"
    ]
    
    for file_name in files_to_package:
        source_path = f"/tmp/{file_name}"
        dest_path = os.path.join(package_dir, file_name)
        
        # Copy file if it exists
        if os.path.exists(source_path):
            with open(source_path, 'rb') as src, open(dest_path, 'wb') as dst:
                dst.write(src.read())
            print(f"Copied {file_name} to package directory")
        else:
            print(f"Warning: {file_name} not found at {source_path}")
    
    # Create requirements file
    requirements_content = """
scikit-learn==1.0.2
pandas==1.3.5
numpy==1.21.6
joblib==1.1.0
"""
    
    requirements_path = os.path.join(package_dir, "requirements.txt")
    with open(requirements_path, 'w') as f:
        f.write(requirements_content)
    print("Created requirements.txt")
    
    # Create tar.gz package
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    package_name = f"sagemaker_submit_directory_{timestamp}.tar.gz"
    package_path = f"/tmp/{package_name}"
    
    with tarfile.open(package_path, "w:gz") as tar:
        tar.add(package_dir, arcname=".")
    
    print(f"Created code package: {package_path}")
    
    # Upload to S3
    s3 = boto3.client('s3', region_name=AWS_REGION)
    s3_key = f"{S3_PREFIX}/code/{package_name}"
    
    try:
        s3.upload_file(package_path, S3_BUCKET, s3_key)
        print(f"Code package uploaded to s3://{S3_BUCKET}/{s3_key}")
        return f"s3://{S3_BUCKET}/{s3_key}"
    except Exception as e:
        print(f"Error uploading code package: {e}")
        return None

def create_sklearn_training_job():
    """Create a SageMaker training job using the sklearn container."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Generate unique job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"flir-scd41-sklearn-training-{timestamp}"
    
    # Upload code package
    code_package_url = create_code_package()
    
    if not code_package_url:
        print("Failed to create code package")
        return None
    
    # Define training job parameters
    training_params = {
        'TrainingJobName': job_name,
        'RoleArn': 'arn:aws:iam::691595239825:role/SageMakerExecutionRole',
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
            'InstanceType': 'ml.m5.2xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 50
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 7200  # 2 hours
        },
        'HyperParameters': {
            'sagemaker_container_log_level': '20',
            'sagemaker_job_name': job_name,
            'sagemaker_program': 'flir_scd41_sagemaker_training.py',
            'sagemaker_submit_directory': code_package_url
        }
    }
    
    try:
        # Create training job
        response = sagemaker.create_training_job(**training_params)
        print(f"Sklearn training job created successfully: {job_name}")
        print(f"Training job ARN: {response['TrainingJobArn']}")
        return job_name
    except Exception as e:
        print(f"Error creating sklearn training job: {e}")
        return None

if __name__ == "__main__":
    print("FLIR+SCD41 Fire Detection - Code Packaging for SageMaker")
    print("=" * 55)
    
    # Create sklearn training job
    job_name = create_sklearn_training_job()
    
    if job_name:
        print(f"\nTraining job created: {job_name}")
        print("You can monitor the job with:")
        print(f"aws sagemaker describe-training-job --training-job-name {job_name} --region {AWS_REGION}")
    else:
        print("\nFailed to create training job")