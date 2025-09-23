#!/usr/bin/env python3
"""
Test pipeline with minimal scripts
"""

import boto3
import os
import tarfile
import tempfile
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET = 'fire-detection-training-691595239825'
S3_PREFIX = 'flir_scd41_training'

def package_minimal_code():
    """Package minimal training code"""
    print("Packaging minimal training code...")
    
    # Create a temporary directory for our code
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy our minimal scripts
        os.system(f"cp /Volumes/Ajay/saafe\\ copy\\ 3/new\\ model/synthetic-fire-prediction-system/minimal_train.py {temp_dir}/train")
        os.system(f"cp /Volumes/Ajay/saafe\\ copy\\ 3/new\\ model/synthetic-fire-prediction-system/minimal_serve.py {temp_dir}/serve")
        
        # Make scripts executable
        os.chmod(f"{temp_dir}/train", 0o755)
        os.chmod(f"{temp_dir}/serve", 0o755)
        
        # Create tar.gz file
        code_tar_path = f"{temp_dir}/minimal_code.tar.gz"
        with tarfile.open(code_tar_path, "w:gz") as tar:
            tar.add(f"{temp_dir}/train", arcname="train")
            tar.add(f"{temp_dir}/serve", arcname="serve")
        
        # Upload to S3
        s3_key = f"{S3_PREFIX}/code/minimal_code_{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar.gz"
        s3 = boto3.client('s3', region_name=AWS_REGION)
        s3.upload_file(code_tar_path, S3_BUCKET, s3_key)
        
        print(f"Code packaged and uploaded to s3://{S3_BUCKET}/{s3_key}")
        return f"s3://{S3_BUCKET}/{s3_key}"

def create_minimal_training_job():
    """Create a minimal SageMaker training job"""
    print("Creating minimal SageMaker training job...")
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Package code
    code_s3_uri = package_minimal_code()
    
    # Generate unique timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"flir-scd41-minimal-test-{timestamp}"
    
    try:
        # Create training job
        response = sagemaker.create_training_job(
            TrainingJobName=job_name,
            RoleArn="arn:aws:iam::691595239825:role/SageMakerExecutionRole",
            AlgorithmSpecification={
                'TrainingInputMode': 'File',
                'TrainingImage': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3'
            },
            InputDataConfig=[
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f's3://{S3_BUCKET}/{S3_PREFIX}/data/',
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'application/json',
                    'CompressionType': 'None'
                }
            ],
            OutputDataConfig={
                'S3OutputPath': f's3://{S3_BUCKET}/{S3_PREFIX}/models/'
            },
            ResourceConfig={
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': 3600  # 1 hour
            },
            HyperParameters={
                'sagemaker_program': 'train',
                'sagemaker_submit_directory': code_s3_uri
            }
        )
        
        print(f"Training job created successfully: {job_name}")
        print(f"Training job ARN: {response['TrainingJobArn']}")
        return job_name
        
    except Exception as e:
        print(f"Error creating training job: {e}")
        raise

def main():
    """Main function"""
    print("FLIR+SCD41 Fire Detection - Minimal Test")
    print("=" * 45)
    
    try:
        job_name = create_minimal_training_job()
        print(f"\nCreated training job: {job_name}")
        print("\nTo monitor progress, use:")
        print(f"aws sagemaker describe-training-job --training-job-name {job_name} --region {AWS_REGION}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()