#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Full Scale Training (Fixed)
This script creates properly configured SageMaker training jobs.
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

def package_training_code():
    """Package training code into a tar.gz file for SageMaker."""
    print("Packaging training code...")
    
    # Create a temporary directory for our code
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy our training and inference scripts
        os.system(f"cp flir_scd41_sagemaker_training_fixed.py {temp_dir}/train")
        os.system(f"cp flir_scd41_inference_fixed.py {temp_dir}/serve")
        
        # Create tar.gz file
        code_tar_path = f"{temp_dir}/code.tar.gz"
        with tarfile.open(code_tar_path, "w:gz") as tar:
            tar.add(f"{temp_dir}/train", arcname="train")
            tar.add(f"{temp_dir}/serve", arcname="serve")
        
        # Upload to S3
        s3_key = f"{S3_PREFIX}/code/code_fixed_{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar.gz"
        s3 = boto3.client('s3', region_name=AWS_REGION)
        s3.upload_file(code_tar_path, S3_BUCKET, s3_key)
        
        print(f"Code packaged and uploaded to s3://{S3_BUCKET}/{s3_key}")
        return f"s3://{S3_BUCKET}/{s3_key}"

def create_training_job(job_name, algorithm_name, instance_type='ml.m5.2xlarge'):
    """Create a SageMaker training job."""
    print(f"Creating {algorithm_name} training job: {job_name}")
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Package code
    code_s3_uri = package_training_code()
    
    try:
        if algorithm_name == 'xgboost':
            # XGBoost training job
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
                                'S3Uri': f's3://{S3_BUCKET}/{S3_PREFIX}/data/',
                                'S3DataDistributionType': 'FullyReplicated'
                            }
                        },
                        'ContentType': 'application/json'
                    }
                ],
                OutputDataConfig={
                    'S3OutputPath': f's3://{S3_BUCKET}/{S3_PREFIX}/models/'
                },
                ResourceConfig={
                    'InstanceType': instance_type,
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
                    'num_round': '100'
                }
            )
        else:
            # Scikit-learn training job
            response = sagemaker.create_training_job(
                TrainingJobName=job_name,
                RoleArn="arn:aws:iam::691595239825:role/SageMakerExecutionRole",
                AlgorithmSpecification={
                    'TrainingInputMode': 'File',
                    'TrainingImage': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3'
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
                        'ContentType': 'application/json'
                    }
                ],
                OutputDataConfig={
                    'S3OutputPath': f's3://{S3_BUCKET}/{S3_PREFIX}/models/'
                },
                ResourceConfig={
                    'InstanceType': instance_type,
                    'InstanceCount': 1,
                    'VolumeSizeInGB': 50
                },
                StoppingCondition={
                    'MaxRuntimeInSeconds': 7200  # 2 hours
                },
                HyperParameters={
                    'sagemaker_program': 'train'
                }
            )
        
        print(f"Training job created successfully: {job_name}")
        print(f"Training job ARN: {response['TrainingJobArn']}")
        return job_name
        
    except Exception as e:
        print(f"Error creating training job: {e}")
        raise

def main():
    """Main function to start full-scale training."""
    print("FLIR+SCD41 Fire Detection - Full Scale Training (Fixed)")
    print("=" * 55)
    
    # Generate unique timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create training jobs
    jobs = []
    
    # Random Forest training job
    rf_job_name = f"flir-scd41-rf-training-fixed-{timestamp}"
    rf_job = create_training_job(rf_job_name, 'randomforest')
    jobs.append(('Random Forest', rf_job))
    
    # XGBoost training job
    xgb_job_name = f"flir-scd41-xgb-training-fixed-{timestamp}"
    xgb_job = create_training_job(xgb_job_name, 'xgboost')
    jobs.append(('XGBoost', xgb_job))
    
    print("\n" + "=" * 55)
    print("TRAINING JOBS CREATED SUCCESSFULLY")
    print("=" * 55)
    for model_type, job_name in jobs:
        print(f"{model_type} job: {job_name}")
    
    print("\nFor monitoring, use:")
    for model_type, job_name in jobs:
        print(f"aws sagemaker describe-training-job --training-job-name {job_name} --region {AWS_REGION}")
    
    print("\nOr use the monitoring script:")
    print("python comprehensive_monitor_deploy.py")

if __name__ == "__main__":
    main()