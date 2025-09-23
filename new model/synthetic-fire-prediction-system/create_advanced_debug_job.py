#!/usr/bin/env python3
"""
Create advanced debug training job
"""

import boto3
import datetime

def create_advanced_debug_job():
    sagemaker = boto3.client('sagemaker', region_name='us-east-1')
    
    # Generate job name
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f'flir-scd41-advanced-debug-{timestamp}'
    
    print(f"Creating advanced debug training job: {job_name}")
    
    try:
        response = sagemaker.create_training_job(
            TrainingJobName=job_name,
            RoleArn="arn:aws:iam::691595239825:role/SageMakerExecutionRole",
            AlgorithmSpecification={
                'TrainingInputMode': 'File',
                'TrainingImage': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3'
            },
            InputDataConfig=[
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': 's3://fire-detection-training-691595239825/flir_scd41_training/data/',
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'application/json',
                    'CompressionType': 'None'
                }
            ],
            OutputDataConfig={
                'S3OutputPath': 's3://fire-detection-training-691595239825/flir_scd41_training/models/'
            },
            ResourceConfig={
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': 3600
            },
            HyperParameters={
                'sagemaker_program': 'train',
                'sagemaker_submit_directory': 's3://fire-detection-training-691595239825/flir_scd41_training/code/advanced_debug_code.tar.gz'
            }
        )
        
        print(f"✅ Advanced debug training job created successfully!")
        print(f"Job ARN: {response['TrainingJobArn']}")
        return job_name
        
    except Exception as e:
        print(f"❌ Error creating advanced debug training job: {e}")
        raise

if __name__ == "__main__":
    job_name = create_advanced_debug_job()
    print(f"\nTo monitor the job, use:")
    print(f"aws sagemaker describe-training-job --training-job-name {job_name} --region us-east-1")