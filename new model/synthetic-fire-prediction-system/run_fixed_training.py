#!/usr/bin/env python3
"""
Run a fixed training job
"""

import boto3
import datetime

def create_fixed_training_job():
    """Create a training job with our fixed code"""
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name='us-east-1')
    
    # Generate unique job name
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f'flir-scd41-fixed-ensemble-{timestamp}'
    
    print(f"Creating fixed training job: {job_name}")
    print("=" * 50)
    
    try:
        # Create training job with our fixed code
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
                'InstanceType': 'ml.m5.4xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 100
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': 14400  # 4 hours
            },
            HyperParameters={
                'sagemaker_program': 'train',
                'sagemaker_submit_directory': 's3://fire-detection-training-691595239825/flir_scd41_training/code/fixed_ensemble_code.tar.gz'
            }
        )
        
        print(f"✅ Training job created successfully!")
        print(f"Job ARN: {response['TrainingJobArn']}")
        print()
        print(f"📝 To monitor the job, run:")
        print(f"aws sagemaker describe-training-job --training-job-name {job_name} --region us-east-1")
        print()
        print(f"Or use our monitoring script:")
        print(f"python monitor_fixed_jobs.py")
        
        return job_name
        
    except Exception as e:
        print(f"❌ Error creating training job: {e}")
        raise

if __name__ == "__main__":
    create_fixed_training_job()