#!/usr/bin/env python3
"""
Run ensemble debug training job
"""

import boto3
import datetime

def create_ensemble_debug_training_job():
    """Create ensemble debug SageMaker training job"""
    print("Creating ensemble debug SageMaker training job...")
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name='us-east-1')
    
    # Generate unique timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f'flir-scd41-ensemble-debug-{timestamp}'
    
    try:
        # Create training job
        response = sagemaker.create_training_job(
            TrainingJobName=job_name,
            RoleArn='arn:aws:iam::691595239825:role/SageMakerExecutionRole',
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
                'MaxRuntimeInSeconds': 3600  # 1 hour
            },
            HyperParameters={
                'sagemaker_program': 'train',
                'sagemaker_submit_directory': 's3://fire-detection-training-691595239825/flir_scd41_training/code/ensemble_debug_code.tar.gz'
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
    print("FLIR+SCD41 Fire Detection - Ensemble Debug Training")
    print("=" * 50)
    
    try:
        job_name = create_ensemble_debug_training_job()
        print(f"\nCreated training job: {job_name}")
        print("\nTo monitor progress, use:")
        print(f"aws sagemaker describe-training-job --training-job-name {job_name} --region us-east-1")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()