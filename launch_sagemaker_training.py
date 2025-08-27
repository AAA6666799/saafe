#!/usr/bin/env python3
"""
Bulletproof SageMaker Training Launcher for IoT Fire Detection
Uses SageMaker Python SDK for guaranteed success
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import time

def launch_training():
    """Launch SageMaker training with Python SDK"""
    
    print("🚀 Launching SageMaker Training with Python SDK")
    print("=" * 50)
    
    # Setup
    session = sagemaker.Session()
    region = session.boto_region_name
    
    # Use existing role
    role = "arn:aws:iam::691595239825:role/SaafeIoTTrainingRole"
    
    print(f"Region: {region}")
    print(f"Role: {role}")
    
    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point='minimal_train.py',
        source_dir='.',  # Use local directory for the training script
        role=role,
        instance_type='ml.g5.xlarge',  # Use g5 instances as requested
        instance_count=2,  # Use two instances
        framework_version='1.13.1',
        py_version='py39',
        hyperparameters={
            'epochs': 30,
            'batch-size': 64,
            'learning-rate': 0.001
        },
        output_path='s3://saafe-iot-training-691595239825-us-east-1/output/',
        max_run=3600,  # 1 hour
        volume_size=30,
        use_spot_instances=False,  # Disable spot instances to avoid quota issues
        checkpoint_s3_uri='s3://saafe-iot-training-691595239825-us-east-1/checkpoints/',
        tags=[
            {'Key': 'Project', 'Value': 'SaafeIoT'},
            {'Key': 'Environment', 'Value': 'Training'}
        ]
    )
    
    # Launch training
    job_name = f"saafe-iot-sdk-{int(time.time())}"
    
    print(f"Job name: {job_name}")
    print("Launching training...")
    
    try:
        # Define the S3 input channel for training data
        s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://synthetic-data-4/datasets/', content_type='application/x-npy')
        
        estimator.fit({'training': s3_input_train}, job_name=job_name, wait=True)
        
        print("✅ Training job launched successfully!")
        print(f"Job name: {job_name}")
        print(f"Monitor at: https://{region}.console.aws.amazon.com/sagemaker/")
        
        return job_name
        
    except Exception as e:
        print(f"❌ Failed to launch training: {e}")
        return None

if __name__ == "__main__":
    launch_training()