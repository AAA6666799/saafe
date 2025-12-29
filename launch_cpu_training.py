#!/usr/bin/env python3
"""
CPU-based SageMaker Training for IoT Fire Detection
Uses CPU instances which have higher availability
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import time

def launch_cpu_training():
    """Launch SageMaker training on CPU instances"""
    
    print("🚀 Launching SageMaker Training on CPU")
    print("=" * 50)
    
    # Setup
    session = sagemaker.Session()
    region = session.boto_region_name
    role = "arn:aws:iam::691595239825:role/SaafeIoTTrainingRole"
    
    print(f"Region: {region}")
    print(f"Role: {role}")
    
    # Create PyTorch estimator with CPU instance
    estimator = PyTorch(
        entry_point='minimal_train.py',
        source_dir='s3://saafe-iot-training-691595239825-us-east-1/code/',
        role=role,
        instance_type='ml.m5.2xlarge',  # CPU instance with good performance
        instance_count=1,
        framework_version='1.13.1',
        py_version='py39',
        hyperparameters={
            'epochs': 20,  # Fewer epochs for CPU
            'batch-size': 32,  # Smaller batch for CPU
            'learning-rate': 0.001
        },
        output_path='s3://saafe-iot-training-691595239825-us-east-1/output/',
        max_run=3600,  # 1 hour
        volume_size=30,
        tags=[
            {'Key': 'Project', 'Value': 'SaafeIoT'},
            {'Key': 'Environment', 'Value': 'CPUTraining'}
        ]
    )
    
    # Launch training
    job_name = f"saafe-iot-cpu-{int(time.time())}"
    
    print(f"Job name: {job_name}")
    print("Instance: ml.m5.2xlarge (8 vCPUs, 32GB RAM)")
    print("Expected time: 20-30 minutes")
    print("Expected cost: $2-4")
    print("Launching training...")
    
    try:
        estimator.fit(job_name=job_name, wait=False)
        
        print("✅ CPU Training job launched successfully!")
        print(f"Job name: {job_name}")
        print(f"Monitor at: https://{region}.console.aws.amazon.com/sagemaker/")
        
        return job_name
        
    except Exception as e:
        print(f"❌ Failed to launch training: {e}")
        return None

if __name__ == "__main__":
    launch_cpu_training()