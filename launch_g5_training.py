#!/usr/bin/env python3
"""
G5 GPU Training for IoT Fire Detection
Uses ml.g5.xlarge with NVIDIA A10G GPU (approved quota)
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import time

def launch_g5_training():
    """Launch SageMaker training on G5 instances with approved quota"""
    
    print("🚀 Launching SageMaker Training on G5 (NVIDIA A10G)")
    print("=" * 60)
    
    # Setup
    session = sagemaker.Session()
    region = session.boto_region_name
    role = "arn:aws:iam::691595239825:role/SaafeIoTTrainingRole"
    
    print(f"Region: {region}")
    print(f"Role: {role}")
    print(f"Instance: ml.g5.xlarge (NVIDIA A10G GPU)")
    print(f"Quota: APPROVED ✅")
    
    # Create PyTorch estimator with G5 instance
    estimator = PyTorch(
        entry_point='minimal_train.py',
        source_dir='s3://saafe-iot-training-691595239825-us-east-1/code/',
        role=role,
        instance_type='ml.g5.xlarge',  # NVIDIA A10G GPU
        instance_count=1,
        framework_version='1.13.1',
        py_version='py39',
        hyperparameters={
            'epochs': 50,  # More epochs with GPU
            'batch-size': 128,  # Larger batch with GPU
            'learning-rate': 0.001
        },
        output_path='s3://saafe-iot-training-691595239825-us-east-1/output/',
        max_run=3600,  # 1 hour
        volume_size=30,
        tags=[
            {'Key': 'Project', 'Value': 'SaafeIoT'},
            {'Key': 'Environment', 'Value': 'G5Training'},
            {'Key': 'GPU', 'Value': 'A10G'}
        ]
    )
    
    # Launch training
    job_name = f"saafe-iot-g5-{int(time.time())}"
    
    print(f"Job name: {job_name}")
    print("GPU: NVIDIA A10G (24GB VRAM)")
    print("Expected time: 15-25 minutes")
    print("Expected cost: $3-5")
    print("Launching training...")
    
    try:
        estimator.fit(job_name=job_name, wait=False)
        
        print("✅ G5 Training job launched successfully!")
        print(f"🎯 Job name: {job_name}")
        print(f"🔍 Monitor at: https://{region}.console.aws.amazon.com/sagemaker/")
        print(f"💰 Cost: ~$0.20/hour")
        
        return job_name
        
    except Exception as e:
        print(f"❌ Failed to launch training: {e}")
        return None

if __name__ == "__main__":
    launch_g5_training()