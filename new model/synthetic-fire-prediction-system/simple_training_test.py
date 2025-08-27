#!/usr/bin/env python3
"""
Simplified training script to diagnose the core issue.
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

def main():
    print("🔧 Simple Training Test")
    print("=" * 30)
    
    try:
        # Initialize SageMaker
        session = sagemaker.Session()
        role = "arn:aws:iam::691595239825:role/SageMakerExecutionRole"
        
        print(f"✅ SageMaker session initialized")
        print(f"✅ Using role: {role}")
        
        # Create the simplest possible estimator
        estimator = PyTorch(
            entry_point='train_pytorch_model.py',
            source_dir='src/training/pytorch',
            role=role,
            instance_type='ml.m5.large',
            instance_count=1,
            framework_version='1.12.0',
            py_version='py38',
            hyperparameters={
                'model_type': 'lstm_classifier',
                'epochs': 1,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        )
        
        print(f"✅ Estimator created successfully")
        
        # Check if training data exists
        train_uri = "s3://fire-detection-training-691595239825/training-data/fire_detection_train_1756294284.csv"
        val_uri = "s3://fire-detection-training-691595239825/training-data/fire_detection_val_1756294284.csv"
        
        s3 = boto3.client('s3')
        try:
            s3.head_object(Bucket='fire-detection-training-691595239825', Key='training-data/fire_detection_train_1756294284.csv')
            print(f"✅ Training data found")
        except:
            print(f"❌ Training data not found at {train_uri}")
            return
        
        # Try to start training
        print("🚀 Starting training...")
        estimator.fit({
            'training': train_uri,
            'validation': val_uri
        }, wait=False)  # Don't wait
        
        job_name = estimator.latest_training_job.name
        print(f"✅ Training job started: {job_name}")
        
        # Check job status
        sagemaker_client = boto3.client('sagemaker')
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        
        print(f"📊 Job status: {status}")
        
        if 'FailureReason' in response:
            print(f"❌ Failure reason: {response['FailureReason']}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()