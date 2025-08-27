#!/usr/bin/env python3
"""
Test sklearn training specifically.
"""

import boto3
import sagemaker
from sagemaker.sklearn import SKLearn

def main():
    print("🔧 SKLearn Training Test")
    print("=" * 30)
    
    try:
        # Initialize SageMaker
        session = sagemaker.Session()
        role = "arn:aws:iam::691595239825:role/SageMakerExecutionRole"
        
        print(f"✅ SageMaker session initialized")
        
        # Try different sklearn versions to find one that works
        versions_to_try = ['1.0-1', '0.23-1', '1.2-1']
        
        for version in versions_to_try:
            print(f"🧪 Trying sklearn version: {version}")
            
            try:
                estimator = SKLearn(
                    entry_point='train_sklearn_model.py',
                    source_dir='src/training/sklearn',
                    role=role,
                    instance_type='ml.m5.large',
                    instance_count=1,
                    framework_version=version,
                    py_version='py3',
                    hyperparameters={
                        'model_type': 'random_forest'
                    }
                )
                
                print(f"✅ Estimator created with version {version}")
                
                # Try to start training
                train_uri = "s3://fire-detection-training-691595239825/training-data/fire_detection_train_1756294284.csv"
                val_uri = "s3://fire-detection-training-691595239825/training-data/fire_detection_val_1756294284.csv"
                
                estimator.fit({
                    'training': train_uri,
                    'validation': val_uri
                }, wait=False)
                
                job_name = estimator.latest_training_job.name
                print(f"✅ Training job started: {job_name}")
                
                # Check job status after a moment
                import time
                time.sleep(5)
                
                sagemaker_client = boto3.client('sagemaker')
                response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
                status = response['TrainingJobStatus']
                
                print(f"📊 Job status: {status}")
                
                if status == 'Failed' and 'FailureReason' in response:
                    print(f"❌ Failure reason: {response['FailureReason']}")
                else:
                    print(f"✅ Version {version} works!")
                    break
                    
            except Exception as e:
                print(f"❌ Version {version} failed: {str(e)}")
                continue
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()