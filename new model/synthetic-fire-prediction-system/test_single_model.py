#!/usr/bin/env python3
"""
Test script to validate single model training works before running full ensemble.
"""

import boto3
import time
from sagemaker.pytorch import PyTorch
from sagemaker.sklearn import SKLearn

def test_pytorch_model():
    """Test a simple PyTorch model training."""
    print("🧪 Testing PyTorch model training...")
    
    # Get SageMaker session
    import sagemaker
    session = sagemaker.Session()
    role = "arn:aws:iam::691595239825:role/SageMakerExecutionRole"
    
    # Create PyTorch estimator
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
            'epochs': 2,  # Very short for testing
            'batch_size': 64,
            'learning_rate': 0.001
        }
    )
    
    # Use existing training data
    train_uri = "s3://fire-detection-training-691595239825/training-data/fire_detection_train_1756294284.csv"
    val_uri = "s3://fire-detection-training-691595239825/training-data/fire_detection_val_1756294284.csv"
    
    try:
        print("🚀 Starting PyTorch training job...")
        estimator.fit({
            'training': train_uri,
            'validation': val_uri
        }, wait=True)
        
        job_name = estimator.latest_training_job.name
        print(f"✅ PyTorch training completed: {job_name}")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch training failed: {str(e)}")
        return False

def test_sklearn_model():
    """Test a simple sklearn model training."""
    print("🧪 Testing SKLearn model training...")
    
    # Get SageMaker session
    import sagemaker
    session = sagemaker.Session()
    role = "arn:aws:iam::691595239825:role/SageMakerExecutionRole"
    
    # Create SKLearn estimator
    estimator = SKLearn(
        entry_point='train_sklearn_model.py',
        source_dir='src/training/sklearn',
        role=role,
        instance_type='ml.m5.large',
        instance_count=1,
        framework_version='1.0-1',
        py_version='py3',
        hyperparameters={
            'model_type': 'random_forest'
        }
    )
    
    # Use existing training data
    train_uri = "s3://fire-detection-training-691595239825/training-data/fire_detection_train_1756294284.csv"
    val_uri = "s3://fire-detection-training-691595239825/training-data/fire_detection_val_1756294284.csv"
    
    try:
        print("🚀 Starting SKLearn training job...")
        estimator.fit({
            'training': train_uri,
            'validation': val_uri
        }, wait=True)
        
        job_name = estimator.latest_training_job.name
        print(f"✅ SKLearn training completed: {job_name}")
        return True
        
    except Exception as e:
        print(f"❌ SKLearn training failed: {str(e)}")
        return False

def main():
    """Run single model tests."""
    print("🔧 Single Model Training Test")
    print("=" * 40)
    
    # Test one model at a time
    pytorch_success = test_pytorch_model()
    
    if pytorch_success:
        print("\n" + "=" * 40)
        sklearn_success = test_sklearn_model()
    else:
        sklearn_success = False
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"PyTorch Model: {'✅ PASS' if pytorch_success else '❌ FAIL'}")
    print(f"SKLearn Model: {'✅ PASS' if sklearn_success else '❌ FAIL'}")
    
    if pytorch_success and sklearn_success:
        print("\n🎉 All tests passed! Ensemble training should work.")
    else:
        print("\n⚠️ Some tests failed. Fix issues before running full ensemble.")

if __name__ == "__main__":
    main()