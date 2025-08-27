import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import json
from datetime import datetime

class SageMakerGPUTrainer:
    def __init__(self, role_arn, bucket_name):
        self.session = sagemaker.Session()
        self.role = role_arn
        self.bucket = bucket_name
        self.region = boto3.Session().region_name
        
    def create_training_job(self, 
                          script_path="sagemaker_source/sagemaker_iot_train.py",
                          instance_type="ml.g4dn.xlarge",  # GPU instance
                          instance_count=1):
        """Launch SageMaker training job with GPU"""
        
        # Training data location
        training_data = f"s3://{self.bucket}/training-data/"
        
        # Create PyTorch estimator
        estimator = PyTorch(
            entry_point=script_path,
            role=self.role,
            instance_type=instance_type,
            instance_count=instance_count,
            framework_version='1.12.0',
            py_version='py38',
            hyperparameters={
                'epochs': 50,
                'batch_size': 64,
                'learning_rate': 0.001,
                'use_gpu': True
            },
            output_path=f"s3://{self.bucket}/model-output/",
            base_job_name='iot-fire-detection-gpu'
        )
        
        # Start training
        training_input = TrainingInput(training_data, content_type='text/csv')
        
        print(f"Starting training job on {instance_type}...")
        estimator.fit({'training': training_input})
        
        return estimator
    
    def deploy_model(self, estimator, instance_type="ml.g4dn.xlarge"):
        """Deploy trained model to endpoint"""
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=f"iot-fire-detection-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        return predictor

def setup_sagemaker_resources():
    """Setup S3 bucket and IAM role for SageMaker"""
    
    # Create S3 bucket
    s3 = boto3.client('s3')
    bucket_name = f"sagemaker-iot-training-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Created S3 bucket: {bucket_name}")
    except Exception as e:
        print(f"Bucket creation failed: {e}")
        return None, None
    
    # Get SageMaker execution role
    try:
        iam = boto3.client('iam')
        role_name = 'SageMakerExecutionRole'
        role_arn = f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/{role_name}"
        print(f"Using IAM role: {role_arn}")
    except Exception as e:
        print(f"Role setup failed: {e}")
        return bucket_name, None
    
    return bucket_name, role_arn

if __name__ == "__main__":
    # Setup AWS resources
    bucket, role = setup_sagemaker_resources()
    
    if bucket and role:
        # Initialize trainer
        trainer = SageMakerGPUTrainer(role, bucket)
        
        # Start training
        estimator = trainer.create_training_job()
        
        print("Training completed successfully!")
        print(f"Model artifacts saved to: s3://{bucket}/model-output/")