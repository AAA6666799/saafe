#!/usr/bin/env python3
"""
Complete pipeline: Data cleaning -> SageMaker GPU training
"""

from data_cleaning_pipeline import DataCleaner
from sagemaker_gpu_training import SageMakerGPUTrainer, setup_sagemaker_resources
import boto3

def main():
    print("=== Starting Complete ML Pipeline ===\n")
    
    # Step 1: Clean data
    print("Step 1: Cleaning data...")
    cleaner = DataCleaner()
    cleaner.process_all_datasets()
    
    # Step 2: Setup AWS resources
    print("\nStep 2: Setting up AWS resources...")
    bucket, role = setup_sagemaker_resources()
    
    if not bucket or not role:
        print("AWS setup failed. Please check your credentials and permissions.")
        return
    
    # Step 3: Upload cleaned data to S3
    print("\nStep 3: Uploading cleaned data to S3...")
    cleaner.upload_to_s3(bucket)
    
    # Step 4: Start SageMaker training
    print("\nStep 4: Starting SageMaker GPU training...")
    trainer = SageMakerGPUTrainer(role, bucket)
    
    # Use GPU instance for training
    estimator = trainer.create_training_job(
        instance_type="ml.g4dn.xlarge",  # Cost-effective GPU
        instance_count=1
    )
    
    print("\n=== Pipeline Complete ===")
    print(f"✓ Data cleaned and uploaded to S3")
    print(f"✓ Model training completed on GPU")
    print(f"✓ Model artifacts saved to s3://{bucket}/model-output/")
    
    # Optional: Deploy model
    deploy = input("\nDeploy model to endpoint? (y/n): ")
    if deploy.lower() == 'y':
        print("Deploying model...")
        predictor = trainer.deploy_model(estimator)
        print(f"Model deployed to endpoint: {predictor.endpoint_name}")

if __name__ == "__main__":
    main()