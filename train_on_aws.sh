#!/bin/bash
# Quick AWS Training Script

echo "🔥 Training Saafe Models on AWS"
echo "================================"

# 1. Train locally first
echo "1. Training models locally..."
python quick_aws_train.py

# 2. Upload to S3
echo "2. Uploading to S3..."
aws s3 cp models/transformer_model.pth s3://saafe-models/transformer_model.pth
aws s3 cp models/anti_hallucination.pkl s3://saafe-models/anti_hallucination.pkl

# 3. Create SageMaker endpoint
echo "3. Creating SageMaker endpoint..."
python -c "
import boto3
import json

sagemaker = boto3.client('sagemaker')

# Create model
model_name = 'saafe-fire-model'
sagemaker.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.0-gpu-py39-cu118-ubuntu20.04-sagemaker',
        'ModelDataUrl': 's3://saafe-models/transformer_model.pth'
    },
    ExecutionRoleArn='arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole'
)

print('✅ Model created on SageMaker')
"

echo "✅ Training pipeline completed!"