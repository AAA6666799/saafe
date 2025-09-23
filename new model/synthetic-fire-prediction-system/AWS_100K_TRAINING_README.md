# FLIR+SCD41 Fire Detection System - AWS 100K Training Pipeline

This document explains how to use the AWS training pipeline for the FLIR+SCD41 fire detection system with 100K+ samples.

## Overview

The AWS 100K training pipeline is designed to:

1. Generate 100K+ synthetic training samples with realistic fire patterns
2. Train multiple ensemble models using AWS SageMaker
3. Validate and test models using proper data splitting techniques
4. Deploy the best performing model to a SageMaker endpoint
5. Monitor training progress and endpoint performance

## Prerequisites

Before running the pipeline, ensure you have:

1. AWS CLI configured with appropriate credentials
2. SageMaker execution role with necessary permissions
3. S3 bucket `fire-detection-training-691595239825` accessible
4. Required Python packages installed (boto3, numpy, pandas, scikit-learn)

## Pipeline Components

### 1. Data Generation (`aws_100k_training_pipeline.py`)

Generates 100K synthetic samples with realistic fire patterns and uploads them to S3.

### 2. Training Jobs (`aws_100k_training_pipeline.py`)

Creates SageMaker training jobs for ensemble methods:
- Random Forest
- Gradient Boosting
- Logistic Regression

### 3. Monitoring (`monitor_100k_training.py`)

Tracks training job progress and provides detailed metrics.

### 4. Deployment (`deploy_100k_model.py`)

Deploys trained models to SageMaker endpoints for inference.

## Usage Instructions

### Step 1: Run the Training Pipeline

```bash
python aws_100k_training_pipeline.py
```

This will:
1. Generate 100K synthetic samples
2. Upload data to S3
3. Create SageMaker training jobs
4. Output job names for monitoring

### Step 2: Monitor Training Progress

```bash
python monitor_100k_training.py job_name_1 job_name_2 job_name_3 --wait --metrics
```

Options:
- `--wait`: Wait until all jobs complete
- `--metrics`: Show detailed metrics for completed jobs
- `--interval N`: Check interval in seconds (default: 60)

### Step 3: Deploy the Best Model

Identify the best performing model from the training results, then deploy:

```bash
python deploy_100k_model.py \
  --model-uri s3://fire-detection-training-691595239825/flir_scd41_training/models/model-name/output/model.tar.gz \
  --model-name flir-scd41-best-model \
  --instance-type ml.t2.medium \
  --wait \
  --test
```

Options:
- `--wait`: Wait for endpoint to be in service
- `--test`: Test the deployed endpoint with sample data
- `--instance-type`: EC2 instance type (default: ml.t2.medium)

## Training Process Details

### Data Generation

The pipeline generates 100K synthetic samples with:
- 15 FLIR thermal camera features
- 3 SCD41 gas sensor features
- Realistic fire probability based on multiple interacting factors
- Proper stratification to prevent data leakage

### Model Training

Each training job uses:
- 70% of data for training
- 15% for validation
- 15% for testing
- Performance-based ensemble weighting
- Early stopping and regularization

### Ensemble Methods

The pipeline trains three models:
1. **Random Forest**: 200 estimators, max depth 10
2. **Gradient Boosting**: 200 estimators, learning rate 0.1
3. **Logistic Regression**: L2 regularization

Ensemble weights are calculated based on validation performance using exponential scaling.

## Monitoring and Validation

### Training Job Monitoring

Use the monitoring script to track:
- Job status (InProgress, Completed, Failed)
- Training time and billable time
- Instance type and count
- Final metrics (if available)

### Model Performance Metrics

The pipeline evaluates models using:
- Accuracy
- F1-Score
- Precision
- Recall
- AUC (Area Under Curve)

## Deployment and Inference

### Endpoint Deployment

Deployed endpoints support:
- CSV input format (18 features)
- JSON output (fire probability)
- Real-time inference
- Automatic scaling

### Sample Inference Request

```python
import boto3

sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Sample data (18 features)
test_data = "25.0,5.0,45.0,40.0,10.0,5.0,2.0,1.0,3.0,1.5,5.0,2.0,30.0,5.0,1.0,450.0,50.0,5.0"

response = sagemaker_runtime.invoke_endpoint(
    EndpointName='flir-scd41-best-model-endpoint',
    ContentType='text/csv',
    Body=test_data
)

result = response['Body'].read().decode()
print(f"Fire probability: {result}")
```

## Cost Considerations

### Instance Types

- **Training**: ml.m5.4xlarge (recommended for 100K samples)
- **Inference**: ml.t2.medium (default for endpoints)

### Estimated Costs

For 100K samples:
- **Training**: ~$5-10 per job (1-2 hours)
- **Storage**: ~$0.10 per GB-month for S3
- **Inference**: ~$0.05-0.10 per hour for endpoint

## Troubleshooting

### Common Issues

1. **Training Job Failures**
   - Check CloudWatch logs for detailed error messages
   - Verify S3 permissions and data accessibility
   - Ensure sufficient volume size for large datasets

2. **Deployment Failures**
   - Verify model artifact URI is correct
   - Check SageMaker execution role permissions
   - Ensure instance type is available in your region

3. **Performance Issues**
   - Monitor CPU and memory utilization
   - Consider larger instance types for better performance
   - Optimize model hyperparameters

### Useful AWS CLI Commands

```bash
# Describe training job
aws sagemaker describe-training-job --training-job-name job-name --region us-east-1

# List training jobs
aws sagemaker list-training-jobs --region us-east-1

# Describe endpoint
aws sagemaker describe-endpoint --endpoint-name endpoint-name --region us-east-1

# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name endpoint-name --region us-east-1
```

## Next Steps

1. **Model Optimization**: Fine-tune hyperparameters for better performance
2. **Feature Engineering**: Add more sophisticated features
3. **Real Data Integration**: Incorporate real FLIR and SCD41 sensor data
4. **Continuous Training**: Implement automated retraining pipeline
5. **A/B Testing**: Compare multiple model versions in production

## Support

For issues with the pipeline, contact the development team or check the AWS documentation for SageMaker and related services.