# FLIR+SCD41 Fire Detection System - AWS Training Pipeline

This document explains how to use the AWS-based training pipeline for the FLIR+SCD41 fire detection system.

## Overview

The AWS training pipeline leverages Amazon SageMaker and S3 to train machine learning models for fire detection using FLIR thermal camera and SCD41 CO₂ sensor data.

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Python 3.8+
3. Required Python packages (installed via `requirements-aws.txt`)

## Setup

1. Install required dependencies:
   ```bash
   pip install -r requirements-aws.txt
   ```

2. Ensure AWS credentials are configured:
   ```bash
   aws configure
   ```

## Usage

### Basic Usage

Run the training pipeline with default settings:
```bash
python aws_training_pipeline_fixed.py
```

### Custom Parameters

```bash
python aws_training_pipeline_fixed.py \
  --samples 50000 \
  --instance-type ml.m5.xlarge \
  --bucket my-custom-bucket \
  --prefix my-training-prefix
```

### Parameters

- `--samples`: Number of synthetic samples to generate (default: 10000)
- `--instance-type`: SageMaker instance type for training (default: ml.m5.large)
- `--bucket`: S3 bucket for data storage (default: fire-detection-training-691595239825)
- `--prefix`: S3 prefix for data organization (default: flir_scd41_training)

## Pipeline Steps

1. **Data Generation**: Creates synthetic FLIR+SCD41 sensor data
2. **Feature Extraction**: Extracts relevant features from raw sensor data
3. **Model Training**: Trains multiple models using SageMaker
4. **Ensemble Creation**: Combines individual models into an ensemble

## AWS Resources Used

- **Amazon S3**: Data storage for training datasets and model artifacts
- **Amazon SageMaker**: Managed machine learning service for model training
- **IAM Roles**: For SageMaker execution permissions

## Data Structure

The pipeline generates data with 18 features from FLIR+SCD41 sensors:
- 15 thermal features from FLIR Lepton 3.5 camera
- 3 gas features from Sensirion SCD41 CO₂ sensor

## Model Architecture

The system supports multiple model types:
- XGBoost classifiers
- Neural networks
- Ensemble methods combining multiple models

## Monitoring and Evaluation

Training jobs can be monitored through:
- AWS CloudWatch logs
- SageMaker console
- S3 model artifacts

## Cost Considerations

- SageMaker training instances are charged per hour
- S3 storage costs for datasets and models
- Data transfer costs between services

## Troubleshooting

### Common Issues

1. **AWS Credentials**: Ensure `aws configure` has been run
2. **Permissions**: Verify IAM role has SageMaker and S3 permissions
3. **S3 Bucket**: Ensure the specified bucket exists and is accessible

### Error Messages

- `AccessDenied`: Check IAM permissions for S3 and SageMaker
- `ResourceNotFound`: Verify AWS region and resource names

## Advanced Usage

### Custom Training Scripts

To use custom training logic, modify the training script in the pipeline.

### Hyperparameter Tuning

SageMaker automatic hyperparameter tuning can be enabled by modifying the training job configuration.

### Distributed Training

For large datasets, configure SageMaker distributed training options.

## Security

- All data is encrypted at rest in S3
- SageMaker training jobs use VPC isolation
- IAM roles follow principle of least privilege

## Cleanup

To avoid ongoing charges:
1. Stop running SageMaker training jobs
2. Delete unused S3 objects
3. Terminate SageMaker notebook instances if used

## Support

For issues with the pipeline, check:
1. AWS service status pages
2. CloudWatch logs for detailed error messages
3. SageMaker training job logs