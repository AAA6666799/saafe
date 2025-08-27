# Fire Detection AI Training on AWS SageMaker with ml.p3.16xlarge

This package contains scripts and guides to help you run the Fire Detection AI training notebooks on AWS SageMaker using the powerful ml.p3.16xlarge instance type.

## Files Included

- `upload_notebooks_to_s3.py`: Python script to upload notebooks to an S3 bucket
- `upload_to_s3.sh`: Shell script wrapper for the upload process
- `download_notebooks_from_s3.py`: Python script to download notebooks from S3 to SageMaker
- `download_from_s3.sh`: Shell script wrapper for the download process
- `sagemaker_p3_16xlarge_guide.md`: Comprehensive guide for the entire process
- `README_SAGEMAKER_DEPLOYMENT.md`: This file
- `test_s3_connection.py`: Script to test S3 connection and permissions

## Quick Start Guide

### Step 0: Test S3 Connection (Recommended)

Before uploading or downloading notebooks, it's recommended to test your S3 connection and permissions:

```bash
./test_s3_connection.py --bucket your-bucket-name
```

This script will:
- Verify your AWS credentials
- Check if the bucket exists and is accessible
- Test upload and download permissions
- Check your SageMaker quota for ml.p3.16xlarge instances (if possible)

### Step 1: Upload Notebooks to S3

Run the upload script on your local machine:

```bash
./upload_to_s3.sh -b your-bucket-name
```

Replace `your-bucket-name` with your actual S3 bucket name.

### Step 2: Launch a SageMaker Notebook Instance

1. Go to Amazon SageMaker in the AWS Console
2. Navigate to "Notebooks" → "Notebook instances"
3. Click "Create notebook instance"
4. Configure the instance:
   - Name: `fire-detection-training`
   - Notebook instance type: `ml.p3.16xlarge`
   - Platform identifier: Select the latest version
   - IAM role: Create or select a role with SageMaker and S3 access
   - VPC: Optional, select if needed
   - Root access: Enable for users (recommended)
5. Click "Create notebook instance"

### Step 3: Download Notebooks to SageMaker Instance

1. Upload `download_notebooks_from_s3.py` and `download_from_s3.sh` to your SageMaker instance
2. Open a terminal in JupyterLab
3. Run:

```bash
chmod +x download_from_s3.sh
./download_from_s3.sh -b your-bucket-name
```

### Step 4: Run the Notebooks

1. Navigate to the `fire_detection` directory in JupyterLab
2. Run the GPU setup script:

```bash
python fire_detection/gpu_setup.py
```

3. Open `fire_detection_5m_training_simplified.ipynb` or start with `fire_detection_5m_training_part1.ipynb`
4. Select the "Fire Detection" kernel
5. Run the notebook cells

## Detailed Instructions

For detailed instructions, please refer to `sagemaker_p3_16xlarge_guide.md`.

## Cost Management

The ml.p3.16xlarge is expensive (~$24.48 per hour). To manage costs:

1. Always stop the instance when not in use
2. Add this code at the end of your notebook to stop the instance when done:

```python
import boto3

def stop_notebook_instance(instance_name):
    client = boto3.client('sagemaker')
    client.stop_notebook_instance(NotebookInstanceName=instance_name)
    print(f"Stopping notebook instance: {instance_name}")

# Replace with your instance name
stop_notebook_instance('fire-detection-training')
```

3. Set up CloudWatch alarms to notify you if the instance runs for too long

## Workflow Overview

The complete workflow for running Fire Detection AI training on SageMaker consists of:

1. **Local Machine**:
   - Test S3 connection with `test_s3_connection.py`
   - Upload notebooks to S3 with `upload_to_s3.sh`

2. **AWS Console**:
   - Launch a SageMaker notebook instance with ml.p3.16xlarge

3. **SageMaker Instance**:
   - Download notebooks from S3 with `download_from_s3.sh`
   - Set up the environment and GPU configuration
   - Run the notebooks with the Fire Detection kernel

4. **After Training**:
   - Save models to S3
   - Stop the instance to avoid unnecessary costs

## Troubleshooting

If you encounter issues:

1. **Connection Issues**:
   - Run `test_s3_connection.py` to diagnose S3 permissions
   - Check your AWS credentials and IAM permissions
   - Verify network connectivity to AWS services

2. **Instance Quota Issues**:
   - Verify that your account has quota for ml.p3.16xlarge instances
   - Request a quota increase if needed through AWS Service Quotas console

3. **SageMaker Issues**:
   - Check the SageMaker instance logs in CloudWatch
   - Verify that the instance has the necessary IAM permissions
   - Try restarting the instance if Jupyter is unresponsive

4. **Training Issues**:
   - Check GPU utilization with `nvidia-smi`
   - Adjust batch size if you encounter out-of-memory errors
   - Use gradient accumulation for large models

For more detailed troubleshooting, refer to the troubleshooting section in `sagemaker_p3_16xlarge_guide.md`.

## Next Steps

After training, you can:

1. Save your models to S3
2. Deploy models using SageMaker endpoints
3. Set up monitoring and evaluation pipelines

For production training, consider converting your notebook to a SageMaker Training Job for better resource management and scalability.