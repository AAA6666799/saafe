# Running Fire Detection AI Training on SageMaker with ml.p3.16xlarge

This guide provides step-by-step instructions for running the Fire Detection AI training notebooks on AWS SageMaker using the powerful ml.p3.16xlarge instance type.

## Overview

The process consists of four main steps:
1. Upload notebooks to an S3 bucket
2. Launch a SageMaker notebook instance with ml.p3.16xlarge
3. Download notebooks from S3 to the SageMaker instance
4. Configure the environment and run the notebooks

## Prerequisites

- AWS account with appropriate permissions
- AWS CLI installed and configured with your credentials
- Python 3.6+ with boto3 and tqdm installed
- Service quota for ml.p3.16xlarge instances (may require a quota increase request)

## Step 1: Upload Notebooks to S3

We've created a script `upload_notebooks_to_s3.py` that will upload all the Fire Detection AI training notebooks to your S3 bucket.

### 1.1 Install Required Packages

```bash
pip install boto3 tqdm
```

### 1.2 Configure AWS Credentials

If you haven't already configured your AWS credentials, run:

```bash
aws configure
```

Enter your AWS Access Key ID, Secret Access Key, default region, and output format when prompted.

### 1.3 Run the Upload Script

```bash
python upload_notebooks_to_s3.py --bucket your-bucket-name --prefix fire-detection-notebooks
```

Replace `your-bucket-name` with your actual S3 bucket name. If the bucket doesn't exist, the script will attempt to create it.

The script will upload:
- All fire detection training notebooks (simplified and parts 1-6)
- Supporting configuration files
- Requirements file

## Step 2: Launch a SageMaker Notebook Instance

### 2.1 Via AWS Console

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
   - Lifecycle configuration: Optional, create if you need custom setup
5. Click "Create notebook instance"

### 2.2 Via AWS CLI

```bash
aws sagemaker create-notebook-instance \
    --notebook-instance-name "fire-detection-training" \
    --instance-type "ml.p3.16xlarge" \
    --role-arn "arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE" \
    --volume-size-in-gb 100 \
    --platform-identifier "notebook-al2-v2" \
    --root-access "Enabled"
```

Replace `YOUR_ACCOUNT_ID` and `YOUR_SAGEMAKER_ROLE` with your actual values.

### 2.3 Wait for the Instance to Start

The instance will take a few minutes to start. You can check the status in the AWS Console or using the AWS CLI:

```bash
aws sagemaker describe-notebook-instance \
    --notebook-instance-name "fire-detection-training" \
    --query 'NotebookInstanceStatus'
```

Wait until the status is `InService`.

## Step 3: Download Notebooks to SageMaker Instance

Once your SageMaker instance is running, you can connect to it and download the notebooks.

### 3.1 Connect to JupyterLab

1. In the AWS Console, go to the SageMaker Notebook instances page
2. Click "Open JupyterLab" next to your running instance

### 3.2 Upload the Download Script

1. In JupyterLab, click the upload button (up arrow) in the file browser
2. Upload the `download_notebooks_from_s3.py` script
3. Alternatively, you can create a new file and copy-paste the script content

### 3.3 Open a Terminal and Run the Download Script

1. In JupyterLab, click File → New → Terminal
2. In the terminal, run:

```bash
python download_notebooks_from_s3.py --bucket your-bucket-name --prefix fire-detection-notebooks --output-dir fire_detection
```

Replace `your-bucket-name` with your actual S3 bucket name.

## Step 4: Configure the Environment and Run the Notebooks

### 4.1 Create a Conda Environment

In the JupyterLab terminal:

```bash
# Create a conda environment for your project
conda create -n fire-detection python=3.8 -y
conda activate fire-detection

# Install required packages
pip install -r fire_detection/supporting_files/requirements_gpu.txt

# Register the conda environment as a kernel
python -m ipykernel install --user --name fire-detection --display-name "Fire Detection"
```

### 4.2 Configure for Multi-GPU Training

Create a new file called `gpu_setup.py` in your fire_detection directory:

```python
import torch
import os

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Set environment variables for optimal performance
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  # All 8 GPUs on p3.16xlarge
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_P2P_DISABLE'] = '1'  # May help with some multi-GPU issues
```

Run this script to verify GPU setup:

```bash
python fire_detection/gpu_setup.py
```

### 4.3 Open and Run the Notebooks

1. Navigate to your `fire_detection` directory in JupyterLab
2. Open the simplified notebook or any of the part notebooks
3. Select the "Fire Detection" kernel from the kernel dropdown menu
4. Run the notebook cells

For the simplified notebook:
- Open `fire_detection_5m_training_simplified.ipynb`
- Add the multi-GPU code from Step 4.2 at the beginning
- Run all cells

For the multi-part approach:
- Start with `fire_detection_5m_training_part1.ipynb`
- Add the multi-GPU code from Step 4.2 at the beginning
- Run through each notebook in sequence (parts 1-6)

## Cost Management

The ml.p3.16xlarge is expensive (~$24.48 per hour). To manage costs:

1. Add this code at the end of your notebook to stop the instance when done:

```python
import boto3

def stop_notebook_instance(instance_name):
    client = boto3.client('sagemaker')
    client.stop_notebook_instance(NotebookInstanceName=instance_name)
    print(f"Stopping notebook instance: {instance_name}")

# Replace with your instance name
stop_notebook_instance('fire-detection-training')
```

2. Set up CloudWatch alarms to notify you if the instance runs for too long
3. Always stop the instance when not in use

## Troubleshooting

### Instance Limit Exceeded
If you get a limit exceeded error, request a quota increase for ml.p3.16xlarge instances through the AWS Service Quotas console.

### Out of Memory Errors
Adjust batch size or use gradient accumulation:

```python
# Add to your training loop
accumulation_steps = 4  # Accumulate gradients over 4 batches
for i, batch in enumerate(train_loader):
    outputs = model(batch)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Slow Data Loading
Use SageMaker's pipe mode or optimize your data pipeline:

```python
# Use multiple workers for data loading
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
```

### Package Installation Failures
If you encounter package installation issues, try:

```bash
# Update pip
pip install --upgrade pip

# Install packages one by one
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Next Steps

After training, you can:

1. Save your models to S3
2. Deploy models using SageMaker endpoints
3. Set up monitoring and evaluation pipelines

For production training, consider converting your notebook to a SageMaker Training Job for better resource management and scalability.