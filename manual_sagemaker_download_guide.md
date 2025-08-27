# Manual Download of Fire Detection AI Notebooks from S3 to SageMaker

This guide provides step-by-step instructions for manually downloading the Fire Detection AI training notebooks from AWS S3 to your SageMaker notebook instance with ml.p3.16xlarge.

## Prerequisites

- AWS account with appropriate permissions
- S3 bucket with the Fire Detection AI notebooks uploaded
- SageMaker notebook instance with ml.p3.16xlarge running

## Step 1: Launch Your SageMaker Notebook Instance

### Via AWS Console

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
6. Wait for the instance status to change to "InService"
7. Click "Open JupyterLab"

## Step 2: Download Notebooks Using the AWS CLI

### Option 1: Using the Terminal

1. In JupyterLab, click File → New → Terminal
2. In the terminal, create a directory for your notebooks:

```bash
mkdir -p fire_detection
```

3. Use the AWS CLI to download your notebooks:

```bash
aws s3 cp s3://your-bucket-name/fire-detection-notebooks/ fire_detection/ --recursive
```

Replace `your-bucket-name` with your actual bucket name.

4. Verify the download:

```bash
ls -la fire_detection/
```

### Option 2: Using the JupyterLab Interface

1. In JupyterLab, click on the "+" button in the file browser to create a new launcher
2. Click on "Terminal"
3. Follow the same commands as in Option 1

## Step 3: Set Up the Environment

1. In the terminal, create a conda environment:

```bash
# Create a conda environment for your project
conda create -n fire-detection python=3.8 -y
conda activate fire-detection

# Install required packages
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm
pip install boto3 sagemaker

# Register the conda environment as a kernel
python -m ipykernel install --user --name fire-detection --display-name "Fire Detection"
```

## Step 4: Create GPU Setup Script

1. In JupyterLab, click File → New → Text File
2. Rename the file to `gpu_setup.py`
3. Copy and paste the following code:

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

print("\nGPU setup complete. You can now run your training notebooks.")
print("Remember to use torch.nn.DataParallel(model) for multi-GPU training.")
```

4. Save the file

## Step 5: Run the GPU Setup Script

1. In the terminal:

```bash
python gpu_setup.py
```

2. Verify that all 8 GPUs are detected

## Step 6: Open and Run the Notebooks

1. Navigate to your `fire_detection` directory in JupyterLab
2. Open the simplified notebook or any of the part notebooks
3. Select the "Fire Detection" kernel from the kernel dropdown menu
4. Run the notebook cells

For the simplified notebook:
- Open `fire_detection_5m_training_simplified.ipynb`
- Add the multi-GPU code from Step 4 at the beginning
- Run all cells

For the multi-part approach:
- Start with `fire_detection_5m_training_part1.ipynb`
- Add the multi-GPU code from Step 4 at the beginning
- Run through each notebook in sequence (parts 1-6)

## Step 7: Cost Management

Add this code at the end of your notebook to stop the instance when done:

```python
import boto3

def stop_notebook_instance(instance_name):
    client = boto3.client('sagemaker')
    client.stop_notebook_instance(NotebookInstanceName=instance_name)
    print(f"Stopping notebook instance: {instance_name}")

# Replace with your instance name
stop_notebook_instance('fire-detection-training')
```

## Troubleshooting

### Access Issues

If you encounter permission issues:

1. Check that your SageMaker role has access to your S3 bucket
2. In the AWS Console, go to IAM → Roles
3. Find your SageMaker execution role
4. Add the necessary S3 permissions if missing

### Package Installation Issues

If you encounter package installation issues:

```bash
# Update pip
pip install --upgrade pip

# Install packages one by one
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### GPU Issues

If not all GPUs are detected:

```bash
# Check NVIDIA driver status
nvidia-smi

# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues

If you encounter out-of-memory errors:

1. Reduce batch size in the notebooks
2. Implement gradient accumulation
3. Use mixed precision training (FP16)