# Fire Detection AI Training Guide

This repository contains code for training a Fire Detection AI model using a 5M sample dataset on AWS SageMaker with an ml.p3.16xlarge instance (8 NVIDIA V100 GPUs). The training process has been optimized to reduce training time from 43 hours to 2-4 hours while maintaining reasonable accuracy.

## Files Overview

### Main Files

1. **`fire_detection_5m_compact_final.py`**: Complete standalone Python script for training the model. This is the recommended file to use for training on SageMaker.

2. **`fire_detection_combined_part_A.py`**, **`fire_detection_combined_part_B.py`**, **`fire_detection_combined_part_C.py`**: These files contain the complete notebook code split into three parts due to file size limitations. You can copy and paste these parts sequentially into a single notebook.

3. **`install_dependencies.py`**: Script to install required dependencies for the training process.

### Support Files

1. **`dependency_installation_cell.txt`**: Contains the dependency installation commands that can be copied into a notebook cell.

2. **`manual_sagemaker_download_guide.md`**: Guide for manually downloading data from S3 to SageMaker.

## Setup Instructions

### Option 1: Using the Compact Python Script (Recommended)

1. Create a new SageMaker notebook instance with an ml.p3.16xlarge instance type.

2. Upload `fire_detection_5m_compact_final.py` to the notebook instance.

3. Run the script:
   ```bash
   python fire_detection_5m_compact_final.py
   ```

4. The trained model will be saved to the `models` directory.

### Option 2: Using the Notebook Parts

1. Create a new SageMaker notebook instance with an ml.p3.16xlarge instance type.

2. Create a new notebook and copy the contents of the following files in sequence:
   - `fire_detection_combined_part_A.py`
   - `fire_detection_combined_part_B.py`
   - `fire_detection_combined_part_C.py`

3. Run the notebook cells sequentially.

## Dependencies

The following dependencies are required:

- PyTorch with CUDA support
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- boto3
- sagemaker
- numexpr>=2.8.4

You can install these dependencies using the `install_dependencies.py` script or by running the commands in `dependency_installation_cell.txt`.

## Data Loading

The script supports three methods of data loading:

1. **Loading from S3**: The script will attempt to load data from the S3 bucket "synthetic-data-4" in the "datasets/" prefix.

2. **Loading from saved numpy files**: If numpy files (`X_train.npy`, `y_train.npy`, etc.) exist in the current directory, the script will load data from these files.

3. **Creating synthetic data**: If neither S3 data nor numpy files are available, the script will create synthetic data for demonstration purposes.

## Model Architecture

The model uses a transformer architecture with the following components:

- Input projection layer
- Positional encoding
- Transformer encoder layers
- Global average pooling
- Output projection layers

The model is configured with the following parameters:

- d_model: 128
- num_heads: 4
- num_layers: 3
- dropout: 0.1

## Training Configuration

The training process is configured with the following parameters:

- Epochs: 50
- Batch size: 256
- Early stopping patience: 5
- Learning rate: 0.002
- Optimizer: AdamW with weight decay 0.01
- Learning rate scheduler: ReduceLROnPlateau

## Multi-GPU Training

The script is optimized for multi-GPU training on an ml.p3.16xlarge instance with 8 NVIDIA V100 GPUs. It uses PyTorch's DataParallel to distribute the training across all available GPUs.

## Monitoring Training Progress

The training progress is logged to the console and to a log file in the `logs` directory. The log includes information about:

- Epoch progress
- Training and validation loss
- Training and validation accuracy
- Validation precision, recall, and F1 score
- Learning rate

## Model Saving

The trained model is saved to the `models` directory. The saved model includes:

- Model state dict
- Model architecture parameters
- Class names

If AWS is available, the model is also uploaded to the S3 bucket.

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size
   - Reduce model size
   - Use gradient accumulation

2. **Slow Training**:
   - Ensure you're using the correct instance type (ml.p3.16xlarge)
   - Check that DataParallel is working correctly
   - Increase batch size if memory allows

3. **Data Loading Errors**:
   - Check S3 bucket permissions
   - Verify dataset path and format
   - Try using the synthetic data generation as a fallback

## Performance Expectations

With the 5M sample dataset on an ml.p3.16xlarge instance, the training process should complete in approximately 2-4 hours, depending on the exact configuration and data characteristics.

## Contact

For any issues or questions, please contact the AI team.