# Fire Detection AI - 5M Dataset Training (Combined Notebook)
# PART C: Evaluation, Visualization, and Model Saving

# This is part C of the combined notebook. Copy and paste all parts in sequence into your SageMaker notebook.

# ===== CELL 17: Evaluate Model on Test Set =====
# Evaluate model on test set
test_loss, test_acc, test_precision, test_recall, test_f1, test_targets, test_predictions = evaluate(
    model, test_loader, criterion, device
)

# Print test results
logger.info(f"Test Results:")
logger.info(f"  Loss: {test_loss:.4f}")
logger.info(f"  Accuracy: {test_acc:.4f}")
logger.info(f"  Precision: {test_precision:.4f}")
logger.info(f"  Recall: {test_recall:.4f}")
logger.info(f"  F1 Score: {test_f1:.4f}")

# Print classification report
class_names = ['Normal', 'Warning', 'Fire']
print("\nClassification Report:")
print(classification_report(test_targets, test_predictions, target_names=class_names))

# ===== CELL 18: Visualize Confusion Matrix =====
def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap (white to blue)
    cmap = LinearSegmentedColormap.from_list('blue_cmap', ['#FFFFFF', '#0068C9'], N=256)
    
    # Plot confusion matrix
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    # Add counts to cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j + 0.5, 
                i + 0.7, 
                f'({cm[i, j]})',
                ha='center', 
                va='center', 
                color='gray',
                fontsize=9
            )
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save figure if enabled
    if VISUALIZATION_CONFIG['save_figures']:
        plt.savefig(f"{VISUALIZATION_CONFIG['figure_dir']}/confusion_matrix.png", dpi=300)
    
    plt.show()

# Plot confusion matrix
plot_confusion_matrix(test_targets, test_predictions, class_names)

# ===== CELL 19: Visualize Feature Importance =====
def visualize_feature_importance(model, input_dim, class_names):
    """Visualize feature importance using gradient-based attribution"""
    
    # Create sample input
    sample_input = torch.randn(1, X_train.shape[1], input_dim).to(device)
    sample_input.requires_grad = True
    
    # Forward pass
    model.eval()
    output = model(sample_input)
    
    # Calculate gradients for each class
    feature_importance = []
    
    for c in range(len(class_names)):
        # Zero gradients
        if sample_input.grad is not None:
            sample_input.grad.zero_()
        
        # Backward pass for class c
        output[0, c].backward(retain_graph=True)
        
        # Get gradients
        gradients = sample_input.grad.abs().mean(dim=1).cpu().detach().numpy()
        feature_importance.append(gradients)
    
    # Stack feature importance for all classes
    feature_importance = np.stack(feature_importance)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot feature importance for each class
    for i, class_name in enumerate(class_names):
        plt.subplot(len(class_names), 1, i+1)
        plt.plot(feature_importance[i][0])
        plt.title(f'Feature Importance for {class_name}')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save figure if enabled
    if VISUALIZATION_CONFIG['save_figures']:
        plt.savefig(f"{VISUALIZATION_CONFIG['figure_dir']}/feature_importance.png", dpi=300)
    
    plt.show()

# Visualize feature importance
try:
    visualize_feature_importance(model, input_dim, class_names)
except Exception as e:
    logger.warning(f"Could not visualize feature importance: {e}")

# ===== CELL 20: Visualize Model Predictions =====
def visualize_predictions(model, dataloader, class_names, num_samples=5):
    """Visualize model predictions on random samples"""
    
    # Get random samples
    all_inputs = []
    all_targets = []
    
    for inputs, targets in dataloader:
        all_inputs.append(inputs)
        all_targets.append(targets)
        
        if len(all_inputs) >= num_samples:
            break
    
    # Concatenate samples
    inputs = torch.cat(all_inputs, dim=0)[:num_samples]
    targets = torch.cat(all_targets, dim=0)[:num_samples]
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        _, predictions = outputs.max(1)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    # Plot each sample
    for i in range(num_samples):
        ax = axes[i]
        
        # Get data
        input_data = inputs[i].cpu().numpy()
        target = targets[i].item()
        prediction = predictions[i].item()
        probs = probabilities[i].cpu().numpy()
        
        # Plot input sequence
        ax.plot(input_data.mean(axis=1), label='Input Sequence')
        
        # Add prediction information
        title = f"True: {class_names[target]} | Predicted: {class_names[prediction]}"
        if target == prediction:
            title += " ✓"
        else:
            title += " ✗"
        
        ax.set_title(title)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True)
        
        # Add probability bar chart as inset
        ax_inset = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
        ax_inset.bar(range(len(class_names)), probs, tick_label=class_names)
        ax_inset.set_title('Class Probabilities')
        ax_inset.set_ylim(0, 1)
        
        # Highlight correct class
        ax_inset.get_children()[target].set_facecolor('green')
        
        # Highlight predicted class if different
        if target != prediction:
            ax_inset.get_children()[prediction].set_facecolor('red')
    
    plt.tight_layout()
    
    # Save figure if enabled
    if VISUALIZATION_CONFIG['save_figures']:
        plt.savefig(f"{VISUALIZATION_CONFIG['figure_dir']}/model_predictions.png", dpi=300)
    
    plt.show()

# Visualize predictions
visualize_predictions(model, test_loader, class_names)

# ===== CELL 21: Save Model for Deployment =====
def save_model_for_deployment(model, input_dim, class_names, save_dir='models'):
    """Save model for deployment"""
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model architecture and weights
    model_path = os.path.join(save_dir, 'fire_detection_model.pt')
    
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    torch.save({
        'model_state_dict': model_state_dict,
        'input_dim': input_dim,
        'd_model': TRANSFORMER_CONFIG['d_model'],
        'num_heads': TRANSFORMER_CONFIG['num_heads'],
        'num_layers': TRANSFORMER_CONFIG['num_layers'],
        'num_classes': len(class_names),
        'dropout': TRANSFORMER_CONFIG['dropout'],
        'class_names': class_names
    }, model_path)
    
    logger.info(f"✅ Model saved to {model_path}")
    
    # Save model to S3 if available
    if AWS_AVAILABLE:
        try:
            # Create S3 client
            s3_client = boto3.client('s3')
            
            # Upload model to S3
            s3_key = f"models/fire_detection_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            s3_client.upload_file(model_path, DATASET_BUCKET, s3_key)
            
            logger.info(f"✅ Model uploaded to s3://{DATASET_BUCKET}/{s3_key}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to upload model to S3: {e}")
    
    # Create model loading script
    load_script = """
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class FireDetectionTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        x = self.output_projection(x)
        return x

def load_fire_detection_model(model_path):
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Create model
    model = FireDetectionTransformer(
        input_dim=checkpoint['input_dim'],
        d_model=checkpoint['d_model'],
        num_heads=checkpoint['num_heads'],
        num_layers=checkpoint['num_layers'],
        num_classes=checkpoint['num_classes'],
        dropout=checkpoint['dropout']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model, checkpoint['class_names']
"""
    
    # Save loading script
    load_script_path = os.path.join(save_dir, 'load_model.py')
    with open(load_script_path, 'w') as f:
        f.write(load_script)
    
    logger.info(f"✅ Model loading script saved to {load_script_path}")
    
    # Create inference script
    inference_script = """
import torch
import torch.nn.functional as F
import numpy as np
from load_model import load_fire_detection_model

def preprocess_input(input_data, seq_length=60):
    # Ensure input is numpy array
    if isinstance(input_data, list):
        input_data = np.array(input_data)
    
    # Reshape if needed
    if len(input_data.shape) == 1:
        input_data = input_data.reshape(1, -1)
    
    # Ensure we have the right sequence length
    if input_data.shape[0] < seq_length:
        # Pad with zeros
        padding = np.zeros((seq_length - input_data.shape[0], input_data.shape[1]))
        input_data = np.vstack([input_data, padding])
    elif input_data.shape[0] > seq_length:
        # Use the most recent data
        input_data = input_data[-seq_length:]
    
    # Add batch dimension and convert to tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    
    return input_tensor

def predict(model, input_data, class_names):
    # Preprocess input
    input_tensor = preprocess_input(input_data)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
    
    # Get class name and probability
    class_name = class_names[prediction]
    probability = probabilities[0, prediction].item()
    
    # Get all probabilities
    all_probs = {class_names[i]: probabilities[0, i].item() for i in range(len(class_names))}
    
    return {
        'prediction': class_name,
        'probability': probability,
        'probabilities': all_probs
    }

if __name__ == '__main__':
    # Load model
    model, class_names = load_fire_detection_model('fire_detection_model.pt')
    
    # Example input (replace with your data)
    example_input = np.random.randn(60, 6)
    
    # Make prediction
    result = predict(model, example_input, class_names)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
    print("All probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")
"""
    
    # Save inference script
    inference_script_path = os.path.join(save_dir, 'inference.py')
    with open(inference_script_path, 'w') as f:
        f.write(inference_script)
    
    logger.info(f"✅ Inference script saved to {inference_script_path}")
    
    return model_path

# Save model for deployment
model_path = save_model_for_deployment(model, input_dim, class_names)

# ===== CELL 22: Create SageMaker Deployment Script =====
def create_sagemaker_deployment_script(model_path, save_dir='models'):
    """Create script for deploying model to SageMaker"""
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create deployment script
    deployment_script = """
import os
import torch
import numpy as np
import json
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model loading function
sys.path.append(os.path.dirname(__file__))
from load_model import load_fire_detection_model

# Global variables
model = None
class_names = None

def model_fn(model_dir):
    """Load model from disk"""
    global model, class_names
    
    logger.info("Loading model...")
    
    # Find model file
    model_path = os.path.join(model_dir, 'fire_detection_model.pt')
    if not os.path.exists(model_path):
        # Look for any .pt file
        pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        if pt_files:
            model_path = os.path.join(model_dir, pt_files[0])
        else:
            raise FileNotFoundError(f"No model file found in {model_dir}")
    
    # Load model
    model, class_names = load_fire_detection_model(model_path)
    
    logger.info(f"Model loaded successfully with {len(class_names)} classes: {class_names}")
    
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    logger.info(f"Received request with content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        # Parse JSON input
        input_data = json.loads(request_body)
        
        # Convert to numpy array
        if isinstance(input_data, list):
            # Check if it's a list of lists (sequence of features)
            if isinstance(input_data[0], list):
                input_array = np.array(input_data, dtype=np.float32)
            else:
                # Single feature vector, reshape to sequence of length 1
                input_array = np.array([input_data], dtype=np.float32)
        else:
            raise ValueError("Input must be a list of feature vectors or a single feature vector")
        
        logger.info(f"Parsed input shape: {input_array.shape}")
        
        return input_array
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction"""
    logger.info(f"Making prediction for input shape: {input_data.shape}")
    
    # Preprocess input
    if len(input_data.shape) == 2:
        # Add batch dimension
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    else:
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
    
    # Get class name and probability
    class_name = class_names[prediction]
    probability = probabilities[0, prediction].item()
    
    # Get all probabilities
    all_probs = {class_names[i]: probabilities[0, i].item() for i in range(len(class_names))}
    
    logger.info(f"Prediction: {class_name} with probability {probability:.4f}")
    
    return {
        'prediction': class_name,
        'probability': probability,
        'probabilities': all_probs
    }

def output_fn(prediction, response_content_type):
    """Format prediction output"""
    logger.info(f"Formatting output with content type: {response_content_type}")
    
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
"""
    
    # Save deployment script
    deployment_script_path = os.path.join(save_dir, 'sagemaker_deploy.py')
    with open(deployment_script_path, 'w') as f:
        f.write(deployment_script)
    
    logger.info(f"✅ SageMaker deployment script saved to {deployment_script_path}")
    
    # Create SageMaker deployment notebook
    deployment_notebook = """
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Deploy Fire Detection Model to SageMaker\n",
    "\n",
    "This notebook demonstrates how to deploy the trained fire detection model to SageMaker for real-time inference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import boto3\n",
    "import sagemaker\n",
    "from sagemaker.pytorch import PyTorchModel\n",
    "from sagemaker import get_execution_role\n",
    "\n",
    "# Initialize SageMaker session\n",
    "sagemaker_session = sagemaker.Session()\n",
    "role = get_execution_role()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Package Model for Deployment\n",
    "\n",
    "First, we need to package the model and deployment scripts into a tarball and upload to S3."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import os\n",
    "import tarfile\n",
    "\n",
    "# Create model directory\n",
    "model_dir = 'model_package'\n",
    "os.makedirs(model_dir, exist_ok=True)\n",
    "\n",
    "# Copy model files\n",
    "!cp models/fire_detection_model.pt {model_dir}/\n",
    "!cp models/load_model.py {model_dir}/\n",
    "!cp models/sagemaker_deploy.py {model_dir}/inference.py\n",
    "\n",
    "# Create tarball\n",
    "tarball_path = 'fire_detection_model.tar.gz'\n",
    "with tarfile.open(tarball_path, 'w:gz') as tar:\n",
    "    tar.add(model_dir, arcname='')\n",
    "\n",
    "# Upload to S3\n",
    "model_data = sagemaker_session.upload_data(tarball_path, key_prefix='models')\n",
    "print(f\"Model uploaded to: {model_data}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Create SageMaker Model\n",
    "\n",
    "Now we can create a SageMaker model using the uploaded model package."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Create PyTorch model\n",
    "pytorch_model = PyTorchModel(\n",
    "    model_data=model_data,\n",
    "    role=role,\n",
    "    entry_point='inference.py',\n",
    "    framework_version='1.13.1',\n",
    "    py_version='py39',\n",
    "    name='fire-detection-model'\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Deploy Model to Endpoint\n",
    "\n",
    "Now we can deploy the model to a SageMaker endpoint for real-time inference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Deploy model to endpoint\n",
    "predictor = pytorch_model.deploy(\n",
    "    initial_instance_count=1,\n",
    "    instance_type='ml.m5.xlarge',\n",
    "    endpoint_name='fire-detection-endpoint'\n",
    ")\n",
    "\n",
    "print(f\"Model deployed to endpoint: {predictor.endpoint_name}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Test Endpoint\n",
    "\n",
    "Let's test the endpoint with some sample data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import numpy as np\n",
    "import json\n",
    "\n",
    "# Create sample input (random data)\n",
    "sample_input = np.random.randn(60, 6).tolist()\n",
    "\n",
    "# Make prediction\n",
    "response = predictor.predict(\n",
    "    json.dumps(sample_input),\n",
    "    initial_args={'ContentType': 'application/json'}\n",
    ")\n",
    "\n",
    "# Parse response\n",
    "result = json.loads(response)\n",
    "print(f\"Prediction: {result['prediction']}\")\n",
    "print(f\"Probability: {result['probability']:.4f}\")\n",
    "print(\"All probabilities:\")\n",
    "for class_name, prob in result['probabilities'].items():\n",
    "    print(f\"  {class_name}: {prob:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Clean Up\n",
    "\n",
    "When you're done, you can delete the endpoint to avoid incurring charges."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Delete endpoint\n",
    "predictor.delete_endpoint()\n",
    "print(\"Endpoint deleted\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
"""
    
    # Save deployment notebook
    deployment_notebook_path = os.path.join(save_dir, 'sagemaker_deployment.ipynb')
    with open(deployment_notebook_path, 'w') as f:
        f.write(deployment_notebook)
    
    logger.info(f"✅ SageMaker deployment notebook saved to {deployment_notebook_path}")
    
    return deployment_script_path

# Create SageMaker deployment script
deployment_script_path = create_sagemaker_deployment_script(model_path)

# ===== CELL 23: Create Instructions for Running on SageMaker =====
def create_sagemaker_instructions():
    """Create instructions for running on SageMaker"""
    
    instructions = """
# Running Fire Detection AI Training on SageMaker

This document provides instructions for running the Fire Detection AI training notebook on AWS SageMaker.

## Prerequisites

1. AWS account with SageMaker access
2. S3 bucket with the dataset
3. IAM role with SageMaker and S3 access

## Steps

### 1. Create a SageMaker Notebook Instance

1. Go to the SageMaker console
2. Click on "Notebook instances" in the left sidebar
3. Click "Create notebook instance"
4. Configure the notebook instance:
   - Name: `fire-detection-training`
   - Instance type: `ml.p3.16xlarge` (for multi-GPU training)
   - Volume size: At least 100 GB
   - IAM role: Select a role with SageMaker and S3 access
5. Click "Create notebook instance"

### 2. Upload the Notebook

1. Wait for the notebook instance to be "InService"
2. Click "Open JupyterLab"
3. Create a new notebook with the PyTorch kernel
4. Copy and paste the contents of all three parts of the Fire Detection AI training notebook into the new notebook

### 3. Configure the Notebook

1. Update the `DATASET_BUCKET` and `DATASET_PREFIX` variables to point to your S3 bucket and dataset location
2. Adjust other configuration parameters as needed:
   - `SAMPLE_SIZE`: Number of samples to use (default: 5M)
   - `EPOCHS`: Number of training epochs (default: 50)
   - `BATCH_SIZE`: Batch size for training (default: 256)
   - `LEARNING_RATE`: Learning rate for optimizer (default: 0.002)

### 4. Run the Notebook

1. Run all cells in the notebook
2. Monitor the training progress
3. The trained model will be saved to the `models` directory and optionally uploaded to S3

### 5. Deploy the Model (Optional)

1. Use the provided SageMaker deployment notebook to deploy the model to a SageMaker endpoint
2. Follow the instructions in the deployment notebook

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce `BATCH_SIZE`
   - Reduce model size by adjusting `TRANSFORMER_CONFIG`
   - Use gradient accumulation

2. **Slow Training**:
   - Ensure you're using the correct instance type (`ml.p3.16xlarge`)
   - Check that DataParallel is working correctly
   - Increase `BATCH_SIZE` if memory allows

3. **Data Loading Errors**:
   - Check S3 bucket permissions
   - Verify dataset path and format
   - Try using the synthetic data generation as a fallback

### Getting Help

If you encounter issues, check the logs in the `logs` directory or contact the AI team for assistance.
"""
    
    # Save instructions
    instructions_path = 'sagemaker_instructions.md'
    with open(instructions_path, 