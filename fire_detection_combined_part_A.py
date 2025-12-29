# Fire Detection AI - 5M Dataset Training (Combined Notebook)
# PART A: Setup, Configuration, and Data Loading

# This is part A of the combined notebook. Copy and paste all parts in sequence into your SageMaker notebook.

# ===== CELL 1: Install Required Packages =====
# Run this cell to install the required dependencies
# This may take a few minutes to complete

# Install PyTorch with CUDA support
!pip install torch torchvision torchaudio

# Install other required packages
!pip install pandas numpy scikit-learn matplotlib seaborn
!pip install "numexpr>=2.8.4"  # Update numexpr to required version
!pip install xgboost lightgbm
!pip install boto3 sagemaker

print("\nAfter running this cell, restart the kernel (Kernel > Restart) and then continue with the notebook.")

# ===== CELL 2: Import Libraries =====
# Standard libraries
import numpy as np
import pandas as pd
import json
import os
import time
import logging
import sys
import traceback
import functools
import threading
import gc
import pickle
from datetime import datetime
from IPython import display

# Machine learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# AWS libraries
try:
    import boto3
    import sagemaker
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
    print("✅ AWS libraries available")
except ImportError:
    AWS_AVAILABLE = False
    print("❌ AWS libraries not available - install with: pip install boto3 sagemaker")

# Optional ML libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGB_AVAILABLE = False
    print("❌ XGBoost not available - install with: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    print("✅ LightGBM available")
except ImportError:
    LGB_AVAILABLE = False
    print("❌ LightGBM not available - install with: pip install lightgbm")

# Configure notebook settings
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# ===== CELL 3: GPU Setup for ml.p3.16xlarge =====
# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Set environment variables for optimal performance
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  # All 8 GPUs on p3.16xlarge
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_P2P_DISABLE'] = '1'  # May help with some multi-GPU issues

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Function to monitor GPU memory
def print_gpu_memory():
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    else:
        print("CUDA not available")

# Run nvidia-smi for detailed GPU info
!nvidia-smi

# ===== CELL 4: Configuration Parameters =====
# Dataset configuration
DATASET_BUCKET = "synthetic-data-4"
DATASET_PREFIX = "datasets/"
SAMPLE_SIZE = 5000000  # 5M samples
RANDOM_SEED = 42

# Sampling configuration
PRESERVE_TEMPORAL_PATTERNS = True
ENSURE_CLASS_BALANCE = True
CLASS_DISTRIBUTION_TARGET = {
    0: 0.70,  # Normal: 70%
    1: 0.20,  # Warning: 20%
    2: 0.10   # Fire: 10%
}

# Training configuration
EPOCHS = 50  # Reduced from 100
BATCH_SIZE = 256
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 0.002

# Model configuration
TRANSFORMER_CONFIG = {
    'd_model': 128,  # Reduced from 256
    'num_heads': 4,  # Reduced from 8
    'num_layers': 3,  # Reduced from 6
    'dropout': 0.1
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'update_interval': 1,  # Update visualizations every N epochs
    'save_figures': True,
    'figure_dir': 'figures',
    'create_animations': True
}

# Create directories
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs(VISUALIZATION_CONFIG['figure_dir'], exist_ok=True)

# ===== CELL 5: Logging Configuration =====
def setup_logging():
    """Configure comprehensive logging for the training process"""
    
    # Generate timestamp for log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/fire_detection_5m_training_{timestamp}.log'
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    logger = logging.getLogger()
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Create S3 handler if AWS is available
    if AWS_AVAILABLE:
        try:
            s3_handler = S3LogHandler(
                bucket=DATASET_BUCKET,
                key_prefix=f'logs/fire_detection_5m_training_{timestamp}/'
            )
            s3_handler.setLevel(logging.INFO)
            s3_handler.setFormatter(file_formatter)
            logger.addHandler(s3_handler)
            logger.info("✅ S3 logging configured")
        except Exception as e:
            logger.warning(f"⚠️ Failed to configure S3 logging: {e}")
    
    logger.info("🔧 Logging configured successfully")
    logger.info(f"📝 Log file: {log_file}")
    
    return logger

class S3LogHandler(logging.Handler):
    """Custom logging handler that writes logs to S3"""
    
    def __init__(self, bucket, key_prefix):
        super().__init__()
        self.bucket = bucket
        self.key_prefix = key_prefix
        self.buffer = []
        self.buffer_size = 100  # Number of logs to buffer before writing to S3
        self.s3_client = boto3.client('s3')
        
        # Start background thread for uploading logs
        self.stop_event = threading.Event()
        self.upload_thread = threading.Thread(target=self._upload_logs_periodically)
        self.upload_thread.daemon = True
        self.upload_thread.start()
    
    def emit(self, record):
        """Process a log record"""
        log_entry = self.format(record)
        self.buffer.append(log_entry)
        
        # Upload logs if buffer is full
        if len(self.buffer) >= self.buffer_size:
            self._upload_logs()
    
    def _upload_logs(self):
        """Upload buffered logs to S3"""
        if not self.buffer:
            return
        
        try:
            # Create log content
            log_content = '\n'.join(self.buffer)
            
            # Generate key with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            key = f"{self.key_prefix}log_{timestamp}.txt"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=log_content
            )
            
            # Clear buffer
            self.buffer = []
            
        except Exception as e:
            # Don't raise exception, just print to stderr
            print(f"Error uploading logs to S3: {e}", file=sys.stderr)
    
    def _upload_logs_periodically(self):
        """Upload logs periodically in background thread"""
        while not self.stop_event.is_set():
            time.sleep(60)  # Upload every minute
            self._upload_logs()
    
    def close(self):
        """Clean up handler resources"""
        self.stop_event.set()
        self._upload_logs()  # Final upload
        super().close()

# Initialize logger
logger = setup_logging()

# ===== CELL 6: Error Handling Framework =====
class TrainingError(Exception):
    """Base class for training-related exceptions"""
    pass

class DataLoadingError(TrainingError):
    """Exception raised for errors during data loading"""
    pass

class ModelInitializationError(TrainingError):
    """Exception raised for errors during model initialization"""
    pass

class TrainingProcessError(TrainingError):
    """Exception raised for errors during training process"""
    pass

class EvaluationError(TrainingError):
    """Exception raised for errors during model evaluation"""
    pass

class ModelSavingError(TrainingError):
    """Exception raised for errors during model saving"""
    pass

def error_handler(func):
    """Decorator for handling errors in functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TrainingError as e:
            # Log specific training errors
            logger.error(f"❌ {e.__class__.__name__}: {str(e)}")
            # Re-raise to be caught by global handler
            raise
        except Exception as e:
            # Log unexpected errors
            logger.error(f"❌ Unexpected error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            # Wrap in TrainingProcessError
            raise TrainingProcessError(f"Error in {func.__name__}: {str(e)}") from e
    return wrapper

# ===== CELL 7: Create Synthetic Data =====
@error_handler
def create_synthetic_data(n_samples=5000000, n_features=6, n_timesteps=60, n_areas=5):
    """Create synthetic data for demonstration"""
    logger.info(f"Creating synthetic data with {n_samples} samples")
    
    # Create features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create labels (0: normal, 1: warning, 2: fire)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
    
    # Create area IDs
    areas = np.random.choice(range(n_areas), size=n_samples)
    
    logger.info(f"✅ Created synthetic data: X={X.shape}, y={y.shape}, areas={areas.shape}")
    
    return X, y, areas

# ===== CELL 8: Load Data from S3 =====
# Check if we should use saved numpy files
if os.path.exists('X_train.npy') and os.path.exists('y_train.npy'):
    logger.info("Loading data from saved numpy files...")
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    if os.path.exists('areas_train.npy'):
        areas_train = np.load('areas_train.npy')
        areas_val = np.load('areas_val.npy')
        areas_test = np.load('areas_test.npy')
    else:
        # Create dummy area values if not available
        areas_train = np.zeros(len(y_train), dtype=int)
        areas_val = np.zeros(len(y_val), dtype=int)
        areas_test = np.zeros(len(y_test), dtype=int)
    
    logger.info(f"Loaded data: X_train={X_train.shape}, y_train={y_train.shape}")

# If numpy files don't exist, try loading from S3
elif AWS_AVAILABLE:
    try:
        # List files in S3
        s3_client = boto3.client('s3')
        logger.info(f"Listing files in s3://{DATASET_BUCKET}/{DATASET_PREFIX}")
        response = s3_client.list_objects_v2(Bucket=DATASET_BUCKET, Prefix=DATASET_PREFIX)

        if 'Contents' not in response:
            logger.error(f"No files found in s3://{DATASET_BUCKET}/{DATASET_PREFIX}")
            files = []
        else:
            files = [obj['Key'] for obj in response['Contents'] if not obj['Key'].endswith('/')]
            
            # Continue listing if there are more files
            while response.get('IsTruncated', False):
                continuation_token = response.get('NextContinuationToken')
                response = s3_client.list_objects_v2(
                    Bucket=DATASET_BUCKET, 
                    Prefix=DATASET_PREFIX,
                    ContinuationToken=continuation_token
                )
                files.extend([obj['Key'] for obj in response['Contents'] if not obj['Key'].endswith('/')])

        logger.info(f"Found {len(files)} files in s3://{DATASET_BUCKET}/{DATASET_PREFIX}")

        # For demonstration, limit to first few files
        files = files[:5]  # Adjust as needed
        logger.info(f"Using {len(files)} files for sampling")

        # Calculate samples per file
        samples_per_file = SAMPLE_SIZE // len(files)
        logger.info(f"Sampling approximately {samples_per_file} rows from each file")

        # Sample data from S3
        all_data = []
        all_labels = []
        all_areas = []

        for i, file_key in enumerate(files):
            try:
                logger.info(f"Processing file {i+1}/{len(files)}: {file_key}")
                
                # Get file from S3
                response = s3_client.get_object(Bucket=DATASET_BUCKET, Key=file_key)
                
                # Extract area from filename
                area_name = file_key.split('/')[-1].split('_')[0]
                area_idx = hash(area_name) % 5  # Map to 5 areas
                
                # Read file in chunks
                chunk_size = 100000  # 100K rows per chunk
                
                # Stream data from S3 in chunks and sample
                chunk_iter = pd.read_csv(response['Body'], chunksize=chunk_size)
                
                total_rows = 0
                for chunk in chunk_iter:
                    # Sample from chunk
                    if len(chunk) > samples_per_file:
                        chunk = chunk.sample(n=samples_per_file, random_state=RANDOM_SEED)
                    
                    # Extract features and labels
                    feature_cols = [col for col in chunk.columns if col not in ['timestamp', 'label', 'is_anomaly']]
                    
                    if 'label' in chunk.columns:
                        labels = chunk['label'].values
                    elif 'is_anomaly' in chunk.columns:
                        labels = chunk['is_anomaly'].values
                    else:
                        # Generate synthetic labels
                        values = chunk[feature_cols[0]].values
                        q95 = np.percentile(values, 95)
                        q85 = np.percentile(values, 85)
                        
                        labels = np.zeros(len(values))
                        labels[values > q95] = 2  # Fire (top 5%)
                        labels[(values > q85) & (values <= q95)] = 1  # Warning (85-95%)
                    
                    # Extract features
                    features = chunk[feature_cols].values
                    
                    # Standardize to 6 features
                    if features.shape[1] < 6:
                        padding = np.zeros((features.shape[0], 6 - features.shape[1]))
                        features = np.hstack([features, padding])
                    elif features.shape[1] > 6:
                        features = features[:, :6]
                    
                    # Add to lists
                    all_data.append(features)
                    all_labels.append(labels)
                    all_areas.append(np.full(len(labels), area_idx))
                    
                    total_rows += len(chunk)
                    logger.info(f"  Sampled {total_rows} rows from {file_key}")
                    
                    # Stop if we have enough samples from this file
                    if total_rows >= samples_per_file:
                        break
            
            except Exception as e:
                logger.error(f"Error processing file {file_key}: {e}")
                continue
        
        # Combine all data
        X = np.vstack(all_data)
        y = np.concatenate(all_labels)
        areas = np.concatenate(all_areas)
        
        # Ensure we have exactly the requested sample size
        if len(X) > SAMPLE_SIZE:
            indices = np.random.choice(len(X), SAMPLE_SIZE, replace=False)
            X = X[indices]
            y = y[indices]
            areas = areas[indices]
        
        logger.info(f"Raw dataset: X={X.shape}, y={y.shape}, areas={areas.shape}")
        
        # Create sequences
        seq_len = 60
        step = 10
        logger.info(f"Creating sequences with length {seq_len} and step {step}")
        
        X_sequences = []
        y_sequences = []
        areas_sequences = []
        
        # Process each area separately
        unique_areas = np.unique(areas)
        for area_idx in unique_areas:
            area_mask = areas == area_idx
            X_area = X[area_mask]
            y_area = y[area_mask]
            
            sequences = []
            labels = []
            
            for i in range(0, len(X_area) - seq_len, step):
                sequences.append(X_area[i:i+seq_len])
                labels.append(y_area[i+seq_len-1])  # Use label of last timestep
            
            if sequences:
                X_sequences.append(np.array(sequences))
                y_sequences.append(np.array(labels))
                areas_sequences.append(np.full(len(sequences), area_idx))
                
                logger.info(f"Created {len(sequences)} sequences for area {area_idx}")
        
        # Combine sequences from all areas
        X = np.vstack(X_sequences)
        y = np.hstack(y_sequences)
        areas = np.hstack(areas_sequences)
        
        logger.info(f"Sequence dataset: X={X.shape}, y={y.shape}, areas={areas.shape}")
        
        # Split into train/validation/test sets
        X_train, X_test, y_train, y_test, areas_train, areas_test = train_test_split(
            X, y, areas, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        X_train, X_val, y_train, y_val, areas_train, areas_val = train_test_split(
            X_train, y_train, areas_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
        )
        
        # Save to numpy files for faster loading in the future
        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        np.save('areas_train.npy', areas_train)
        np.save('X_val.npy', X_val)
        np.save('y_val.npy', y_val)
        np.save('areas_val.npy', areas_val)
        np.save('X_test.npy', X_test)
        np.save('y_test.npy', y_test)
        np.save('areas_test.npy', areas_test)
        
        logger.info(f"Data saved to numpy files")
    
    except Exception as e:
        logger.error(f"Error loading data from S3: {e}")
        logger.info("Falling back to synthetic data generation")
        
        # Create synthetic data
        X, y, areas = create_synthetic_data(n_samples=50000)  # Smaller sample for demonstration
        
        # Create sequences
        seq_len = 60
        step = 10
        
        X_sequences = []
        y_sequences = []
        areas_sequences = []
        
        # Process each area separately
        unique_areas = np.unique(areas)
        for area_idx in unique_areas:
            area_mask = areas == area_idx
            X_area = X[area_mask]
            y_area = y[area_mask]
            
            # Reshape X_area to be 3D if it's 2D
            if len(X_area.shape) == 2:
                X_area = X_area.reshape(X_area.shape[0], 1, X_area.shape[1])
                
            sequences = []
            labels = []
            
            for i in range(0, len(X_area) - seq_len, step):
                sequences.append(X_area[i:i+seq_len])
                labels.append(y_area[i+seq_len-1])
            
            if sequences:
                X_sequences.append(np.array(sequences))
                y_sequences.append(np.array(labels))
                areas_sequences.append(np.full(len(sequences), area_idx))
        
        # Combine sequences from all areas
        X = np.vstack(X_sequences)
        y = np.hstack(y_sequences)
        areas = np.hstack(areas_sequences)
        
        # Split into train/validation/test sets
        X_train, X_test, y_train, y_test, areas_train, areas_test = train_test_split(
            X, y, areas, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        X_train, X_val, y_train, y_val, areas_train, areas_val = train_test_split(
            X_train, y_train, areas_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
        )
else:
    # Create synthetic data if AWS is not available
    logger.info("AWS not available. Creating synthetic data...")
    
    # Create synthetic data
    X, y, areas = create_synthetic_data(n_samples=50000)  # Smaller sample for demonstration
    
    # Create sequences
    seq_len = 60
    step = 10
    
    X_sequences = []
    y_sequences = []
    areas_sequences = []
    
    # Process each area separately
    unique_areas = np.unique(areas)
    for area_idx in unique_areas:
        area_mask = areas == area_idx
        X_area = X[area_mask]
        y_area = y[area_mask]
        
        # Reshape X_area to be 3D if it's 2D
        if len(X_area.shape) == 2:
            X_area = X_area.reshape(X_area.shape[0], 1, X_area.shape[1])
            
        sequences = []
        labels = []
        
        for i in range(0, len(X_area) - seq_len, step):
            sequences.append(X_area[i:i+seq_len])
            labels.append(y_area[i+seq_len-1])
        
        if sequences:
            X_sequences.append(np.array(sequences))
            y_sequences.append(np.array(labels))
            areas_sequences.append(np.full(len(sequences), area_idx))
    
    # Combine sequences from all areas
    X = np.vstack(X_sequences)
    y = np.hstack(y_sequences)
    areas = np.hstack(areas_sequences)
    
    # Split into train/validation/test sets
    X_train, X_test, y_train, y_test, areas_train, areas_test = train_test_split(
        X, y, areas, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    X_train, X_val, y_train, y_val, areas_train, areas_val = train_test_split(
        X_train, y_train, areas_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )

# Print data split summary
logger.info(f"📊 Data splits:")
logger.info(f"   Train: {X_train.shape}, {np.bincount(y_train.astype(int))}")
logger.info(f"   Validation: {X_val.shape}, {np.bincount(y_val.astype(int))}")
logger.info(f"   Test: {X_test.shape}, {np.bincount(y_test.astype(int))}")

# ===== CELL 9: Visualize Class Distribution =====
def visualize_class_distribution(y_train, y_val, y_test):
    """Visualize class distribution across train/val/test sets"""
    
    class_names = ['Normal', 'Warning', 'Fire']
    
    # Count classes in each set
    train_counts = np.bincount(y_train.astype(int), minlength=3)
    val_counts = np.bincount(y_val.astype(int), minlength=3)
    test_counts = np.bincount(y_test.astype(int), minlength=3)
    
    # Convert to percentages
    train_pct = train_counts / train_counts.sum() * 100
    val_pct = val_counts / val_counts.sum() * 100
    test_pct = test_counts / test_counts.sum() * 100
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Raw counts
    x = np.arange(len(class_names))
    width = 0.25
    
    ax1.bar(x - width, train_counts, width, label='Train')
    ax1.bar(x, val_counts, width, label='Validation')
    ax1.bar(x + width, test_counts, width, label='Test')
    
    ax1.set_title('Class Distribution (Counts)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names)
    ax1.set_ylabel('Number of Samples')
    ax1.legend()
    
    # Add count labels
    for i, v in enumerate(train_counts):
        ax1.text(i - width, v + 0.1, f"{v:,}", ha='center')
    for i, v in enumerate(val_counts):
        ax1.text(i, v + 0.1, f"{v:,}", ha='center')
    for i, v in enumerate(test_counts):
        ax1.text(i + width, v + 0.1, f"{v:,}", ha='center')
    
    # Percentages
    ax2.bar(x - width, train_pct, width, label='Train')
    ax2.bar(x, val_pct, width, label='Validation')
    ax2.bar(x + width, test_pct, width, label='Test')
    
    ax2.set_title('Class Distribution (Percentage)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.set_ylabel('Percentage (%)')
    ax2.legend()
    
    # Add percentage labels
    for i, v in enumerate(train_pct):
        ax2.text(i - width, v + 0.5, f"{v:.1f}%", ha='center')
    for i, v in enumerate(val_pct):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    for i, v in enumerate(test_pct):
        ax2.text(i + width, v + 0.5, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    
    # Save figure if enabled
    if VISUALIZATION_CONFIG['save_figures']:
        plt.savefig(f"{VISUALIZATION_CONFIG['figure_dir']}/class_distribution.png", dpi=300)
    
    plt.show()

# Visualize class distribution
visualize_class_distribution(y_train, y_val, y_test)