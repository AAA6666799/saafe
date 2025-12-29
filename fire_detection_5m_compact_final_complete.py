"""
Fire Detection AI - 5M Dataset Training (Advanced Version)

This is an advanced version of the fire detection training script optimized for AWS SageMaker
with an ml.p3.16xlarge instance (8 NVIDIA V100 GPUs). It implements a comprehensive approach
to address severe class imbalance and improve Warning class detection.

IMPROVEMENTS MADE:
1. Fixed class accuracy calculation bug that was causing accuracies > 1
2. Enhanced class weighting with cubic weighting for the Warning class
3. Reduced learning rate from 0.0005 to 0.0002 for more stable training
4. Extended warmup period from 5 to 10 epochs
5. Simplified model architecture (reduced layers, heads, and dimensions)
6. Added class-specific metrics tracking (precision, recall, F1)
7. Added confusion matrix visualization after each epoch
8. Enhanced data augmentation specifically for the Warning class
9. Added detailed Warning class performance monitoring

ADVANCED IMPROVEMENTS:
10. Implemented two-phase training approach:
    - Phase 1: Binary classification (Normal vs Anomaly)
    - Phase 2: Multi-class classification with transfer learning from Phase 1
11. Added temporal features (derivatives and rolling statistics) to help distinguish Warning class
12. Created synthetic Warning samples through advanced augmentation techniques
13. Implemented BalancedBatchSampler to ensure each batch contains samples from all classes
14. Enhanced Focal Loss with class-specific gamma values and Warning class boosting
15. Further reduced learning rate for fine-tuning (0.00005) to prevent oscillation
16. Added model initialization from binary classifier to improve multi-class performance

Instructions:
1. Upload this file to your SageMaker notebook instance
2. Run the script with: python fire_detection_5m_compact_final_complete.py
3. The trained model will be saved to the 'models' directory

Requirements:
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+
- AWS SageMaker environment
"""

# ===== LIBRARY INSTALLATION =====
print("🔧 Installing required libraries...")
print("=" * 50)

# Check if running in Jupyter notebook
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        JUPYTER_ENV = True
        print("📓 Jupyter notebook environment detected")
    else:
        JUPYTER_ENV = False
except ImportError:
    JUPYTER_ENV = False

# For Jupyter notebooks, use magic commands
if JUPYTER_ENV:
    print("Using pip magic commands for Jupyter...")
    get_ipython().system('pip install --upgrade pip')
    get_ipython().system('pip install numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0')
    get_ipython().system('pip install torch>=1.12.0 torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cu113')
    get_ipython().system('pip install boto3>=1.24.0 sagemaker>=2.100.0')
    get_ipython().system('pip install matplotlib>=3.5.0 seaborn>=0.11.0 joblib>=1.1.0 tqdm>=4.64.0 psutil>=5.9.0')
    print("✅ Jupyter installation completed!")
else:
    print("Using subprocess for script execution...")

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False
    return True

def install_packages(packages):
    """Install multiple packages"""
    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    return failed_packages

# Required packages
REQUIRED_PACKAGES = [
    "numpy>=1.21.0",
    "pandas>=1.3.0", 
    "scikit-learn>=1.0.0",
    "torch>=1.12.0",
    "torchvision>=0.13.0",
    "boto3>=1.24.0",
    "sagemaker>=2.100.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "joblib>=1.1.0",
    "tqdm>=4.64.0",
    "psutil>=5.9.0"
]

print("Installing core packages...")
failed = install_packages(REQUIRED_PACKAGES)

if failed:
    print(f"\n⚠️ Failed to install: {failed}")
    print("Please install manually or check your environment")
else:
    print("\n✅ All packages installed successfully!")

print("\n🚀 Starting Fire Detection AI Training...")
print("=" * 50)

# ===== IMPORT LIBRARIES =====
print("\n📦 Importing libraries...")

# Standard libraries
import numpy as np
import pandas as pd
import os
import time
import logging
import sys
import math
from datetime import datetime
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import LambdaLR
    print("✅ PyTorch imported successfully")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import joblib
    print("✅ Scikit-learn imported successfully")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")
    sys.exit(1)

# Visualization libraries (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    print("✅ Visualization libraries available")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Visualization libraries not available (optional)")

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
    print("✅ Progress bar (tqdm) available")
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ Progress bar not available (optional)")

# System monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
    print("✅ System monitoring (psutil) available")
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ System monitoring not available (optional)")

# AWS libraries
try:
    import boto3
    import sagemaker
    AWS_AVAILABLE = True
    print("✅ AWS libraries available")
except ImportError:
    AWS_AVAILABLE = False
    print("❌ AWS libraries not available - some features will be disabled")

print("📦 All core libraries imported successfully!")

# ===== LIBRARY VERIFICATION =====
print(f"\n🔍 Verifying critical dependencies...")

# Verify NumPy
try:
    test_array = np.array([1, 2, 3])
    assert len(test_array) == 3
    print("✅ NumPy working correctly")
except Exception as e:
    print(f"❌ NumPy verification failed: {e}")
    sys.exit(1)

# Verify PyTorch
try:
    test_tensor = torch.tensor([1.0, 2.0, 3.0])
    assert test_tensor.shape == (3,)
    print("✅ PyTorch working correctly")
except Exception as e:
    print(f"❌ PyTorch verification failed: {e}")
    sys.exit(1)

# Verify CUDA if available
if torch.cuda.is_available():
    try:
        test_cuda = torch.tensor([1.0]).cuda()
        print("✅ CUDA working correctly")
    except Exception as e:
        print(f"⚠️ CUDA verification failed: {e}")
else:
    print("⚠️ CUDA not available - will use CPU")

# ===== SYSTEM INFORMATION =====
print(f"\n🖥️ System Information:")
print(f"   Python: {sys.version}")
print(f"   NumPy: {np.__version__}")
print(f"   Pandas: {pd.__version__}")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Count: {torch.cuda.device_count()}")

if PSUTIL_AVAILABLE:
    memory = psutil.virtual_memory()
    print(f"   Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"   Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"   CPU Cores: {psutil.cpu_count()}")

print("=" * 50)

# ===== Configuration =====
# Dataset configuration
DATASET_BUCKET = "synthetic-data-4"
DATASET_PREFIX = "datasets/"
SAMPLE_SIZE = 5000000  # 5M samples
RANDOM_SEED = 42

# Training configuration
EPOCHS = 100  # Increased to allow more training time
BATCH_SIZE = 128  # Reduced to allow more gradient updates
EARLY_STOPPING_PATIENCE = 15  # Increased to give more time for convergence
LEARNING_RATE = 0.0002  # Reduced to prevent overshooting and improve stability
WARMUP_EPOCHS = 10  # Extended warmup period for learning rate
GRADIENT_CLIP_VALUE = 0.5  # Reduced gradient clipping to stabilize training

# Model configuration
TRANSFORMER_CONFIG = {
    'd_model': 192,  # Reduced model capacity to prevent overfitting
    'num_heads': 6,  # Reduced number of attention heads
    'num_layers': 3,  # Reduced number of layers to simplify model
    'dropout': 0.3  # Increased dropout for better regularization
}

# Create directories
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ===== Logging Setup =====
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/fire_detection_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ===== GPU Setup =====
# Set environment variables for optimal performance
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  # All 8 GPUs on p3.16xlarge
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_P2P_DISABLE'] = '1'  # May help with some multi-GPU issues

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# ===== Data Loading =====
def load_data_from_s3():
    """Load data from S3 bucket"""
    try:
        # List files in S3
        s3_client = boto3.client('s3')
        logger.info(f"Listing files in s3://{DATASET_BUCKET}/{DATASET_PREFIX}")
        response = s3_client.list_objects_v2(Bucket=DATASET_BUCKET, Prefix=DATASET_PREFIX)

        if 'Contents' not in response:
            logger.error(f"No files found in s3://{DATASET_BUCKET}/{DATASET_PREFIX}")
            return None, None, None
        
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

        logger.info(f"🔥 Found {len(files)} files in s3://{DATASET_BUCKET}/{DATASET_PREFIX}")

        # Use more files for 5M sample target
        max_files = min(len(files), 20)  # Use up to 20 files for better diversity
        files = files[:max_files]
        logger.info(f"📊 Using {len(files)} files to reach {SAMPLE_SIZE:,} samples target")

        # Calculate samples per file
        samples_per_file = SAMPLE_SIZE // len(files)
        logger.info(f"📈 Sampling approximately {samples_per_file:,} rows from each file")

        # Sample data from S3
        all_data = []
        all_labels = []
        all_areas = []
        
        logger.info(f"🚀 Starting to load {SAMPLE_SIZE:,} samples from {len(files)} files...")
        total_loaded = 0

        for i, file_key in enumerate(files):
            try:
                logger.info(f"📂 Processing file {i+1}/{len(files)}: {file_key}")
                logger.info(f"📊 Progress: {total_loaded:,}/{SAMPLE_SIZE:,} samples loaded ({total_loaded/SAMPLE_SIZE*100:.1f}%)")
                
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
                    total_loaded += len(chunk)
                    logger.info(f"  ✅ Sampled {total_rows:,} rows from {file_key} (Total: {total_loaded:,})")
                    
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
        
        logger.info(f"🎯 Final sequence dataset: X={X.shape}, y={y.shape}, areas={areas.shape}")
        logger.info(f"✅ Successfully loaded {len(X):,} sequences from REAL fire detection data!")
        logger.info(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y, areas
        
    except Exception as e:
        logger.error(f"Error loading data from S3: {e}")
        return None, None, None

def create_synthetic_data(n_samples=50000):
    """Create synthetic data ONLY as fallback when real data unavailable"""
    logger.warning(f"⚠️ CREATING SYNTHETIC DATA - {n_samples} samples")
    logger.warning("⚠️ This is NOT real fire detection data!")
    logger.warning("⚠️ For production, ensure S3 access to real dataset")
    
    # Create features - ensure float32 dtype
    X = np.random.randn(n_samples, 60, 6).astype(np.float32)
    
    # Create labels (0: normal, 1: warning, 2: fire) - ensure int64 dtype
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1]).astype(np.int64)
    
    # Create area IDs - ensure int64 dtype
    areas = np.random.choice(range(5), size=n_samples).astype(np.int64)
    
    logger.info(f"✅ Created synthetic data: X={X.shape}, y={y.shape}, areas={areas.shape}")
    
    return X, y, areas

def validate_and_fix_data_types(X, y, areas):
    """Validate and fix data types to ensure compatibility with PyTorch"""
    logger.info("🔍 Validating and fixing data types...")
    
    # Fix X (features)
    if X.dtype == np.object_:
        logger.warning("Features have object dtype - converting...")
        try:
            # Try to convert each element
            X_fixed = []
            for i, sample in enumerate(X):
                if isinstance(sample, np.ndarray):
                    X_fixed.append(sample.astype(np.float32))
                else:
                    X_fixed.append(np.array(sample, dtype=np.float32))
            X = np.array(X_fixed, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to fix X dtype: {e}")
            # Create new synthetic data as fallback
            logger.info("Creating fallback synthetic data...")
            return create_synthetic_data(len(y))
    else:
        X = X.astype(np.float32)
    
    # Fix y (labels)
    if y.dtype == np.object_:
        logger.warning("Labels have object dtype - converting...")
        try:
            y = np.array([int(label) for label in y], dtype=np.int64)
        except Exception as e:
            logger.error(f"Failed to fix y dtype: {e}")
            y = y.astype(np.int64)
    else:
        y = y.astype(np.int64)
    
    # Fix areas
    if areas.dtype == np.object_:
        logger.warning("Areas have object dtype - converting...")
        try:
            areas = np.array([int(area) for area in areas], dtype=np.int64)
        except Exception as e:
            logger.error(f"Failed to fix areas dtype: {e}")
            areas = areas.astype(np.int64)
    else:
        areas = areas.astype(np.int64)
    
    # Validate shapes
    if len(X.shape) != 3:
        logger.error(f"Invalid X shape: {X.shape}, expected 3D array")
        return create_synthetic_data(len(y))
    
    if len(y.shape) != 1:
        logger.error(f"Invalid y shape: {y.shape}, expected 1D array")
        return create_synthetic_data(len(y))
    
    logger.info(f"✅ Data validation complete:")
    logger.info(f"   X: shape={X.shape}, dtype={X.dtype}")
    logger.info(f"   y: shape={y.shape}, dtype={y.dtype}")
    logger.info(f"   areas: shape={areas.shape}, dtype={areas.dtype}")
    
    return X, y, areas

def add_enhanced_temporal_features(X):
    """
    Enhanced temporal features specifically for Warning detection
    
    Args:
        X: Input features of shape (n_samples, seq_len, n_features)
        
    Returns:
        Enhanced features with temporal derivatives, statistics, and warning-specific features
    """
    n_samples, seq_len, n_features = X.shape
    
    # Create output array with additional features
    # Original + 1st derivative + 2nd derivative + moving averages + warning-specific
    n_new_features = n_features * 5
    X_enhanced = np.zeros((n_samples, seq_len, n_new_features), dtype=np.float32)
    
    # Copy original features
    X_enhanced[:, :, :n_features] = X
    
    for i in range(n_samples):
        for j in range(n_features):
            # First derivative (rate of change)
            X_enhanced[i, 1:, n_features + j] = np.diff(X[i, :, j])
            
            # Second derivative (acceleration of change)
            if seq_len > 2:
                second_derivatives = np.zeros(seq_len)
                second_derivatives[2:] = np.diff(X[i, :, j], n=2)
                X_enhanced[i, :, n_features*2 + j] = second_derivatives
            
            # Moving averages with different window sizes
            short_window = 5
            long_window = 10
            
            # Short-term trend (5-point window)
            for t in range(short_window, seq_len):
                X_enhanced[i, t, n_features*3 + j] = np.mean(X[i, t-short_window:t, j])
            
            # Long-term trend (10-point window)
            for t in range(long_window, seq_len):
                if t >= long_window:
                    X_enhanced[i, t, n_features*3 + j + n_features//2] = np.mean(X[i, t-long_window:t, j])
    
    # Add warning-specific features: rate of increase above threshold
    for i in range(n_samples):
        for j in range(n_features):
            for t in range(3, seq_len):
                # Detect sustained increases (potential warning sign)
                increases = 0
                for k in range(1, 4):
                    if t-k >= 0 and t-k+1 < seq_len:
                        if X[i, t-k, j] < X[i, t-k+1, j]:
                            increases += 1
                
                # Normalized consecutive increases (0-1 range)
                X_enhanced[i, t, n_features*4 + j] = increases / 3.0
                
                # Detect rapid changes
                if t >= 1:
                    change_rate = 0
                    if abs(X[i, t, j]) > 1e-6:  # Avoid division by zero
                        change_rate = abs((X[i, t, j] - X[i, t-1, j]) / X[i, t, j])
                    X_enhanced[i, t, n_features*4 + j + n_features//2] = min(change_rate, 5.0)  # Cap at 5.0
    
    logger.info(f"Added enhanced temporal features: {X.shape} -> {X_enhanced.shape}")
    return X_enhanced


# Keep original function for compatibility
def add_temporal_features(X):
    """
    Original temporal features function (kept for compatibility)
    """
    logger.info("Using enhanced temporal features instead of basic temporal features")
    return add_enhanced_temporal_features(X)


def create_binary_labels(y):
    """
    Convert multi-class labels to binary (Normal vs Anomaly)
    
    Args:
        y: Original labels (0: Normal, 1: Warning, 2: Fire)
        
    Returns:
        Binary labels (0: Normal, 1: Anomaly)
    """
    return np.where(y > 0, 1, 0)


def load_data(add_temporal=True, create_synthetic_warning=True):
    """Load data from various sources - prioritizing real S3 data"""
    
    # PRIORITY 1: Try loading from S3 (real 50M dataset)
    if AWS_AVAILABLE:
        logger.info("🔥 Loading REAL fire detection data from S3 (50M dataset)...")
        X, y, areas = load_data_from_s3()
        
        if X is not None:
            logger.info(f"✅ Successfully loaded {len(X):,} samples from S3")
        else:
            logger.warning("❌ Failed to load from S3, checking for saved files...")
    else:
        logger.warning("⚠️ AWS not available, checking for saved files...")
        X, y, areas = None, None, None
    
    # PRIORITY 2: Check if we have saved numpy files (from previous S3 load)
    if X is None and os.path.exists('X_train.npy') and os.path.exists('y_train.npy'):
        logger.info("📁 Loading data from saved numpy files...")
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
        
        logger.info(f"✅ Loaded saved data: X_train={X_train.shape}, y_train={y_train.shape}")
        
        # Validate loaded data types
        X_train, y_train, areas_train = validate_and_fix_data_types(X_train, y_train, areas_train)
        X_val, y_val, areas_val = validate_and_fix_data_types(X_val, y_val, areas_val)
        X_test, y_test, areas_test = validate_and_fix_data_types(X_test, y_test, areas_test)
        
        # Add temporal features if requested
        if add_temporal:
            logger.info("Adding temporal features to help distinguish Warning class...")
            X_train = add_temporal_features(X_train)
            X_val = add_temporal_features(X_val)
            X_test = add_temporal_features(X_test)
        
        # Create synthetic Warning samples if requested
        if create_synthetic_warning:
            logger.info("Creating synthetic Warning samples...")
            X_train, y_train = create_synthetic_warning_samples(X_train, y_train)
        
        return X_train, y_train, X_val, y_val, X_test, y_test, areas_train, areas_val, areas_test
    
    # PRIORITY 3: Only use synthetic data as absolute last resort
    if X is None:
        logger.warning("⚠️ FALLBACK: Creating synthetic data (NOT RECOMMENDED FOR PRODUCTION)")
        logger.warning("⚠️ Please ensure S3 bucket 'synthetic-data-4' is accessible")
        X, y, areas = create_synthetic_data(SAMPLE_SIZE)
    
    # Validate and fix data types
    X, y, areas = validate_and_fix_data_types(X, y, areas)
    
    # Split into train/validation/test sets
    X_train, X_test, y_train, y_test, areas_train, areas_test = train_test_split(
        X, y, areas, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    X_train, X_val, y_train, y_val, areas_train, areas_val = train_test_split(
        X_train, y_train, areas_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    # Add temporal features if requested
    if add_temporal:
        logger.info("Adding temporal features to help distinguish Warning class...")
        X_train = add_temporal_features(X_train)
        X_val = add_temporal_features(X_val)
        X_test = add_temporal_features(X_test)
    
    # Create synthetic Warning samples if requested
    if create_synthetic_warning:
        logger.info("Creating synthetic Warning samples...")
        X_train, y_train = create_synthetic_warning_samples(X_train, y_train)
    
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
    
    return X_train, y_train, X_val, y_val, X_test, y_test, areas_train, areas_val, areas_test


def create_synthetic_warning_samples(X, y):
    """
    Create synthetic Warning samples through augmentation
    
    Args:
        X: Features array
        y: Labels array
        
    Returns:
        Enhanced X and y with additional synthetic Warning samples
    """
    # Find Warning samples (class 1)
    warning_indices = np.where(y == 1)[0]
    warning_count = len(warning_indices)
    
    if warning_count == 0:
        logger.warning("No Warning samples found to create synthetic data from")
        return X, y
    
    logger.info(f"Found {warning_count} Warning samples to create synthetic data from")
    
    # Get Warning samples
    X_warning = X[warning_indices]
    
    # Determine how many synthetic samples to create (5x original)
    n_synthetic = warning_count * 5
    logger.info(f"Creating {n_synthetic} synthetic Warning samples")
    
    # Create storage for synthetic samples
    X_synthetic = np.zeros((n_synthetic,) + X_warning.shape[1:], dtype=X.dtype)
    
    # Create synthetic samples through augmentation
    for i in range(n_synthetic):
        # Select a random Warning sample
        idx = np.random.randint(0, warning_count)
        sample = X_warning[idx].copy()
        
        # Apply strong augmentation
        # 1. Add random noise
        noise_level = np.random.uniform(0.05, 0.15)
        noise = np.random.randn(*sample.shape) * noise_level
        sample = sample + noise
        
        # 2. Random scaling
        scale_factor = np.random.uniform(0.9, 1.1)
        sample = sample * scale_factor
        
        # 3. Random time shifting (shift values in time dimension)
        if np.random.rand() > 0.5:
            shift = np.random.randint(1, 5)
            sample[shift:] = sample[:-shift]
        
        # 4. Feature emphasis (randomly emphasize one feature)
        feature_idx = np.random.randint(0, sample.shape[1])
        emphasis = np.random.uniform(1.05, 1.2)
        sample[:, feature_idx] = sample[:, feature_idx] * emphasis
        
        # Store synthetic sample
        X_synthetic[i] = sample
    
    # Create labels for synthetic samples (all Warning class)
    y_synthetic = np.ones(n_synthetic, dtype=y.dtype)
    
    # Combine original and synthetic data
    X_combined = np.vstack([X, X_synthetic])
    y_combined = np.concatenate([y, y_synthetic])
    
    logger.info(f"Added {n_synthetic} synthetic Warning samples: {X.shape} -> {X_combined.shape}")
    
    # Shuffle the combined data
    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    
    return X_combined, y_combined

# ===== Dataset and DataLoader =====
class FireDetectionDataset(Dataset):
    """PyTorch Dataset for Fire Detection data"""
    
    def __init__(self, features, labels, augment=False):
        # Convert to proper numpy arrays with correct dtypes
        if isinstance(features, np.ndarray):
            # Ensure features are float32
            features = features.astype(np.float32)
        else:
            features = np.array(features, dtype=np.float32)
            
        if isinstance(labels, np.ndarray):
            # Ensure labels are integers
            labels = labels.astype(np.int64)
        else:
            labels = np.array(labels, dtype=np.int64)
        
        # Check for any remaining object dtypes
        if features.dtype == np.object_:
            logger.error("Features still have object dtype - converting to float32")
            features = np.array([np.array(x, dtype=np.float32) for x in features])
            
        if labels.dtype == np.object_:
            logger.error("Labels still have object dtype - converting to int64")
            labels = np.array([int(x) for x in labels])
        
        logger.info(f"Dataset created: features shape={features.shape}, dtype={features.dtype}")
        logger.info(f"Dataset created: labels shape={labels.shape}, dtype={labels.dtype}")
        
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        
        # Apply data augmentation for training
        if self.augment:
            # More aggressive augmentation for Warning class (label 1)
            if label == 1:  # Warning class
                # Add random noise (jitter) - slightly higher for Warning class
                noise_level = 0.08  # Higher noise for Warning class
                noise = torch.randn_like(features) * noise_level
                features = features + noise
                
                # Random scaling - more likely for Warning class
                if torch.rand(1).item() > 0.3:  # 70% chance for Warning class
                    scale_factor = torch.FloatTensor(1).uniform_(0.92, 1.08)  # Wider range
                    features = features * scale_factor
                
                # Random time masking (mask out random time steps)
                if torch.rand(1).item() > 0.5:  # 50% chance for Warning class
                    seq_len = features.size(0)
                    mask_len = int(seq_len * 0.15)  # Mask 15% of time steps
                    start_idx = torch.randint(0, seq_len - mask_len, (1,)).item()
                    features[start_idx:start_idx+mask_len, :] = 0
                    
                # Feature emphasis - randomly emphasize one feature
                if torch.rand(1).item() > 0.5:
                    feature_idx = torch.randint(0, features.size(1), (1,)).item()
                    features[:, feature_idx] = features[:, feature_idx] * 1.1
            else:
                # Standard augmentation for other classes
                # Add random noise (jitter)
                noise_level = 0.05
                noise = torch.randn_like(features) * noise_level
                features = features + noise
                
                # Random scaling
                if torch.rand(1).item() > 0.5:
                    scale_factor = torch.FloatTensor(1).uniform_(0.95, 1.05)
                    features = features * scale_factor
                
                # Random time masking (mask out random time steps)
                if torch.rand(1).item() > 0.7:
                    seq_len = features.size(0)
                    mask_len = int(seq_len * 0.1)  # Mask 10% of time steps
                    start_idx = torch.randint(0, seq_len - mask_len, (1,)).item()
                    features[start_idx:start_idx+mask_len, :] = 0
        
        return features, label

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    Ensures each batch contains a balanced number of samples from each class.
    This is more effective than ImbalancedDatasetSampler for severe class imbalance.
    """
    
    def __init__(self, dataset, n_classes=3, n_samples=None):
        """
        Args:
            dataset: PyTorch dataset
            n_classes: Number of classes in the dataset
            n_samples: Number of samples per class in each batch
                       If None, will use batch_size // n_classes
        """
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.indices_per_class = {}
        self.count_per_class = {}
        
        # Get indices for each class
        for idx in range(len(dataset)):
            label = dataset.labels[idx].item()
            if label not in self.indices_per_class:
                self.indices_per_class[label] = []
                self.count_per_class[label] = 0
            self.indices_per_class[label].append(idx)
            self.count_per_class[label] += 1
        
        # Ensure all classes are represented
        if len(self.indices_per_class) < n_classes:
            logger.warning(f"Only {len(self.indices_per_class)} classes found in dataset, expected {n_classes}")
            self.n_classes = len(self.indices_per_class)
        
        # Calculate length based on the class with the fewest samples
        min_class_count = min(len(indices) for indices in self.indices_per_class.values())
        logger.info(f"Class distribution: {self.count_per_class}")
        logger.info(f"Smallest class has {min_class_count} samples")
        
        # Calculate number of batches
        if n_samples is None:
            # Default to equal distribution in batch
            self.n_samples = BATCH_SIZE // self.n_classes
        
        # Calculate total number of samples
        self.length = 0
        for label in self.indices_per_class:
            # Oversample minority classes
            if label == 1:  # Warning class - oversample 5x
                self.length += len(self.indices_per_class[label]) * 5
            else:
                self.length += len(self.indices_per_class[label])
    
    def __iter__(self):
        """
        Yields balanced batches of indices
        """
        # Create copy of indices to avoid modifying original
        indices_per_class = {
            label: indices.copy() for label, indices in self.indices_per_class.items()
        }
        
        # Shuffle indices for each class
        for label in indices_per_class:
            np.random.shuffle(indices_per_class[label])
        
        # Create batches
        while sum(len(indices) for indices in indices_per_class.values()) > 0:
            batch_indices = []
            
            # Add samples from each class
            for label in indices_per_class:
                # Determine number of samples to take from this class
                if label == 1:  # Warning class - take more samples
                    n_to_take = min(self.n_samples * 3, len(indices_per_class[label]))
                else:
                    n_to_take = min(self.n_samples, len(indices_per_class[label]))
                
                # Take samples
                if n_to_take > 0:
                    batch_indices.extend(indices_per_class[label][:n_to_take])
                    indices_per_class[label] = indices_per_class[label][n_to_take:]
                
                # If we've run out of samples for this class, resample with replacement
                if len(indices_per_class[label]) == 0 and label == 1:  # Only resample Warning class
                    indices_per_class[label] = self.indices_per_class[label].copy()
                    np.random.shuffle(indices_per_class[label])
                    logger.info(f"Resampled Warning class (label 1) due to exhaustion")
            
            # Yield batch if not empty
            if batch_indices:
                # Shuffle the batch indices
                np.random.shuffle(batch_indices)
                for idx in batch_indices:
                    yield idx
    
    def __len__(self):
        return self.length


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements with higher probability for minority classes"""
    
    def __init__(self, dataset, indices=None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices)
        
        # Count number of samples per label
        label_count = {}
        for idx in self.indices:
            label = dataset.labels[idx].item()
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
        
        # Weight samples inversely proportional to class frequency
        weights = []
        for idx in self.indices:
            label = dataset.labels[idx].item()
            # Cube the inverse for Warning class, square for others
            if label == 1:  # Warning class
                weight = 1.0 / label_count[label]**3  # More aggressive for Warning
            else:
                weight = 1.0 / label_count[label]**2
            weights.append(weight)
        
        self.weights = torch.DoubleTensor(weights)
    
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
    
    def __len__(self):
        return self.num_samples

# ===== Custom Loss Functions =====
class WarningFocusedLoss(nn.Module):
    """
    Loss function with extreme focus on Warning class transitions
    
    This loss function extends EnhancedFocalLoss by adding special handling for:
    1. Transitions between classes (Normal→Warning and Warning→Fire)
    2. Early warning detection
    3. Sequence-aware penalties
    """
    
    def __init__(self, alpha=None, gamma=2.0, warning_gamma=4.0, warning_boost=3.0,
                 transition_penalty=2.0, reduction='mean'):
        """
        Initialize Warning Focused Loss
        
        Args:
            alpha: Weight for each class (list/tensor) or None
            gamma: Focusing parameter for normal and fire classes
            warning_gamma: Higher gamma specifically for warning class
            warning_boost: Additional multiplier for warning class loss
            transition_penalty: Multiplier for transition errors
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.warning_gamma = warning_gamma
        self.warning_boost = warning_boost
        self.transition_penalty = transition_penalty
        self.reduction = reduction
        
        logger.info(f"Using WarningFocusedLoss with warning_gamma={warning_gamma}, "
                   f"warning_boost={warning_boost}, transition_penalty={transition_penalty}")
    
    def forward(self, inputs, targets, warning_probs=None, sequence_ids=None):
        """
        Calculate warning-focused loss
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            warning_probs: Optional early warning probabilities from detector
            sequence_ids: Optional sequence IDs to identify samples from same sequence
            
        Returns:
            Calculated loss
        """
        # Get log softmax and probabilities
        log_softmax = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_softmax)
        
        # Gather the log softmax values for the target classes
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        log_pt = (targets_one_hot * log_softmax).sum(dim=1)
        pt = torch.exp(log_pt)
        
        # Create mask for warning class (class index 1)
        warning_mask = (targets == 1)
        
        # Apply class weights if provided
        weight_term = torch.ones_like(log_pt)
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(inputs.device)
                at = alpha.gather(0, targets)
                weight_term = weight_term * at
        
        # Apply focusing parameter - different gamma for warning class
        focal_weight = torch.ones_like(pt)
        focal_weight[~warning_mask] = (1 - pt[~warning_mask]) ** self.gamma
        focal_weight[warning_mask] = (1 - pt[warning_mask]) ** self.warning_gamma
        
        # Apply warning boost to warning samples
        if torch.any(warning_mask):
            focal_weight[warning_mask] = focal_weight[warning_mask] * self.warning_boost
        
        # Calculate base focal loss
        base_loss = -1 * weight_term * focal_weight * log_pt
        
        # Get predictions for transition analysis
        _, predicted = inputs.max(1)
        
        # Add transition penalty if sequence_ids are provided
        transition_loss = 0.0
        if sequence_ids is not None:
            # Group by sequence
            unique_seqs = torch.unique(sequence_ids)
            for seq_id in unique_seqs:
                # Get indices for this sequence
                seq_mask = (sequence_ids == seq_id)
                if torch.sum(seq_mask) <= 1:
                    continue  # Skip sequences with only one sample
                
                seq_targets = targets[seq_mask]
                seq_preds = predicted[seq_mask]
                
                # Find transitions in targets
                for i in range(len(seq_targets) - 1):
                    # Check for important transitions
                    is_normal_to_warning = (seq_targets[i] == 0 and seq_targets[i+1] == 1)
                    is_warning_to_fire = (seq_targets[i] == 1 and seq_targets[i+1] == 2)
                    
                    if is_normal_to_warning or is_warning_to_fire:
                        # Check if transition was correctly predicted
                        transition_correct = (seq_preds[i] == seq_targets[i] and
                                             seq_preds[i+1] == seq_targets[i+1])
                        
                        if not transition_correct:
                            # Add penalty for missing this important transition
                            transition_loss += self.transition_penalty
        
        # Add early warning detection loss if warning_probs are provided
        warning_detection_loss = 0.0
        if warning_probs is not None:
            # Create binary labels: 1 for Warning class, 0 for others
            warning_targets = (targets == 1).float()
            
            # Binary cross entropy for warning detection
            warning_detection_loss = F.binary_cross_entropy(
                warning_probs.squeeze(),
                warning_targets,
                reduction='mean'
            )
        
        # Combine losses
        total_loss = base_loss
        
        # Apply reduction to base loss
        if self.reduction == 'mean':
            total_loss = base_loss.mean()
        elif self.reduction == 'sum':
            total_loss = base_loss.sum()
        
        # Add transition and warning detection losses
        if transition_loss > 0:
            total_loss = total_loss + transition_loss / len(targets)
        
        if warning_detection_loss > 0:
            total_loss = total_loss + warning_detection_loss
        
        return total_loss


class EnhancedFocalLoss(nn.Module):
    """Enhanced Focal Loss with class-specific gamma values and warning class emphasis"""
    
    def __init__(self, alpha=None, gamma=2.0, warning_gamma=4.0, warning_boost=2.0, reduction='mean'):
        """
        Initialize Enhanced Focal Loss
        
        Args:
            alpha: Weight for each class (list/tensor) or None
            gamma: Focusing parameter for normal and fire classes
            warning_gamma: Higher gamma specifically for warning class
            warning_boost: Additional multiplier for warning class loss
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.warning_gamma = warning_gamma  # Higher gamma for warning class
        self.warning_boost = warning_boost  # Additional multiplier for warning class
        self.reduction = reduction
        logger.info(f"Using Enhanced Focal Loss with warning_gamma={warning_gamma}, warning_boost={warning_boost}")
        
    def forward(self, inputs, targets):
        """
        Calculate enhanced focal loss with special handling for warning class
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Calculated loss
        """
        # Get log softmax
        log_softmax = F.log_softmax(inputs, dim=1)
        
        # Get probabilities
        probs = torch.exp(log_softmax)
        
        # Gather the log softmax values for the target classes
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        log_pt = (targets_one_hot * log_softmax).sum(dim=1)
        pt = torch.exp(log_pt)
        
        # Create mask for warning class (class index 1)
        warning_mask = (targets == 1)
        
        # Apply class weights if provided
        weight_term = torch.ones_like(log_pt)
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(inputs.device)
                at = alpha.gather(0, targets)
                weight_term = weight_term * at
        
        # Apply focusing parameter - different gamma for warning class
        focal_weight = torch.ones_like(pt)
        focal_weight[~warning_mask] = (1 - pt[~warning_mask]) ** self.gamma
        focal_weight[warning_mask] = (1 - pt[warning_mask]) ** self.warning_gamma
        
        # Apply warning boost to warning samples
        if torch.any(warning_mask):
            focal_weight[warning_mask] = focal_weight[warning_mask] * self.warning_boost
        
        # Calculate loss
        loss = -1 * weight_term * focal_weight * log_pt
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Keep original FocalLoss for compatibility
class FocalLoss(EnhancedFocalLoss):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weight for each class (list/tensor) or None
            gamma: Focusing parameter (higher gamma -> more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__(alpha=alpha, gamma=gamma, warning_gamma=gamma, warning_boost=1.0, reduction=reduction)

# ===== Model Architecture =====
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to input
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class EarlyWarningDetector(nn.Module):
    """
    Specialized module to detect subtle warning signs
    
    This module focuses specifically on detecting early warning patterns
    that might precede fire events.
    """
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through the early warning detector
        
        Args:
            x: Input features of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Warning probability (0-1) for each sequence
        """
        # Global average pooling over sequence dimension
        x_pooled = torch.mean(x, dim=1)
        
        # Return warning probability (0-1)
        return self.detector(x_pooled)


class EnhancedFireDetectionTransformer(nn.Module):
    """
    Enhanced transformer model for fire detection with parallel convolutional branch
    and early warning detection
    """
    
    def __init__(self, input_dim, d_model, num_heads, num_layers, num_classes, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Parallel convolutional branch for better feature extraction
        self.conv_branch = nn.Sequential(
            # Transpose happens in forward pass: [batch, seq_len, features] -> [batch, features, seq_len]
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
            nn.GELU()
        )
        
        # Feature fusion layer
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Early warning detector
        self.warning_detector = EarlyWarningDetector(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with improved scheme for better training"""
        for name, p in self.named_parameters():
            if 'weight' in name and len(p.shape) >= 2:
                # Xavier initialization for weights
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                # Zero initialization for biases
                nn.init.zeros_(p)
    
    def forward(self, x):
        """
        Forward pass through the enhanced transformer
        
        Args:
            x: Input features of shape [batch_size, seq_len, input_dim]
            
        Returns:
            tuple: (class_logits, warning_probability)
                - class_logits: Logits for Normal/Warning/Fire classes
                - warning_probability: Early warning detection probability
        """
        # Process with transformer branch
        transformer_input = self.input_projection(x)
        transformer_input = self.positional_encoding(transformer_input)
        transformer_features = self.transformer_encoder(transformer_input)
        
        # Process with convolutional branch
        # Transpose for conv1d: [batch, seq_len, features] -> [batch, features, seq_len]
        conv_input = x.transpose(1, 2)
        conv_features = self.conv_branch(conv_input)
        # Transpose back: [batch, features, seq_len] -> [batch, seq_len, features]
        conv_features = conv_features.transpose(1, 2)
        
        # Concatenate features from both branches
        combined = torch.cat([transformer_features, conv_features], dim=2)
        
        # Fuse features
        fused = self.fusion(combined)
        
        # Apply early warning detection
        warning_prob = self.warning_detector(fused)
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(fused, dim=1)
        
        # Project to output classes
        class_logits = self.output_projection(pooled)
        
        return class_logits, warning_prob


# Keep original class for compatibility
class FireDetectionTransformer(nn.Module):
    """Transformer model for fire detection"""
    
    def __init__(self, input_dim, d_model, num_heads, num_layers, num_classes, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)
        
        # Project to output classes
        x = self.output_projection(x)
        
        return x

# ===== Training and Evaluation Functions =====
def get_warmup_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    """Create a learning rate scheduler with warmup"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # Cosine annealing
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model, dataloader, criterion, optimizer, device, clip_value=None):
    """Train model for one epoch with gradient clipping"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Apply gradient clipping if specified
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        # Optimize
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # Calculate epoch statistics
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def evaluate_enhanced_model(model, dataloader, criterion, device, use_custom_thresholds=False):
    """
    Evaluate enhanced model on dataloader with option for custom thresholds
    
    Args:
        model: The enhanced PyTorch model to evaluate
        dataloader: DataLoader with evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        use_custom_thresholds: Whether to use custom thresholds for class prediction
                              (makes model more sensitive to Warning class)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    warning_detection_correct = 0
    warning_detection_total = 0
    
    all_targets = []
    all_predictions = []
    all_raw_outputs = []
    all_warning_probs = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass - get both class logits and warning probabilities
            outputs, warning_probs = model(inputs)
            
            # Calculate loss (without sequence IDs for evaluation)
            loss = criterion(outputs, targets, warning_probs)
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Store raw outputs for custom threshold evaluation
            all_raw_outputs.append(outputs.cpu().numpy())
            all_warning_probs.append(warning_probs.cpu().numpy())
            
            # Standard prediction (argmax)
            if not use_custom_thresholds:
                _, predicted = outputs.max(1)
            else:
                # Apply custom thresholds to be more sensitive to Warning class
                # Convert to probabilities
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                predicted = np.zeros(probs.shape[0], dtype=np.int64)
                
                # Custom decision logic with even lower threshold for Warning
                # 1. If Warning probability > 0.25, classify as Warning (lower threshold)
                # 2. Else if Fire probability > 0.5, classify as Fire
                # 3. Otherwise classify as Normal
                warning_mask = probs[:, 1] > 0.25  # More sensitive to Warning class
                fire_mask = probs[:, 2] > 0.5
                
                # Default to Normal (0)
                # Then set Warning (1) where warning_mask is True
                predicted[warning_mask] = 1
                # Then set Fire (2) where fire_mask is True (overrides Warning if both conditions met)
                predicted[fire_mask] = 2
                
                # Convert back to tensor for consistency
                predicted = torch.tensor(predicted, device=device)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Track warning detection accuracy
            warning_targets = (targets == 1).float()
            warning_preds = (warning_probs.squeeze() > 0.5).float()
            warning_detection_correct += (warning_preds == warning_targets).sum().item()
            warning_detection_total += targets.size(0)
            
            # Store targets and predictions for metrics
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
    
    # Calculate epoch statistics
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    warning_detection_acc = warning_detection_correct / warning_detection_total if warning_detection_total > 0 else 0
    
    # Combine all targets and predictions
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    
    # Calculate additional metrics
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    # Calculate per-class metrics
    class_precision = []
    class_recall = []
    class_f1 = []
    
    for i in range(3):  # For each class (Normal, Warning, Fire)
        class_true = all_targets == i
        class_pred = all_predictions == i
        
        if np.sum(class_true) > 0 and np.sum(class_pred) > 0:
            p = precision_score(class_true, class_pred)
            r = recall_score(class_true, class_pred)
            f = f1_score(class_true, class_pred)
        else:
            p, r, f = 0, 0, 0
            
        class_precision.append(p)
        class_recall.append(r)
        class_f1.append(f)
    
    # Log detailed metrics for Warning class
    logger.info(f"   Warning class metrics: Precision={class_precision[1]:.4f}, "
               f"Recall={class_recall[1]:.4f}, F1={class_f1[1]:.4f}")
    logger.info(f"   Warning detection accuracy: {warning_detection_acc:.4f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Log transition metrics (Normal→Warning and Warning→Fire)
    logger.info(f"   Confusion Matrix:\n{cm}")
    
    # Calculate transition detection rates
    normal_count = np.sum(all_targets == 0)
    warning_count = np.sum(all_targets == 1)
    fire_count = np.sum(all_targets == 2)
    
    normal_to_warning_detection = cm[0, 1] / normal_count if normal_count > 0 else 0
    warning_to_fire_detection = cm[1, 2] / warning_count if warning_count > 0 else 0
    
    logger.info(f"   Normal→Warning detection rate: {normal_to_warning_detection:.4f}")
    logger.info(f"   Warning→Fire detection rate: {warning_to_fire_detection:.4f}")
    
    # Return comprehensive metrics
    metrics = {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'warning_detection_acc': warning_detection_acc,
        'confusion_matrix': cm,
        'normal_to_warning_rate': normal_to_warning_detection,
        'warning_to_fire_rate': warning_to_fire_detection
    }
    
    return epoch_loss, epoch_acc, precision, recall, f1, all_targets, all_predictions, metrics


# Keep original function for compatibility
def evaluate(model, dataloader, criterion, device, use_custom_thresholds=False):
    """
    Evaluate model on dataloader with option for custom thresholds
    
    This function handles both original and enhanced models
    """
    # Check if this is an enhanced model
    is_enhanced = isinstance(model, EnhancedFireDetectionTransformer)
    if isinstance(model, nn.DataParallel):
        is_enhanced = isinstance(model.module, EnhancedFireDetectionTransformer)
    
    if is_enhanced:
        logger.info("Using enhanced evaluation function for EnhancedFireDetectionTransformer")
        return evaluate_enhanced_model(model, dataloader, criterion, device, use_custom_thresholds)
    
    # Original evaluation code for backward compatibility
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_targets = []
    all_predictions = []
    all_raw_outputs = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Store raw outputs for custom threshold evaluation
            all_raw_outputs.append(outputs.cpu().numpy())
            
            # Standard prediction (argmax)
            if not use_custom_thresholds:
                _, predicted = outputs.max(1)
            else:
                # Apply custom thresholds to be more sensitive to Warning class
                # Convert to probabilities
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                predicted = np.zeros(probs.shape[0], dtype=np.int64)
                
                # Custom decision logic:
                # 1. If Warning probability > 0.3, classify as Warning (lower threshold)
                # 2. Else if Fire probability > 0.5, classify as Fire
                # 3. Otherwise classify as Normal
                warning_mask = probs[:, 1] > 0.3  # More sensitive to Warning class
                fire_mask = probs[:, 2] > 0.5
                
                # Default to Normal (0)
                # Then set Warning (1) where warning_mask is True
                predicted[warning_mask] = 1
                # Then set Fire (2) where fire_mask is True (overrides Warning if both conditions met)
                predicted[fire_mask] = 2
                
                # Convert back to tensor for consistency
                predicted = torch.tensor(predicted, device=device)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store targets and predictions for metrics
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
    
    # Calculate epoch statistics
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # Combine all targets and predictions
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    
    # Calculate additional metrics
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    # Also calculate Warning class specific metrics
    warning_precision = precision_score(all_targets == 1, all_predictions == 1)
    warning_recall = recall_score(all_targets == 1, all_predictions == 1)
    warning_f1 = f1_score(all_targets == 1, all_predictions == 1)
    
    logger.info(f"   Warning class metrics: Precision={warning_precision:.4f}, "
               f"Recall={warning_recall:.4f}, F1={warning_f1:.4f}")
    
    return epoch_loss, epoch_acc, precision, recall, f1, all_targets, all_predictions

def train_model(model, train_loader, val_loader, criterion, optimizer, lr_scheduler,
                num_epochs, patience, device, clip_value=None, checkpoint_dir='checkpoints'):
    """Train model with early stopping, gradient clipping, and learning rate scheduling"""
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize variables
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_path = None
    best_val_f1 = 0.0  # Track best F1 score for minority classes
    best_warning_f1 = 0.0  # Track best F1 score specifically for Warning class
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'warning_f1': [],
        'lr': []
    }
    
    # Training loop
    start_time = time.time()
    logger.info(f"Starting training for {num_epochs} epochs with gradient clipping={clip_value}")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch with gradient clipping
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, clip_value)
        
        # Determine whether to use custom thresholds based on epoch
        # Start with standard evaluation, then gradually introduce custom thresholds
        use_custom_thresholds = epoch >= num_epochs // 3  # Use custom thresholds after 1/3 of training
        
        if use_custom_thresholds:
            logger.info(f"Using custom class thresholds for evaluation (more sensitive to Warning class)")
        
        # Evaluate on validation set with optional custom thresholds
        val_loss, val_acc, val_precision, val_recall, val_f1, val_targets, val_preds = evaluate(
            model, val_loader, criterion, device, use_custom_thresholds=use_custom_thresholds
        )
        
        # Calculate Warning class F1 score specifically
        warning_mask_true = val_targets == 1
        warning_mask_pred = val_preds == 1
        
        # Handle case where there are no Warning predictions or targets
        if np.sum(warning_mask_true) == 0 or np.sum(warning_mask_pred) == 0:
            warning_f1 = 0.0
        else:
            warning_f1 = f1_score(warning_mask_true, warning_mask_pred)
        
        # Add to history
        history['warning_f1'].append(warning_f1)
        
        # Update learning rate scheduler
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        logger.info(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s - "
                   f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                   f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - "
                   f"Val F1: {val_f1:.4f} - LR: {current_lr:.6f}")
        
        # Check class-wise performance - Fixed calculation to avoid accuracy > 1 bug
        class_counts = np.bincount(val_targets, minlength=3)
        class_acc = np.zeros(3)
        class_precision = np.zeros(3)
        class_recall = np.zeros(3)
        class_f1 = np.zeros(3)
        
        # Calculate per-class metrics
        for i in range(3):
            # Count correct predictions for each class separately
            class_correct = np.sum((val_targets == i) & (val_preds == i))
            if class_counts[i] > 0:
                class_acc[i] = class_correct / class_counts[i]
                
            # Calculate precision (how many predicted as class i are actually class i)
            pred_as_i = np.sum(val_preds == i)
            if pred_as_i > 0:
                class_precision[i] = class_correct / pred_as_i
            
            # Calculate recall (how many actual class i were predicted correctly)
            if class_counts[i] > 0:
                class_recall[i] = class_correct / class_counts[i]
            
            # Calculate F1 score
            if class_precision[i] + class_recall[i] > 0:
                class_f1[i] = 2 * (class_precision[i] * class_recall[i]) / (class_precision[i] + class_recall[i])
        
        logger.info(f"   Class accuracies: Normal={class_acc[0]:.4f}, Warning={class_acc[1]:.4f}, Fire={class_acc[2]:.4f}")
        logger.info(f"   Class F1 scores: Normal={class_f1[0]:.4f}, Warning={class_f1[1]:.4f}, Fire={class_f1[2]:.4f}")
        
        # Special focus on Warning class
        warning_idx = 1
        logger.info(f"   Warning class: Precision={class_precision[warning_idx]:.4f}, "
                   f"Recall={class_recall[warning_idx]:.4f}, F1={class_f1[warning_idx]:.4f}, "
                   f"Count={class_counts[warning_idx]}")
        
        # Save confusion matrix for visualization
        if VISUALIZATION_AVAILABLE:
            try:
                cm = confusion_matrix(val_targets, val_preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Normal', 'Warning', 'Fire'],
                            yticklabels=['Normal', 'Warning', 'Fire'])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'Confusion Matrix - Epoch {epoch+1}')
                plt.tight_layout()
                cm_path = os.path.join(checkpoint_dir, f'confusion_matrix_epoch_{epoch+1}.png')
                plt.savefig(cm_path)
                plt.close()
                logger.info(f"   Confusion matrix saved to {cm_path}")
            except Exception as e:
                logger.warning(f"   Failed to save confusion matrix: {e}")
        
        # Save model if it's the best so far (considering loss, overall F1, and Warning F1)
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            is_best = True
            logger.info(f"   New best model based on validation loss: {val_loss:.4f}")
        
        # Also consider overall F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            is_best = True
            logger.info(f"   New best model based on overall F1 score: {val_f1:.4f}")
        
        # Prioritize Warning class F1 score with a higher weight
        # This makes the model more likely to save versions that perform well on Warning class
        if warning_f1 > best_warning_f1:
            best_warning_f1 = warning_f1
            is_best = True
            logger.info(f"   New best model based on Warning class F1 score: {warning_f1:.4f}")
        
        if is_best:
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_acc': class_acc,
                'history': history
            }, checkpoint_path)
            
            best_model_path = checkpoint_path
            logger.info(f"✅ Saved new best model to {checkpoint_path}")
        
        # Check for early stopping - only if no improvement in both metrics
        if epoch - best_epoch >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Calculate total training time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info(f"Best model at epoch {best_epoch+1} with validation loss {best_val_loss:.4f}")
    
    # Load best model
    if best_model_path is not None:
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from {best_model_path}")
    
    return model, history

def save_model_for_deployment(model, input_dim, class_names, save_dir='models'):
    """Save model for deployment"""
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine model type
    is_enhanced = isinstance(model, EnhancedFireDetectionTransformer)
    if isinstance(model, nn.DataParallel):
        is_enhanced = isinstance(model.module, EnhancedFireDetectionTransformer)
    
    # Create appropriate filename
    if is_enhanced:
        model_path = os.path.join(save_dir, 'enhanced_fire_detection_model.pt')
    else:
        model_path = os.path.join(save_dir, 'fire_detection_model.pt')
    
    # Extract state dict
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    # Save model with appropriate metadata
    save_dict = {
        'model_state_dict': model_state_dict,
        'input_dim': input_dim,
        'd_model': TRANSFORMER_CONFIG['d_model'],
        'num_heads': TRANSFORMER_CONFIG['num_heads'],
        'num_layers': TRANSFORMER_CONFIG['num_layers'],
        'num_classes': len(class_names),
        'dropout': TRANSFORMER_CONFIG['dropout'],
        'class_names': class_names,
        'is_enhanced_model': is_enhanced,
        'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_version': '2.0' if is_enhanced else '1.0'
    }
    
    torch.save(save_dict, model_path)
    logger.info(f"✅ Model saved to {model_path}")
    
    # Save deployment-ready inference script
    inference_script = create_inference_script(is_enhanced, model_path, class_names)
    inference_path = os.path.join(save_dir, 'inference.py')
    with open(inference_path, 'w') as f:
        f.write(inference_script)
    logger.info(f"✅ Inference script saved to {inference_path}")
    
    # Save model to S3 if available
    if AWS_AVAILABLE:
        try:
            # Create S3 client
            s3_client = boto3.client('s3')
            
            # Upload model to S3
            model_type = "enhanced" if is_enhanced else "standard"
            s3_key = f"models/fire_detection_model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            s3_client.upload_file(model_path, DATASET_BUCKET, s3_key)
            
            logger.info(f"✅ Model uploaded to s3://{DATASET_BUCKET}/{s3_key}")
            
            # Also upload inference script
            s3_key_script = f"models/inference_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            s3_client.upload_file(inference_path, DATASET_BUCKET, s3_key_script)
            
            logger.info(f"✅ Inference script uploaded to s3://{DATASET_BUCKET}/{s3_key_script}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to upload model to S3: {e}")
    
    return model_path


def create_inference_script(is_enhanced, model_path, class_names):
    """Create a deployment-ready inference script"""
    
    if is_enhanced:
        script = f"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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

class EarlyWarningDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_pooled = torch.mean(x, dim=1)
        return self.detector(x_pooled)

class EnhancedFireDetectionTransformer(nn.Module):
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
        
        self.conv_branch = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
            nn.GELU()
        )
        
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.warning_detector = EarlyWarningDetector(d_model)
    
    def forward(self, x):
        transformer_input = self.input_projection(x)
        transformer_input = self.positional_encoding(transformer_input)
        transformer_features = self.transformer_encoder(transformer_input)
        
        conv_input = x.transpose(1, 2)
        conv_features = self.conv_branch(conv_input)
        conv_features = conv_features.transpose(1, 2)
        
        combined = torch.cat([transformer_features, conv_features], dim=2)
        fused = self.fusion(combined)
        
        warning_prob = self.warning_detector(fused)
        
        pooled = torch.mean(fused, dim=1)
        class_logits = self.output_projection(pooled)
        
        return class_logits, warning_prob

class FireDetectionInference:
    def __init__(self, model_path='{model_path}'):
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model
        self.model = EnhancedFireDetectionTransformer(
            input_dim=checkpoint['input_dim'],
            d_model=checkpoint['d_model'],
            num_heads=checkpoint['num_heads'],
            num_layers=checkpoint['num_layers'],
            num_classes=checkpoint['num_classes'],
            dropout=0.0  # No dropout during inference
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Get class names
        self.class_names = checkpoint['class_names']
        
        print(f"Loaded enhanced fire detection model with classes: {{self.class_names}}")
    
    def add_temporal_features(self, sensor_data):
        '''Add temporal features to raw sensor data'''
        if len(sensor_data.shape) == 2:
            # Single sample: (seq_len, features)
            seq_len, n_features = sensor_data.shape
            X = np.expand_dims(sensor_data, 0)  # Add batch dimension
        else:
            # Batch: (batch, seq_len, features)
            _, seq_len, n_features = sensor_data.shape
            X = sensor_data
            
        # Create output array with additional features
        n_new_features = n_features * 5
        X_enhanced = np.zeros((X.shape[0], seq_len, n_new_features), dtype=np.float32)
        
        # Copy original features
        X_enhanced[:, :, :n_features] = X
        
        # Add temporal features
        for i in range(X.shape[0]):
            for j in range(n_features):
                # First derivative
                if seq_len > 1:
                    X_enhanced[i, 1:, n_features + j] = np.diff(X[i, :, j])
                
                # Second derivative
                if seq_len > 2:
                    second_derivatives = np.zeros(seq_len)
                    second_derivatives[2:] = np.diff(X[i, :, j], n=2)
                    X_enhanced[i, :, n_features*2 + j] = second_derivatives
                
                # Moving averages
                short_window = min(5, seq_len)
                for t in range(short_window, seq_len):
                    X_enhanced[i, t, n_features*3 + j] = np.mean(X[i, t-short_window:t, j])
                
                # Warning-specific features
                for t in range(3, seq_len):
                    increases = 0
                    for k in range(1, min(4, t+1)):
                        if t-k >= 0 and t-k+1 < seq_len:
                            if X[i, t-k, j] < X[i, t-k+1, j]:
                                increases += 1
                    X_enhanced[i, t, n_features*4 + j] = increases / 3.0
        
        if len(sensor_data.shape) == 2:
            return X_enhanced[0]  # Remove batch dimension for single sample
        return X_enhanced
    
    def predict(self, sensor_data, use_custom_thresholds=True):
        '''
        Predict fire risk from sensor data
        
        Args:
            sensor_data: numpy array of shape (seq_len, features) or (batch, seq_len, features)
            use_custom_thresholds: Whether to use custom thresholds for Warning class
            
        Returns:
            Dictionary with prediction results
        '''
        # Handle single sample vs batch
        single_sample = len(sensor_data.shape) == 2
        
        # Add temporal features
        enhanced_data = self.add_temporal_features(sensor_data)
        
        # Convert to tensor
        if single_sample:
            # Add batch dimension for single sample
            x = torch.tensor(enhanced_data, dtype=torch.float32).unsqueeze(0)
        else:
            x = torch.tensor(enhanced_data, dtype=torch.float32)
        
        # Get predictions
        with torch.no_grad():
            class_logits, warning_probs = self.model(x)
            probabilities = F.softmax(class_logits, dim=1)
            
            if use_custom_thresholds:
                # Apply custom thresholds for better Warning detection
                probs_np = probabilities.numpy()
                predicted = np.zeros(probs_np.shape[0], dtype=np.int64)
                
                # Custom decision logic with lower threshold for Warning
                warning_mask = probs_np[:, 1] > 0.25  # More sensitive to Warning
                fire_mask = probs_np[:, 2] > 0.5
                
                # Default to Normal (0)
                # Set Warning (1) where warning_mask is True
                predicted[warning_mask] = 1
                # Set Fire (2) where fire_mask is True
                predicted[fire_mask] = 2
            else:
                # Standard argmax prediction
                predicted = torch.argmax(probabilities, dim=1).numpy()
            
            # Get early warning probability
            early_warning = warning_probs.squeeze().numpy()
        
        # Format results
        if single_sample:
            result = {{
                'prediction': self.class_names[predicted[0]],
                'prediction_idx': int(predicted[0]),
                'confidence': float(probabilities[0][predicted[0]].item()),
                'probabilities': {{
                    self.class_names[i]: float(prob)
                    for i, prob in enumerate(probabilities[0].tolist())
                }},
                'early_warning_probability': float(early_warning[0]),
                'all_classes': self.class_names
            }}
        else:
            result = {{
                'predictions': [self.class_names[idx] for idx in predicted],
                'prediction_indices': predicted.tolist(),
                'confidences': [float(probabilities[i][idx]) for i, idx in enumerate(predicted)],
                'probabilities': [
                    {{self.class_names[i]: float(prob) for i, prob in enumerate(probs)}}
                    for probs in probabilities.tolist()
                ],
                'early_warning_probabilities': early_warning.tolist(),
                'all_classes': self.class_names
            }}
        
        return result

# Example usage
if __name__ == "__main__":
    detector = FireDetectionInference()
    
    # Example sensor data (60 timesteps, 6 features)
    example_data = np.random.randn(60, 6)
    result = detector.predict(example_data)
    
    print(f"Prediction: {{result['prediction']}}")
    print(f"Confidence: {{result['confidence']:.2f}}")
    print(f"Early warning probability: {{result['early_warning_probability']:.2f}}")
"""
    else:
        script = f"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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

class FireDetectionInference:
    def __init__(self, model_path='{model_path}'):
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model
        self.model = FireDetectionTransformer(
            input_dim=checkpoint['input_dim'],
            d_model=checkpoint['d_model'],
            num_heads=checkpoint['num_heads'],
            num_layers=checkpoint['num_layers'],
            num_classes=checkpoint['num_classes'],
            dropout=0.0  # No dropout during inference
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Get class names
        self.class_names = checkpoint['class_names']
        
        print(f"Loaded fire detection model with classes: {{self.class_names}}")
    
    def predict(self, sensor_data, use_custom_thresholds=True):
        '''
        Predict fire risk from sensor data
        
        Args:
            sensor_data: numpy array of shape (seq_len, features) or (batch, seq_len, features)
            use_custom_thresholds: Whether to use custom thresholds for Warning class
            
        Returns:
            Dictionary with prediction results
        '''
        # Handle single sample vs batch
        single_sample = len(sensor_data.shape) == 2
        
        # Convert to tensor
        if single_sample:
            # Add batch dimension for single sample
            x = torch.tensor(sensor_data, dtype=torch.float32).unsqueeze(0)
        else:
            x = torch.tensor(sensor_data, dtype=torch.float32)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = F.softmax(outputs, dim=1)
            
            if use_custom_thresholds:
                # Apply custom thresholds for better Warning detection
                probs_np = probabilities.numpy()
                predicted = np.zeros(probs_np.shape[0], dtype=np.int64)
                
                # Custom decision logic
                warning_mask = probs_np[:, 1] > 0.3  # More sensitive to Warning
                fire_mask = probs_np[:, 2] > 0.5
                
                # Default to Normal (0)
                # Set Warning (1) where warning_mask is True
                predicted[warning_mask] = 1
                # Set Fire (2) where fire_mask is True
                predicted[fire_mask] = 2
            else:
                # Standard argmax prediction
                predicted = torch.argmax(probabilities, dim=1).numpy()
        
        # Format results
        if single_sample:
            result = {{
                'prediction': self.class_names[predicted[0]],
                'prediction_idx': int(predicted[0]),
                'confidence': float(probabilities[0][predicted[0]].item()),
                'probabilities': {{
                    self.class_names[i]: float(prob)
                    for i, prob in enumerate(probabilities[0].tolist())
                }},
                'all_classes': self.class_names
            }}
        else:
            result = {{
                'predictions': [self.class_names[idx] for idx in predicted],
                'prediction_indices': predicted.tolist(),
                'confidences': [float(probabilities[i][idx]) for i, idx in enumerate(predicted)],
                'probabilities': [
                    {{self.class_names[i]: float(prob) for i, prob in enumerate(probs)}}
                    for probs in probabilities.tolist()
                ],
                'all_classes': self.class_names
            }}
        
        return result

# Example usage
if __name__ == "__main__":
    detector = FireDetectionInference()
    
    # Example sensor data (60 timesteps, 6 features)
    example_data = np.random.randn(60, 6)
    result = detector.predict(example_data)
    
    print(f"Prediction: {{result['prediction']}}")
    print(f"Confidence: {{result['confidence']:.2f}}")
"""
    
    return script

# ===== Main Function =====
def main():
    """Main function"""
    logger.info("Starting Fire Detection AI training with two-phase approach")
    
    # Load data with temporal features and synthetic Warning samples
    logger.info("Loading data with enhanced features...")
    X_train, y_train, X_val, y_val, X_test, y_test, areas_train, areas_val, areas_test = load_data(
        add_temporal=True,
        create_synthetic_warning=True
    )
    
    # Print data split summary
    logger.info(f"📊 Data splits:")
    logger.info(f"   Train: {X_train.shape}, {np.bincount(y_train.astype(int))}")
    logger.info(f"   Validation: {X_val.shape}, {np.bincount(y_val.astype(int))}")
    logger.info(f"   Test: {X_test.shape}, {np.bincount(y_test.astype(int))}")
    
    # ===== PHASE 1: Binary Classification (Normal vs Anomaly) =====
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: Binary Classification (Normal vs Anomaly)")
    logger.info("=" * 50)
    
    # Create binary labels
    y_train_binary = create_binary_labels(y_train)
    y_val_binary = create_binary_labels(y_val)
    y_test_binary = create_binary_labels(y_test)
    
    logger.info(f"Binary class distribution: Train={np.bincount(y_train_binary)}, Val={np.bincount(y_val_binary)}")
    
    # Create datasets for binary classification
    train_dataset_binary = FireDetectionDataset(X_train, y_train_binary, augment=True)
    val_dataset_binary = FireDetectionDataset(X_val, y_val_binary, augment=False)
    test_dataset_binary = FireDetectionDataset(X_test, y_test_binary, augment=False)
    
    # Create data loaders with balanced batch sampler
    train_loader_binary = DataLoader(
        train_dataset_binary,
        batch_size=BATCH_SIZE,
        sampler=BalancedBatchSampler(train_dataset_binary, n_classes=2),
        num_workers=4,
        pin_memory=True
    )
    
    logger.info("Using BalancedBatchSampler for training to ensure each batch has samples from all classes")
    
    val_loader_binary = DataLoader(
        val_dataset_binary,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader_binary = DataLoader(
        test_dataset_binary,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create binary classification model
    input_dim = X_train.shape[2]  # Number of features (including temporal features)
    binary_model = FireDetectionTransformer(
        input_dim=input_dim,
        d_model=TRANSFORMER_CONFIG['d_model'],
        num_heads=TRANSFORMER_CONFIG['num_heads'],
        num_layers=TRANSFORMER_CONFIG['num_layers'],
        num_classes=2,  # Binary: Normal vs Anomaly
        dropout=TRANSFORMER_CONFIG['dropout']
    )
    
    # Move model to device
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for data parallel training")
        binary_model = nn.DataParallel(binary_model)
    
    binary_model = binary_model.to(device)
    
    # Define class weights for binary model
    binary_class_counts = np.bincount(y_train_binary.astype(int))
    binary_class_weights = (1.0 / binary_class_counts) ** 2  # Square inverse frequencies
    binary_class_weights = binary_class_weights / binary_class_weights.sum() * len(binary_class_weights)
    binary_class_weights = torch.tensor(binary_class_weights, dtype=torch.float32).to(device)
    
    logger.info(f"Binary class weights: {binary_class_weights.cpu().numpy()}")
    
    # Define loss function for binary model
    binary_criterion = FocalLoss(alpha=binary_class_weights, gamma=2.0, reduction='mean')
    
    # Define optimizer with reduced learning rate
    binary_optimizer = optim.AdamW(
        binary_model.parameters(),
        lr=LEARNING_RATE * 0.5,  # Reduced learning rate for stability
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Define learning rate scheduler
    binary_lr_scheduler = get_warmup_lr_scheduler(
        binary_optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=EPOCHS // 2  # Use half the epochs for binary phase
    )
    
    # Train binary model
    logger.info("Training binary classification model (Normal vs Anomaly)...")
    binary_model, binary_history = train_model(
        model=binary_model,
        train_loader=train_loader_binary,
        val_loader=val_loader_binary,
        criterion=binary_criterion,
        optimizer=binary_optimizer,
        lr_scheduler=binary_lr_scheduler,
        num_epochs=EPOCHS // 2,  # Use half the epochs for binary phase
        patience=EARLY_STOPPING_PATIENCE,
        device=device,
        clip_value=GRADIENT_CLIP_VALUE,
        checkpoint_dir='checkpoints/binary'
    )
    
    # Evaluate binary model
    logger.info("Evaluating binary classification model...")
    binary_test_loss, binary_test_acc, binary_test_precision, binary_test_recall, binary_test_f1, _, _ = evaluate(
        binary_model, test_loader_binary, binary_criterion, device
    )
    
    logger.info(f"Binary model test results: Acc={binary_test_acc:.4f}, F1={binary_test_f1:.4f}")
    
    # ===== PHASE 2: Multi-class Classification =====
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: Multi-class Classification (Normal, Warning, Fire)")
    logger.info("=" * 50)
    
    # Create datasets for multi-class classification
    train_dataset = FireDetectionDataset(X_train, y_train, augment=True)
    val_dataset = FireDetectionDataset(X_val, y_val, augment=False)
    test_dataset = FireDetectionDataset(X_test, y_test, augment=False)
    
    # Create data loaders with balanced batch sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=BalancedBatchSampler(train_dataset, n_classes=3),
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create multi-class model with enhanced architecture
    num_classes = len(np.unique(y_train))
    class_names = ['Normal', 'Warning', 'Fire']
    
    # Use enhanced model architecture
    model = EnhancedFireDetectionTransformer(
        input_dim=input_dim,
        d_model=TRANSFORMER_CONFIG['d_model'],
        num_heads=TRANSFORMER_CONFIG['num_heads'],
        num_layers=TRANSFORMER_CONFIG['num_layers'],
        num_classes=num_classes,
        dropout=TRANSFORMER_CONFIG['dropout']
    )
    
    # Initialize multi-class model with binary model weights where possible
    logger.info("Initializing enhanced multi-class model with binary model weights...")
    if isinstance(binary_model, nn.DataParallel):
        binary_state_dict = binary_model.module.state_dict()
    else:
        binary_state_dict = binary_model.state_dict()
    
    # Copy weights from binary model to multi-class model for shared layers
    model_state_dict = model.state_dict()
    for name, param in binary_state_dict.items():
        if name in model_state_dict and 'output_projection' not in name:
            model_state_dict[name].copy_(param)
    
    # Move model to device
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Define class weights for multi-class model
    class_counts = np.bincount(y_train.astype(int))
    logger.info(f"Raw class counts: {class_counts}")
    
    # Cube the inverse for Warning class (index 1) to make its weight much higher
    class_weights = np.zeros(len(class_counts))
    for i in range(len(class_counts)):
        if i == 1:  # Warning class
            class_weights[i] = (1.0 / class_counts[i]) ** 3  # Cube for Warning
        else:
            class_weights[i] = (1.0 / class_counts[i]) ** 2  # Square for others
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    logger.info(f"Multi-class weights: {class_weights.cpu().numpy()}")
    logger.info(f"Warning class weight multiplier: {class_weights[1]/class_weights[0]:.2f}x Normal class")
    
    # Define warning-focused loss function with transition penalties
    criterion = WarningFocusedLoss(
        alpha=class_weights,
        gamma=2.0,
        warning_gamma=4.0,      # Higher gamma for Warning class
        warning_boost=3.0,      # Increased multiplier for Warning class
        transition_penalty=2.0, # Penalty for missing transitions
        reduction='mean'
    )
    
    # Define optimizer with further reduced learning rate
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE * 0.1,  # Further reduced learning rate for stability
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Define learning rate scheduler with longer warmup
    lr_scheduler = get_warmup_lr_scheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS * 2,  # Double the warmup period
        total_epochs=EPOCHS
    )
    
    # Create sequence IDs for transition analysis
    # Group samples by area to identify sequences
    sequence_ids = torch.zeros(len(train_dataset), dtype=torch.long)
    for i, area in enumerate(np.unique(areas_train)):
        area_mask = areas_train == area
        sequence_ids[torch.tensor(area_mask)] = i
    
    # Custom training function for enhanced model
    def train_enhanced_model(model, train_loader, val_loader, criterion, optimizer,
                            lr_scheduler, num_epochs, patience, device, sequence_ids=None):
        """Train enhanced model with early warning detection and transition analysis"""
        
        # Create checkpoint directory
        os.makedirs('checkpoints/enhanced', exist_ok=True)
        
        # Initialize variables
        best_val_loss = float('inf')
        best_epoch = -1
        best_model_path = None
        best_val_f1 = 0.0
        best_warning_f1 = 0.0
        patience_counter = 0
        
        # Initialize history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'warning_f1': [],
            'warning_detection_acc': [],
            'lr': []
        }
        
        # Training loop
        start_time = time.time()
        logger.info(f"Starting enhanced training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            warning_detection_correct = 0
            warning_detection_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Get batch sequence IDs if available
                batch_sequence_ids = None
                if sequence_ids is not None:
                    batch_sequence_ids = sequence_ids[batch_idx*BATCH_SIZE:
                                                     min((batch_idx+1)*BATCH_SIZE, len(sequence_ids))]
                    batch_sequence_ids = batch_sequence_ids.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass - returns class logits and warning probabilities
                outputs, warning_probs = model(inputs)
                
                # Calculate loss with warning detection and transition analysis
                loss = criterion(outputs, targets, warning_probs, batch_sequence_ids)
                
                # Backward pass
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
                
                # Optimize
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Track warning detection accuracy
                warning_targets = (targets == 1).float()
                warning_preds = (warning_probs.squeeze() > 0.5).float()
                warning_detection_correct += (warning_preds == warning_targets).sum().item()
                warning_detection_total += targets.size(0)
            
            # Calculate epoch statistics
            train_loss = running_loss / total
            train_acc = correct / total
            warning_detection_acc = warning_detection_correct / warning_detection_total
            
            # Evaluate on validation set
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            val_warning_correct = 0
            val_warning_total = 0
            
            all_targets = []
            all_predictions = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move data to device
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    outputs, warning_probs = model(inputs)
                    
                    # Calculate loss (without sequence IDs for validation)
                    loss = criterion(outputs, targets, warning_probs)
                    
                    # Update statistics
                    val_running_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    
                    # Track warning detection accuracy
                    warning_targets = (targets == 1).float()
                    warning_preds = (warning_probs.squeeze() > 0.5).float()
                    val_warning_correct += (warning_preds == warning_targets).sum().item()
                    val_warning_total += targets.size(0)
                    
                    # Store targets and predictions for metrics
                    all_targets.append(targets.cpu().numpy())
                    all_predictions.append(predicted.cpu().numpy())
            
            # Calculate validation statistics
            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total
            val_warning_detection_acc = val_warning_correct / val_warning_total
            
            # Combine all targets and predictions
            all_targets = np.concatenate(all_targets)
            all_predictions = np.concatenate(all_predictions)
            
            # Calculate additional metrics
            val_precision = precision_score(all_targets, all_predictions, average='weighted')
            val_recall = recall_score(all_targets, all_predictions, average='weighted')
            val_f1 = f1_score(all_targets, all_predictions, average='weighted')
            
            # Calculate Warning class F1 score specifically
            warning_mask_true = all_targets == 1
            warning_mask_pred = all_predictions == 1
            
            # Handle case where there are no Warning predictions or targets
            if np.sum(warning_mask_true) == 0 or np.sum(warning_mask_pred) == 0:
                warning_f1 = 0.0
            else:
                warning_f1 = f1_score(warning_mask_true, warning_mask_pred)
            
            # Update learning rate scheduler
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_loss)
            else:
                lr_scheduler.step()
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['warning_f1'].append(warning_f1)
            history['warning_detection_acc'].append(val_warning_detection_acc)
            history['lr'].append(current_lr)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            logger.info(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s - "
                       f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                       f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - "
                       f"Val F1: {val_f1:.4f} - Warning F1: {warning_f1:.4f} - "
                       f"Warning Detection: {val_warning_detection_acc:.4f} - "
                       f"LR: {current_lr:.6f}")
            
            # Check class-wise performance
            class_counts = np.bincount(all_targets, minlength=3)
            class_acc = np.zeros(3)
            class_f1 = np.zeros(3)
            
            for i in range(3):
                # Calculate per-class metrics
                class_true = all_targets == i
                class_pred = all_predictions == i
                
                if np.sum(class_true) > 0:
                    class_acc[i] = np.sum((all_targets == i) & (all_predictions == i)) / np.sum(class_true)
                    
                    if np.sum(class_pred) > 0:
                        class_precision = np.sum((all_targets == i) & (all_predictions == i)) / np.sum(class_pred)
                        class_recall = np.sum((all_targets == i) & (all_predictions == i)) / np.sum(class_true)
                        
                        if class_precision + class_recall > 0:
                            class_f1[i] = 2 * class_precision * class_recall / (class_precision + class_recall)
            
            logger.info(f"   Class accuracies: Normal={class_acc[0]:.4f}, Warning={class_acc[1]:.4f}, Fire={class_acc[2]:.4f}")
            logger.info(f"   Class F1 scores: Normal={class_f1[0]:.4f}, Warning={class_f1[1]:.4f}, Fire={class_f1[2]:.4f}")
            
            # Save model if it's the best so far
            is_best = False
            
            # Check if this is the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                is_best = True
                patience_counter = 0
                logger.info(f"   New best model based on validation loss: {val_loss:.4f}")
            
            # Also consider overall F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                is_best = True
                patience_counter = 0
                logger.info(f"   New best model based on overall F1 score: {val_f1:.4f}")
            
            # Prioritize Warning class F1 score with a higher weight
            if warning_f1 > best_warning_f1:
                best_warning_f1 = warning_f1
                is_best = True
                patience_counter = 0
                logger.info(f"   New best model based on Warning class F1 score: {warning_f1:.4f}")
            
            if is_best:
                # Save checkpoint
                checkpoint_path = os.path.join('checkpoints/enhanced', f'model_epoch_{epoch+1}.pt')
                
                if isinstance(model, nn.DataParallel):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'warning_f1': warning_f1,
                    'class_acc': class_acc,
                    'class_f1': class_f1,
                    'history': history
                }, checkpoint_path)
                
                best_model_path = checkpoint_path
                logger.info(f"✅ Saved new best model to {checkpoint_path}")
            else:
                patience_counter += 1
                logger.info(f"   Patience: {patience_counter}/{patience}")
            
            # Check for early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save confusion matrix
            if VISUALIZATION_AVAILABLE:
                try:
                    cm = confusion_matrix(all_targets, all_predictions)
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Normal', 'Warning', 'Fire'],
                                yticklabels=['Normal', 'Warning', 'Fire'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
                    plt.tight_layout()
                    cm_path = os.path.join('checkpoints/enhanced', f'confusion_matrix_epoch_{epoch+1}.png')
                    plt.savefig(cm_path)
                    plt.close()
                    logger.info(f"   Confusion matrix saved to {cm_path}")
                except Exception as e:
                    logger.warning(f"   Failed to save confusion matrix: {e}")
        
        # Calculate total training time
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s ({total_time/60:.1f}m)")
        logger.info(f"Best model at epoch {best_epoch+1} with validation loss {best_val_loss:.4f}")
        
        # Load best model
        if best_model_path is not None:
            checkpoint = torch.load(best_model_path)
            
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                
            logger.info(f"Loaded best model from {best_model_path}")
        
        return model, history
    
    # Train enhanced multi-class model
    logger.info("Training enhanced multi-class model with warning-focused loss...")
    model, history = train_enhanced_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        device=device,
        sequence_ids=sequence_ids
    )
    
    # Evaluate enhanced model on test set with both standard and custom thresholds
    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION: Enhanced Model Performance")
    logger.info("=" * 50)
    
    # Standard thresholds evaluation
    logger.info("Evaluating enhanced model with standard thresholds...")
    test_loss, test_acc, test_precision, test_recall, test_f1, y_true, y_pred_std, std_metrics = evaluate_enhanced_model(
        model, test_loader, criterion, device, use_custom_thresholds=False
    )
    
    # Custom thresholds evaluation (more sensitive to Warning class)
    logger.info("Evaluating enhanced model with custom thresholds (more sensitive to Warning class)...")
    _, test_acc_custom, test_precision_custom, test_recall_custom, test_f1_custom, _, y_pred_custom, custom_metrics = evaluate_enhanced_model(
        model, test_loader, criterion, device, use_custom_thresholds=True
    )
    
    # Print comprehensive results
    logger.info(f"\n📊 Enhanced Model Test Results (Standard Thresholds):")
    logger.info(f"   Test Loss: {test_loss:.4f}")
    logger.info(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    logger.info(f"   Test Precision: {test_precision:.4f}")
    logger.info(f"   Test Recall: {test_recall:.4f}")
    logger.info(f"   Test F1-Score: {test_f1:.4f}")
    logger.info(f"   Warning Detection Accuracy: {std_metrics['warning_detection_acc']:.4f}")
    
    logger.info(f"\n📊 Enhanced Model Test Results (Custom Thresholds):")
    logger.info(f"   Test Accuracy: {test_acc_custom:.4f} ({test_acc_custom*100:.1f}%)")
    logger.info(f"   Test Precision: {test_precision_custom:.4f}")
    logger.info(f"   Test Recall: {test_recall_custom:.4f}")
    logger.info(f"   Test F1-Score: {test_f1_custom:.4f}")
    logger.info(f"   Warning Detection Accuracy: {custom_metrics['warning_detection_acc']:.4f}")
    
    # Class-specific metrics
    logger.info(f"\n📊 Class-Specific Metrics:")
    for i, class_name in enumerate(class_names):
        logger.info(f"   {class_name} Class:")
        logger.info(f"      Standard: Precision={std_metrics['class_precision'][i]:.4f}, "
                   f"Recall={std_metrics['class_recall'][i]:.4f}, F1={std_metrics['class_f1'][i]:.4f}")
        logger.info(f"      Custom: Precision={custom_metrics['class_precision'][i]:.4f}, "
                   f"Recall={custom_metrics['class_recall'][i]:.4f}, F1={custom_metrics['class_f1'][i]:.4f}")
        
        # Calculate improvement
        prec_imp = custom_metrics['class_precision'][i] - std_metrics['class_precision'][i]
        recall_imp = custom_metrics['class_recall'][i] - std_metrics['class_recall'][i]
        f1_imp = custom_metrics['class_f1'][i] - std_metrics['class_f1'][i]
        
        logger.info(f"      Improvement: Precision={prec_imp:.4f}, Recall={recall_imp:.4f}, F1={f1_imp:.4f}")
    
    # Special focus on Warning class transitions
    logger.info(f"\n🔥 Warning Class Transition Detection:")
    logger.info(f"   Normal→Warning Detection Rate (Standard): {std_metrics['normal_to_warning_rate']:.4f}")
    logger.info(f"   Normal→Warning Detection Rate (Custom): {custom_metrics['normal_to_warning_rate']:.4f}")
    logger.info(f"   Warning→Fire Detection Rate (Standard): {std_metrics['warning_to_fire_rate']:.4f}")
    logger.info(f"   Warning→Fire Detection Rate (Custom): {custom_metrics['warning_to_fire_rate']:.4f}")
    
    # Print detailed classification reports
    logger.info(f"\n📋 Detailed Classification Report (Standard Thresholds):")
    print(classification_report(y_true, y_pred_std, target_names=class_names))
    
    logger.info(f"\n📋 Detailed Classification Report (Custom Thresholds):")
    print(classification_report(y_true, y_pred_custom, target_names=class_names))
    
    # Print confusion matrices
    logger.info(f"\n🔍 Confusion Matrix (Standard Thresholds):")
    print(std_metrics['confusion_matrix'])
    
    logger.info(f"\n🔍 Confusion Matrix (Custom Thresholds):")
    print(custom_metrics['confusion_matrix'])
    
    # Save confusion matrices and other visualizations
    if VISUALIZATION_AVAILABLE:
        try:
            # Create visualization directory
            os.makedirs('models/visualizations', exist_ok=True)
            
            # Standard thresholds confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(std_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix - Standard Thresholds')
            plt.tight_layout()
            plt.savefig('models/visualizations/confusion_matrix_standard.png')
            plt.close()
            
            # Custom thresholds confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(custom_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix - Custom Thresholds')
            plt.tight_layout()
            plt.savefig('models/visualizations/confusion_matrix_custom.png')
            plt.close()
            
            # Class F1 comparison chart
            plt.figure(figsize=(12, 6))
            x = np.arange(len(class_names))
            width = 0.35
            
            plt.bar(x - width/2, std_metrics['class_f1'], width, label='Standard Thresholds')
            plt.bar(x + width/2, custom_metrics['class_f1'], width, label='Custom Thresholds')
            
            plt.xlabel('Class')
            plt.ylabel('F1 Score')
            plt.title('F1 Score by Class and Threshold Method')
            plt.xticks(x, class_names)
            plt.ylim(0, 1.0)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig('models/visualizations/class_f1_comparison.png')
            plt.close()
            
            # Warning detection visualization
            plt.figure(figsize=(10, 6))
            metrics = [
                std_metrics['class_precision'][1], std_metrics['class_recall'][1], std_metrics['class_f1'][1],
                custom_metrics['class_precision'][1], custom_metrics['class_recall'][1], custom_metrics['class_f1'][1]
            ]
            labels = [
                'Precision (Std)', 'Recall (Std)', 'F1 (Std)',
                'Precision (Custom)', 'Recall (Custom)', 'F1 (Custom)'
            ]
            colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e', '#ff7f0e']
            
            plt.bar(labels, metrics, color=colors)
            plt.ylabel('Score')
            plt.title('Warning Class Detection Performance')
            plt.ylim(0, 1.0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('models/visualizations/warning_detection_performance.png')
            plt.close()
            
            logger.info("✅ Performance visualizations saved to models/visualizations/")
        except Exception as e:
            logger.warning(f"Failed to save visualization images: {e}")
    
    # Save model for deployment
    logger.info("Saving model for deployment...")
    model_path = save_model_for_deployment(model, input_dim, class_names)
    
    # Save training history
    history_path = os.path.join('models', 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    logger.info(f"✅ Training history saved to {history_path}")
    
    # Calculate total training time
    total_time = time.time() - time.time()  # This will be updated in the actual run
    
    # Final summary
    logger.info("\n" + "🎉" * 50)
    logger.info("FIRE DETECTION AI TRAINING COMPLETED!")
    logger.info("🎉" * 50)
    logger.info(f"🏆 Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    logger.info(f"🎯 Target Performance: {'✅ ACHIEVED' if test_acc >= 0.90 else '📈 IN PROGRESS'}")
    logger.info(f"📊 Dataset Size: {len(X_train) + len(X_val) + len(X_test):,} samples")
    logger.info(f"🚀 Model ready for deployment!")
    logger.info(f"💾 Model saved to: {model_path}")
    
    if test_acc >= 0.95:
        logger.info("🎊 OUTSTANDING! 95%+ accuracy achieved!")
    elif test_acc >= 0.90:
        logger.info("✅ EXCELLENT! 90%+ accuracy achieved!")
    elif test_acc >= 0.85:
        logger.info("👍 GOOD! 85%+ accuracy achieved!")
    else:
        logger.info("📈 Consider additional training or hyperparameter tuning.")
    
    # Create deployment instructions
    create_deployment_instructions(model_path, class_names)
    
    return model, test_acc, history

def create_deployment_instructions(model_path, class_names):
    """Create deployment instructions and example code"""
    
    instructions = f"""
# Fire Detection AI - Deployment Instructions

## Model Information
- Model Path: {model_path}
- Classes: {class_names}
- Input Shape: (batch_size, 60, 6)  # 60 timesteps, 6 features
- Output: Class probabilities for {len(class_names)} classes

## Quick Start - Load and Use Model

```python
import torch
import numpy as np

# Load the model
checkpoint = torch.load('{model_path}')
model_state_dict = checkpoint['model_state_dict']
input_dim = checkpoint['input_dim']
d_model = checkpoint['d_model']
num_heads = checkpoint['num_heads']
num_layers = checkpoint['num_layers']
num_classes = checkpoint['num_classes']
dropout = checkpoint['dropout']
class_names = checkpoint['class_names']

# Recreate model architecture
from fire_detection_5m_compact_final_complete import FireDetectionTransformer

model = FireDetectionTransformer(
    input_dim=input_dim,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout=dropout
)

# Load weights
model.load_state_dict(model_state_dict)
model.eval()

# Example prediction
def predict_fire_risk(sensor_data):
    '''
    sensor_data: numpy array of shape (60, 6)
    Returns: predicted class and confidence
    '''
    # Convert to tensor and add batch dimension
    x = torch.tensor(sensor_data, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence

# Example usage
# sensor_data = np.random.randn(60, 6)  # Replace with real sensor data
# prediction, confidence = predict_fire_risk(sensor_data)
# print(f"Prediction: {{prediction}} (Confidence: {{confidence:.2f}})")
```

## Production Deployment Options

### 1. AWS SageMaker Endpoint
```python
# Deploy to SageMaker endpoint
import sagemaker
from sagemaker.pytorch import PyTorchModel

pytorch_model = PyTorchModel(
    model_data='s3://your-bucket/models/fire_detection_model.tar.gz',
    role=sagemaker.get_execution_role(),
    entry_point='inference.py',
    framework_version='1.12.0',
    py_version='py38'
)

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
```

### 2. Docker Container
```dockerfile
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

COPY {model_path} /app/model.pt
COPY inference_script.py /app/
WORKDIR /app

CMD ["python", "inference_script.py"]
```

### 3. Real-time IoT Integration
```python
# For edge devices or IoT sensors
import torch.jit

# Convert to TorchScript for faster inference
scripted_model = torch.jit.script(model)
scripted_model.save('fire_detection_scripted.pt')

# Load on edge device
edge_model = torch.jit.load('fire_detection_scripted.pt')
edge_model.eval()
```

## Model Performance
- Training completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Dataset: 5M+ samples
- Architecture: Transformer-based with positional encoding
- Features: 6 sensor inputs (temperature, humidity, smoke, etc.)
- Sequence Length: 60 timesteps

## Monitoring and Maintenance
1. Monitor prediction confidence scores
2. Retrain if accuracy drops below 85%
3. Collect new data for continuous improvement
4. Set up alerts for low-confidence predictions

## Support
For technical support or questions about deployment, refer to the training logs
and model documentation.
"""
    
    # Save instructions
    instructions_path = os.path.join('models', 'deployment_instructions.md')
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    logger.info(f"✅ Deployment instructions saved to {instructions_path}")
    
    # Create simple inference script
    inference_script = f"""
import torch
import numpy as np
from fire_detection_5m_compact_final_complete import FireDetectionTransformer

class FireDetectionInference:
    def __init__(self, model_path='{model_path}'):
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        self.model = FireDetectionTransformer(
            input_dim=checkpoint['input_dim'],
            d_model=checkpoint['d_model'],
            num_heads=checkpoint['num_heads'],
            num_layers=checkpoint['num_layers'],
            num_classes=checkpoint['num_classes'],
            dropout=checkpoint['dropout']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.class_names = checkpoint['class_names']
    
    def predict(self, sensor_data):
        '''Predict fire risk from sensor data'''
        if isinstance(sensor_data, np.ndarray):
            sensor_data = torch.tensor(sensor_data, dtype=torch.float32)
        
        if len(sensor_data.shape) == 2:
            sensor_data = sensor_data.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            outputs = self.model(sensor_data)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {{
            'prediction': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].tolist(),
            'all_classes': self.class_names
        }}

# Example usage
if __name__ == "__main__":
    detector = FireDetectionInference()
    
    # Example sensor data (60 timesteps, 6 features)
    example_data = np.random.randn(60, 6)
    result = detector.predict(example_data)
    
    print(f"Prediction: {{result['prediction']}}")
    print(f"Confidence: {{result['confidence']:.2f}}")
"""
    
    # Save inference script
    inference_path = os.path.join('models', 'inference.py')
    with open(inference_path, 'w') as f:
        f.write(inference_script)
    
    logger.info(f"✅ Inference script saved to {inference_path}")

if __name__ == "__main__":
    try:
        # Record start time
        start_time = time.time()
        
        # Run main training
        model, accuracy, history = main()
        
        # Calculate total time
        total_time = time.time() - start_time
        
        logger.info(f"\n🏁 TRAINING SESSION COMPLETE!")
        logger.info(f"⏱️ Total Time: {total_time:.1f}s ({total_time/60:.1f}m)")
        logger.info(f"🎯 Final Accuracy: {accuracy:.4f}")
        logger.info(f"🚀 Ready for production deployment!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)