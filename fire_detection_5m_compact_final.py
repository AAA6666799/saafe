"""
Fire Detection AI - 5M Dataset Training (Compact Version)

This is a compact version of the fire detection training script optimized for AWS SageMaker
with an ml.p3.16xlarge instance (8 NVIDIA V100 GPUs). It focuses on the essential components
for training on the 5M dataset.

Instructions:
1. Upload this file to your SageMaker notebook instance
2. Run the script with: python fire_detection_5m_compact_final.py
3. The trained model will be saved to the 'models' directory
"""

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

# Machine learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# AWS libraries
try:
    import boto3
    import sagemaker
    AWS_AVAILABLE = True
    print("✅ AWS libraries available")
except ImportError:
    AWS_AVAILABLE = False
    print("❌ AWS libraries not available - install with: pip install boto3 sagemaker")

# ===== Configuration =====
# Dataset configuration
DATASET_BUCKET = "synthetic-data-4"
DATASET_PREFIX = "datasets/"
SAMPLE_SIZE = 5000000  # 5M samples
RANDOM_SEED = 42

# Training configuration
EPOCHS = 50
BATCH_SIZE = 256
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 0.002

# Model configuration
TRANSFORMER_CONFIG = {
    'd_model': 128,
    'num_heads': 4,
    'num_layers': 3,
    'dropout': 0.1
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
        
        return X, y, areas
        
    except Exception as e:
        logger.error(f"Error loading data from S3: {e}")
        return None, None, None

def create_synthetic_data(n_samples=50000):
    """Create synthetic data for demonstration"""
    logger.info(f"Creating synthetic data with {n_samples} samples")
    
    # Create features
    X = np.random.randn(n_samples, 60, 6).astype(np.float32)
    
    # Create labels (0: normal, 1: warning, 2: fire)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
    
    # Create area IDs
    areas = np.random.choice(range(5), size=n_samples)
    
    logger.info(f"✅ Created synthetic data: X={X.shape}, y={y.shape}, areas={areas.shape}")
    
    return X, y, areas

def load_data():
    """Load data from various sources"""
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
        
        return X_train, y_train, X_val, y_val, X_test, y_test, areas_train, areas_val, areas_test
    
    # If numpy files don't exist, try loading from S3
    elif AWS_AVAILABLE:
        X, y, areas = load_data_from_s3()
        
        if X is None:
            # Fall back to synthetic data
            X, y, areas = create_synthetic_data()
    else:
        # Create synthetic data
        X, y, areas = create_synthetic_data()
    
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
    
    return X_train, y_train, X_val, y_val, X_test, y_test, areas_train, areas_val, areas_test

# ===== Dataset and DataLoader =====
class FireDetectionDataset(Dataset):
    """PyTorch Dataset for Fire Detection data"""
    
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
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
        
        # Backward pass and optimize
        loss.backward()
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

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataloader"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
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
    
    return epoch_loss, epoch_acc, precision, recall, f1, all_targets, all_predictions

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, patience, device, checkpoint_dir='checkpoints'):
    """Train model with early stopping"""
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize variables
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_path = None
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'lr': []
    }
    
    # Training loop
    start_time = time.time()
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on validation set
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
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
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'history': history
            }, checkpoint_path)
            
            best_model_path = checkpoint_path
            logger.info(f"✅ Saved new best model to {checkpoint_path}")
        
        # Check for early stopping
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
    
    return model_path

# ===== Main Function =====
def main():
    """Main function"""
    logger.info("Starting Fire Detection AI training")
    
    # Load data
    logger.info("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, areas_train, areas_val, areas_test = load_data()
    
    # Print data split summary
    logger.info(f"📊 Data splits:")
    logger.info(f"   Train: {X_train.shape}, {np.bincount(y_train.astype(int))}")
    logger.info(f"   Validation: {X_val.shape}, {np.bincount(y_val.astype(int))}")
    logger.info(f"   Test: {X_test.shape}, {np.bincount(y_test.astype(int))}")
    
    # Create datasets
    train_dataset = FireDetectionDataset(X_train, y_train)
    val_dataset = FireDetectionDataset(X_val, y_val)
    test_dataset = FireDetectionDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
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
    
    logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches")
    
    # Create model
    input_dim = X_train.shape[2]  # Number of features
    num_classes = len(np.unique(y_train))
    class_names = ['Normal', 'Warning', 'Fire']
    
    model = FireDetectionTransformer(
        input_dim=input_dim,
        d_model=TRANSFORMER_CONFIG['d_model'],
        num_heads=TRANSFORMER_CONFIG['num_heads'],
        num_layers=TRANSFORMER_CONFIG['num_layers'],
        num_classes=num_classes,
        dropout=TRANSFORMER_CONFIG['dropout']
    )
    
    # Move model to device
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for data parallel training")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Print model summary
    logger.info(f"Model architecture:")
    logger.info(f"{model}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Define class weights to handle imbalance
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # Train model
    logger.info("Starting model training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        device=device
    )
    
    # Evaluate model on test set
    logger.info