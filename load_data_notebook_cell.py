# Cell to load data from S3 and prepare it for training
# Copy and paste this into your SageMaker notebook

import boto3
import pandas as pd
import numpy as np
import time
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# S3 configuration
S3_BUCKET = "synthetic-data-4"
S3_PREFIX = "datasets/"
SAMPLE_SIZE = 5000000  # 5M samples (reduce if memory issues)
RANDOM_SEED = 42

# Start timer
start_time = time.time()
logger.info("Starting data loading and preparation")

# List files in S3
s3_client = boto3.client('s3')
logger.info(f"Listing files in s3://{S3_BUCKET}/{S3_PREFIX}")
response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)

if 'Contents' not in response:
    logger.error(f"No files found in s3://{S3_BUCKET}/{S3_PREFIX}")
    files = []
else:
    files = [obj['Key'] for obj in response['Contents'] if not obj['Key'].endswith('/')]
    
    # Continue listing if there are more files
    while response.get('IsTruncated', False):
        continuation_token = response.get('NextContinuationToken')
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET, 
            Prefix=S3_PREFIX,
            ContinuationToken=continuation_token
        )
        files.extend([obj['Key'] for obj in response['Contents'] if not obj['Key'].endswith('/')])

logger.info(f"Found {len(files)} files in s3://{S3_BUCKET}/{S3_PREFIX}")

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
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
        
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

# Print data split summary
logger.info(f"Data splits:")
logger.info(f"  Train: {X_train.shape}, {np.bincount(y_train.astype(int))}")
logger.info(f"  Validation: {X_val.shape}, {np.bincount(y_val.astype(int))}")
logger.info(f"  Test: {X_test.shape}, {np.bincount(y_test.astype(int))}")

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

total_time = time.time() - start_time
logger.info(f"Total time: {total_time:.2f} seconds")

# Display class distribution
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot
plt.figure(figsize=(15, 6))
class_names = ['Normal', 'Warning', 'Fire']

# Count classes in each set
train_counts = np.bincount(y_train.astype(int), minlength=3)
val_counts = np.bincount(y_val.astype(int), minlength=3)
test_counts = np.bincount(y_test.astype(int), minlength=3)

# Plot counts
plt.subplot(1, 2, 1)
plt.bar(class_names, train_counts, alpha=0.7, label='Train')
plt.bar(class_names, val_counts, alpha=0.7, label='Validation')
plt.bar(class_names, test_counts, alpha=0.7, label='Test')
plt.title('Class Distribution')
plt.ylabel('Count')
plt.legend()

# Plot percentages
plt.subplot(1, 2, 2)
train_pct = train_counts / train_counts.sum() * 100
val_pct = val_counts / val_counts.sum() * 100
test_pct = test_counts / test_counts.sum() * 100

plt.bar(class_names, train_pct, alpha=0.7, label='Train')
plt.bar(class_names, val_pct, alpha=0.7, label='Validation')
plt.bar(class_names, test_pct, alpha=0.7, label='Test')
plt.title('Class Distribution (%)')
plt.ylabel('Percentage')
plt.legend()

plt.tight_layout()
plt.show()