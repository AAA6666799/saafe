# Script to load data from S3 and sample it to create a 5M dataset

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
SAMPLE_SIZE = 5000000  # 5M samples
RANDOM_SEED = 42

def list_s3_files(bucket, prefix):
    """List all files in the S3 bucket with the given prefix"""
    s3_client = boto3.client('s3')
    
    logger.info(f"Listing files in s3://{bucket}/{prefix}")
    
    # List objects in the bucket
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    if 'Contents' not in response:
        logger.error(f"No files found in s3://{bucket}/{prefix}")
        return []
    
    files = [obj['Key'] for obj in response['Contents'] if not obj['Key'].endswith('/')]
    
    # Continue listing if there are more files
    while response.get('IsTruncated', False):
        continuation_token = response.get('NextContinuationToken')
        response = s3_client.list_objects_v2(
            Bucket=bucket, 
            Prefix=prefix,
            ContinuationToken=continuation_token
        )
        files.extend([obj['Key'] for obj in response['Contents'] if not obj['Key'].endswith('/')])
    
    logger.info(f"Found {len(files)} files in s3://{bucket}/{prefix}")
    return files

def sample_from_s3(bucket, files, total_sample_size):
    """Sample data from S3 files to create a dataset of the specified size"""
    s3_client = boto3.client('s3')
    
    # Calculate how many samples to take from each file
    samples_per_file = total_sample_size // len(files)
    if samples_per_file == 0:
        samples_per_file = 1
    
    logger.info(f"Sampling approximately {samples_per_file} rows from each of {len(files)} files")
    
    all_data = []
    all_labels = []
    all_areas = []
    
    for i, file_key in enumerate(files):
        try:
            logger.info(f"Processing file {i+1}/{len(files)}: {file_key}")
            
            # Get file from S3
            response = s3_client.get_object(Bucket=bucket, Key=file_key)
            
            # Extract area from filename
            area_name = file_key.split('/')[-1].split('_')[0]
            area_idx = hash(area_name) % 5  # Map to 5 areas
            
            # Read file in chunks
            chunk_size = 100000  # 100K rows per chunk
            chunks = []
            
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
    if len(X) > total_sample_size:
        indices = np.random.choice(len(X), total_sample_size, replace=False)
        X = X[indices]
        y = y[indices]
        areas = areas[indices]
    
    logger.info(f"Final dataset: X={X.shape}, y={y.shape}, areas={areas.shape}")
    return X, y, areas

def create_sequences(X, y, areas, seq_len=60, step=10):
    """Create time series sequences"""
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
    
    logger.info(f"Total sequences: X={X.shape}, y={y.shape}, areas={areas.shape}")
    return X, y, areas

def main():
    """Main function to load and prepare data"""
    start_time = time.time()
    
    logger.info("Starting data loading and preparation")
    
    # List files in S3
    files = list_s3_files(S3_BUCKET, S3_PREFIX)
    
    if not files:
        logger.error("No files found. Exiting.")
        return
    
    # Sample data from S3
    X, y, areas = sample_from_s3(S3_BUCKET, files, SAMPLE_SIZE)
    
    # Create sequences
    X_seq, y_seq, areas_seq = create_sequences(X, y, areas)
    
    # Split into train/validation/test sets
    X_train, X_test, y_train, y_test, areas_train, areas_test = train_test_split(
        X_seq, y_seq, areas_seq, test_size=0.2, random_state=RANDOM_SEED, stratify=y_seq
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

if __name__ == "__main__":
    main()