#!/usr/bin/env python3
"""
Improved SageMaker Training Job for FLIR+SCD41 Fire Detection
This script creates properly configured SageMaker training jobs for the fire detection system.
"""

import boto3
import json
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET = 'fire-detection-training-691595239825'
S3_PREFIX = 'flir_scd41_training'

def create_improved_training_job():
    """Create an improved SageMaker training job with proper configuration."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Generate unique job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"flir-scd41-improved-training-{timestamp}"
    
    # Define training job parameters with proper configuration
    training_params = {
        'TrainingJobName': job_name,
        'RoleArn': 'arn:aws:iam::691595239825:role/SageMakerExecutionRole',
        'AlgorithmSpecification': {
            'TrainingInputMode': 'File',
            'TrainingImage': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3'
        },
        'InputDataConfig': [
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://{S3_BUCKET}/{S3_PREFIX}/data/',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                }
            }
        ],
        'OutputDataConfig': {
            'S3OutputPath': f's3://{S3_BUCKET}/{S3_PREFIX}/models/'
        },
        'ResourceConfig': {
            'InstanceType': 'ml.m5.2xlarge',  # More powerful instance for full-scale training
            'InstanceCount': 1,
            'VolumeSizeInGB': 50  # Larger volume for larger datasets
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 7200  # 2 hours for full-scale training
        },
        'HyperParameters': {
            'sagemaker_submit_directory': f's3://{S3_BUCKET}/{S3_PREFIX}/code/sagemaker_submit_directory.tar.gz',
            'sagemaker_program': 'flir_scd41_training.py',
            'epochs': '100',
            'batch_size': '64'
        }
    }
    
    try:
        # Create training job
        response = sagemaker.create_training_job(**training_params)
        print(f"Training job created successfully: {job_name}")
        print(f"Training job ARN: {response['TrainingJobArn']}")
        return job_name
    except Exception as e:
        print(f"Error creating training job: {e}")
        return None

def create_xgboost_training_job():
    """Create a specialized XGBoost training job for fire detection."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Generate unique job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"flir-scd41-xgboost-training-{timestamp}"
    
    # Define training job parameters for XGBoost
    training_params = {
        'TrainingJobName': job_name,
        'RoleArn': 'arn:aws:iam::691595239825:role/SageMakerExecutionRole',
        'AlgorithmSpecification': {
            'TrainingInputMode': 'File',
            'TrainingImage': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-xgboost:1.5-1-cpu-py3'
        },
        'InputDataConfig': [
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://{S3_BUCKET}/{S3_PREFIX}/data/',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                }
            }
        ],
        'OutputDataConfig': {
            'S3OutputPath': f's3://{S3_BUCKET}/{S3_PREFIX}/models/'
        },
        'ResourceConfig': {
            'InstanceType': 'ml.m5.2xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 50
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 7200  # 2 hours
        },
        'HyperParameters': {
            'max_depth': '5',
            'eta': '0.2',
            'gamma': '4',
            'min_child_weight': '6',
            'subsample': '0.8',
            'objective': 'binary:logistic',
            'num_round': '100'
        }
    }
    
    try:
        # Create training job
        response = sagemaker.create_training_job(**training_params)
        print(f"XGBoost training job created successfully: {job_name}")
        print(f"Training job ARN: {response['TrainingJobArn']}")
        return job_name
    except Exception as e:
        print(f"Error creating XGBoost training job: {e}")
        return None

def monitor_training_job(job_name):
    """Monitor the status of a training job."""
    
    if not job_name:
        print("No job name provided")
        return
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    try:
        # Get training job status
        response = sagemaker.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        print(f"Training job status: {status}")
        
        if 'SecondaryStatus' in response:
            print(f"Secondary status: {response['SecondaryStatus']}")
            
        if 'FailureReason' in response:
            print(f"Failure reason: {response['FailureReason']}")
            
        return status
    except Exception as e:
        print(f"Error monitoring training job: {e}")
        return None

def list_recent_training_jobs():
    """List recent training jobs."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    try:
        # List training jobs
        response = sagemaker.list_training_jobs(
            MaxResults=10,
            SortBy='CreationTime',
            SortOrder='Descending'
        )
        
        print("Recent training jobs:")
        for job in response['TrainingJobSummaries']:
            print(f"  - {job['TrainingJobName']}: {job['TrainingJobStatus']}")
            
    except Exception as e:
        print(f"Error listing training jobs: {e}")

def create_training_script():
    """Create a training script for the FLIR+SCD41 fire detection system."""
    
    training_script = '''
import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(data_dir):
    """Load training data from S3."""
    print(f"Loading data from {data_dir}")
    
    # Find JSON files in the directory
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not data_files:
        raise ValueError(f"No JSON files found in {data_dir}")
    
    # Load the first data file (assuming it's our demo data)
    data_file = data_files[0]
    with open(os.path.join(data_dir, data_file), 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['samples'])} samples")
    return data

def prepare_features(data):
    """Prepare features and labels for training."""
    print("Preparing features and labels")
    
    # Extract features and labels
    features_list = []
    labels_list = []
    
    for sample in data['samples']:
        features = sample['features']
        label = sample['label']
        
        # Convert features to list in the correct order
        feature_values = [
            features['t_mean'], features['t_std'], features['t_max'], features['t_p95'],
            features['t_hot_area_pct'], features['t_hot_largest_blob_pct'],
            features['t_grad_mean'], features['t_grad_std'],
            features['t_diff_mean'], features['t_diff_std'],
            features['flow_mag_mean'], features['flow_mag_std'],
            features['tproxy_val'], features['tproxy_delta'], features['tproxy_vel'],
            features['gas_val'], features['gas_delta'], features['gas_vel']
        ]
        
        features_list.append(feature_values)
        labels_list.append(label)
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label array shape: {y.shape}")
    
    return X, y

def train_model(X, y):
    """Train a Random Forest classifier."""
    print("Training Random Forest model")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler

def save_model(model, scaler, model_dir):
    """Save the trained model and scaler."""
    print(f"Saving model to {model_dir}")
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    # Save model info
    model_info = {
        'model_type': 'RandomForestClassifier',
        'features': [
            't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
            't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
            't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
            'tproxy_val', 'tproxy_delta', 'tproxy_vel',
            'gas_val', 'gas_delta', 'gas_vel'
        ],
        'num_features': 18
    }
    
    info_path = os.path.join(model_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("Model saved successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS', '[]')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST', 'local'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data'))
    parser.add_argument('--num-gpus', type=int, default=os.environ.get('SM_NUM_GPUS', 0))
    
    args = parser.parse_args()
    
    try:
        # Load data
        data = load_data(args.data_dir)
        
        # Prepare features
        X, y = prepare_features(data)
        
        # Train model
        model, scaler = train_model(X, y)
        
        # Save model
        save_model(model, scaler, args.model_dir)
        
        print("Training completed successfully")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
'''
    
    # Save the training script
    script_path = "/tmp/flir_scd41_training.py"
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    print(f"Training script saved to {script_path}")
    
    # Upload to S3
    s3 = boto3.client('s3', region_name=AWS_REGION)
    s3_key = f"{S3_PREFIX}/code/flir_scd41_training.py"
    
    try:
        s3.upload_file(script_path, S3_BUCKET, s3_key)
        print(f"Training script uploaded to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"Error uploading training script: {e}")
    
    return script_path

if __name__ == "__main__":
    print("FLIR+SCD41 Fire Detection - Improved SageMaker Training")
    print("=" * 55)
    
    # List recent training jobs
    list_recent_training_jobs()
    
    # Create training script
    print("\nCreating training script...")
    script_path = create_training_script()
    
    # Create improved training jobs
    print("\nCreating improved training jobs...")
    
    # Create Random Forest training job
    rf_job = create_improved_training_job()
    
    # Create XGBoost training job
    xgb_job = create_xgboost_training_job()
    
    print(f"\nRandom Forest job: {rf_job}")
    print(f"XGBoost job: {xgb_job}")
    
    print("\nFor monitoring, use:")
    if rf_job:
        print(f"aws sagemaker describe-training-job --training-job-name {rf_job} --region {AWS_REGION}")
    if xgb_job:
        print(f"aws sagemaker describe-training-job --training-job-name {xgb_job} --region {AWS_REGION}")