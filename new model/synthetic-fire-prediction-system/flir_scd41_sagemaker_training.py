#!/usr/bin/env python3
"""
SageMaker Compatible Training Script for FLIR+SCD41 Fire Detection
This script is designed to work with SageMaker's training container requirements.
"""

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
    
    return model, scaler, accuracy

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

def convert_to_csv_format(data, output_path):
    """Convert JSON data to CSV format for SageMaker."""
    print(f"Converting data to CSV format: {output_path}")
    
    # Prepare data for CSV
    rows = []
    for sample in data['samples']:
        features = sample['features']
        label = sample['label']
        
        # Create row with label first (SageMaker convention)
        row = [label] + [
            features['t_mean'], features['t_std'], features['t_max'], features['t_p95'],
            features['t_hot_area_pct'], features['t_hot_largest_blob_pct'],
            features['t_grad_mean'], features['t_grad_std'],
            features['t_diff_mean'], features['t_diff_std'],
            features['flow_mag_mean'], features['flow_mag_std'],
            features['tproxy_val'], features['tproxy_delta'], features['tproxy_vel'],
            features['gas_val'], features['gas_delta'], features['gas_vel']
        ]
        rows.append(row)
    
    # Create DataFrame
    columns = ['label'] + [
        't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
        't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
        't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
        'tproxy_val', 'tproxy_delta', 'tproxy_vel',
        'gas_val', 'gas_delta', 'gas_vel'
    ]
    
    df = pd.DataFrame(rows, columns=columns)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"CSV data saved to {output_path}")
    print(f"CSV shape: {df.shape}")

if __name__ == '__main__':
    # Parse arguments - these are provided by SageMaker
    parser = argparse.ArgumentParser()
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS', '[]')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST', 'local'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data'))
    parser.add_argument('--num-gpus', type=int, default=os.environ.get('SM_NUM_GPUS', 0))
    
    args = parser.parse_args()
    
    try:
        print("Starting FLIR+SCD41 Fire Detection Training")
        print("=" * 50)
        
        # Load data
        data = load_data(args.data_dir)
        
        # Convert to CSV for algorithm mode
        csv_path = os.path.join(args.data_dir, 'training_data.csv')
        convert_to_csv_format(data, csv_path)
        
        # Prepare features for sklearn mode
        X, y = prepare_features(data)
        
        # Train model
        model, scaler, accuracy = train_model(X, y)
        
        # Save model
        save_model(model, scaler, args.model_dir)
        
        # Save training metrics
        metrics_path = os.path.join(args.model_dir, 'evaluation.json')
        metrics = {
            'accuracy': accuracy,
            'num_samples': len(data['samples']),
            'num_features': X.shape[1]
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("Training completed successfully")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise