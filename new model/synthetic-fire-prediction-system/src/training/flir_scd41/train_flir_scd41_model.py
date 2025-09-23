#!/usr/bin/env python3
"""
FLIR+SCD41 Training Script for SageMaker
Handles model training with FLIR Lepton 3.5 + SCD41 CO₂ sensor data.
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.feature_engineering.extractors.flir_thermal_extractor import FlirThermalExtractor
from src.feature_engineering.extractors.scd41_gas_extractor import Scd41GasExtractor


class FlirScd41Dataset(Dataset):
    """Dataset for FLIR+SCD41 fire detection."""
    
    def __init__(self, data, labels, scaler=None):
        self.data = data
        self.labels = labels
        
        if scaler is None:
            self.scaler = StandardScaler()
            self.data_scaled = self.scaler.fit_transform(data)
        else:
            self.scaler = scaler
            self.data_scaled = self.scaler.transform(data)
    
    def __len__(self):
        return len(self.data_scaled)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data_scaled[idx]), torch.LongTensor([self.labels[idx]])


class FlirScd41LSTM(nn.Module):
    """LSTM-based fire classifier for FLIR+SCD41 data."""
    
    def __init__(self, input_size=18, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # For single time step, we need to add sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        output = lstm_out[:, -1, :]
        output = self.dropout(output)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        
        return output


class FlirScd41XGBoost:
    """XGBoost classifier for FLIR+SCD41 data."""
    
    def __init__(self, **params):
        self.params = {
            'max_depth': 6,
            'eta': 0.3,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'random_state': 42,
            'verbosity': 1
        }
        self.params.update(params)
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_scaled, label=y)
        
        # Train model
        self.model = xgb.train(self.params, dtrain, num_boost_round=100)
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create DMatrix
        dtest = xgb.DMatrix(X_scaled)
        
        # Make predictions
        pred_proba = self.model.predict(dtest)
        return (pred_proba > 0.5).astype(int)
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create DMatrix
        dtest = xgb.DMatrix(X_scaled)
        
        # Make predictions
        return self.model.predict(dtest)


def load_flir_scd41_data(data_dir, file_pattern='*.csv'):
    """Load FLIR+SCD41 data from directory."""
    import glob
    
    # Find CSV files in directory
    csv_files = glob.glob(os.path.join(data_dir, file_pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    # Load and concatenate all CSV files
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)
    
    # Concatenate all dataframes
    data = pd.concat(dataframes, ignore_index=True)
    
    # Ensure we have the expected 18 features (15 thermal + 3 gas)
    expected_features = [
        't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
        't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
        't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
        'tproxy_val', 'tproxy_delta', 'tproxy_vel',
        'gas_val', 'gas_delta', 'gas_vel'
    ]
    
    # Check if we have the expected features
    available_features = [col for col in data.columns if col in expected_features]
    missing_features = set(expected_features) - set(available_features)
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    # Select available features
    X = data[available_features].values
    y = data['fire_detected'].values if 'fire_detected' in data.columns else np.zeros(len(X))
    
    return X, y, available_features


def train_xgboost_model(X_train, y_train, X_val, y_val, model_dir):
    """Train XGBoost model for FLIR+SCD41 data."""
    print("Training XGBoost model for FLIR+SCD41 data...")
    
    # Calculate class weights for imbalanced data
    scale_pos_weight = len(y_train) / (2 * max(1, sum(y_train)))
    
    # Create and train model
    model = FlirScd41XGBoost(
        max_depth=6,
        eta=0.3,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    val_pred_proba = model.predict_proba(X_val)
    
    # Calculate metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, train_pred),
        'f1_score': f1_score(y_train, train_pred),
        'precision': precision_score(y_train, train_pred),
        'recall': recall_score(y_train, train_pred)
    }
    
    val_metrics = {
        'accuracy': accuracy_score(y_val, val_pred),
        'f1_score': f1_score(y_val, val_pred),
        'precision': precision_score(y_val, val_pred),
        'recall': recall_score(y_val, val_pred),
        'auc': roc_auc_score(y_val, val_pred_proba) if len(np.unique(y_val)) > 1 else 0.0
    }
    
    print('XGBoost Training Metrics:')
    for metric, value in train_metrics.items():
        print(f'  {metric}: {value:.4f}')
    
    print('XGBoost Validation Metrics:')
    for metric, value in val_metrics.items():
        print(f'  {metric}: {value:.4f}')
    
    # Save model components
    model_path = os.path.join(model_dir, 'xgboost_model.json')
    model.model.save_model(model_path)
    
    scaler_path = os.path.join(model_dir, 'xgboost_scaler.joblib')
    joblib.dump(model.scaler, scaler_path)
    
    # Save metrics
    metrics_path = os.path.join(model_dir, 'xgboost_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'model_type': 'xgboost'
        }, f, indent=2)
    
    print(f'XGBoost model saved to {model_path}')
    print(f'XGBoost scaler saved to {scaler_path}')
    print(f'XGBoost metrics saved to {metrics_path}')
    
    return val_metrics


def train_lstm_model(X_train, y_train, X_val, y_val, model_dir, epochs=50, batch_size=64):
    """Train LSTM model for FLIR+SCD41 data."""
    print("Training LSTM model for FLIR+SCD41 data...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    train_dataset = FlirScd41Dataset(X_train, y_train)
    val_dataset = FlirScd41Dataset(X_val, y_val, scaler=train_dataset.scaler)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = FlirScd41LSTM(input_size=X_train.shape[1]).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(batch_labels.cpu().numpy())
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device).squeeze()
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        train_acc = accuracy_score(train_targets, train_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        val_precision = precision_score(val_targets, val_preds, average='weighted')
        val_recall = recall_score(val_targets, val_preds, average='weighted')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}')
            print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_acc:.4f}')
            print(f'Validation F1: {val_f1:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}')
            print('-' * 50)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final metrics
    final_metrics = {
        'accuracy': best_val_acc,
        'f1_score': val_f1,
        'precision': val_precision,
        'recall': val_recall
    }
    
    print('LSTM Final Validation Metrics:')
    for metric, value in final_metrics.items():
        print(f'  {metric}: {value:.4f}')
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': train_dataset.scaler,
        'metrics': final_metrics,
        'model_type': 'lstm'
    }, os.path.join(model_dir, 'lstm_model.pth'))
    
    print(f'LSTM model saved to {os.path.join(model_dir, "lstm_model.pth")}')
    
    return final_metrics


def main():
    """Main training function for FLIR+SCD41 models."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', type=str, nargs='+', default=['xgboost', 'lstm'],
                        help='Model types to train (xgboost, lstm)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    
    # SageMaker environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', './data/val'))
    
    args = parser.parse_args()
    
    print(f'Training FLIR+SCD41 models: {args.model_types}')
    print(f'Training data directory: {args.train}')
    print(f'Validation data directory: {args.validation}')
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    X_train, y_train, feature_names = load_flir_scd41_data(args.train)
    print(f'Training data shape: {X_train.shape}')
    print(f'Training fire samples: {sum(y_train)}')
    
    # Load validation data
    print("Loading validation data...")
    X_val, y_val, _ = load_flir_scd41_data(args.validation)
    print(f'Validation data shape: {X_val.shape}')
    print(f'Validation fire samples: {sum(y_val)}')
    
    # Print feature information
    print(f'Features ({len(feature_names)}): {feature_names}')
    
    # Train models
    results = {}
    
    if 'xgboost' in args.model_types:
        xgboost_metrics = train_xgboost_model(X_train, y_train, X_val, y_val, args.model_dir)
        results['xgboost'] = xgboost_metrics
    
    if 'lstm' in args.model_types:
        lstm_metrics = train_lstm_model(X_train, y_train, X_val, y_val, args.model_dir, 
                                       epochs=args.epochs, batch_size=args.batch_size)
        results['lstm'] = lstm_metrics
    
    # Save overall results
    results_path = os.path.join(args.model_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Training results saved to {results_path}')
    
    # Print final summary
    print("\nTraining Summary:")
    for model_type, metrics in results.items():
        print(f"{model_type.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()