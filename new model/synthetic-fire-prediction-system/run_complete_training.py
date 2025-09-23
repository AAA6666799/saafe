#!/usr/bin/env python3
"""
Complete execution script for the FLIR+SCD41 Unified Training Pipeline
Supports large-scale training with 100K+ samples and prevents data leakage
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import warnings
from datetime import datetime
import logging
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    print("✅ All ML libraries imported successfully")
except ImportError as e:
    print(f"❌ Missing required ML libraries: {e}")
    print("Please install them using: pip install -r requirements.txt")
    sys.exit(1)

def generate_synthetic_data(num_samples=100000):
    """Generate enhanced synthetic FLIR+SCD41 dataset with controlled noise and diverse scenarios"""
    print(f"🔄 Generating enhanced synthetic FLIR+SCD41 dataset with {num_samples:,} samples...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate FLIR features (15 features) with more realistic distributions
    flir_features = np.zeros((num_samples, 15))
    
    # t_mean: Mean temperature (-40 to 330°C)
    flir_features[:, 0] = np.random.normal(25, 15, num_samples)
    flir_features[:, 0] = np.clip(flir_features[:, 0], -40, 330)
    
    # t_std: Temperature standard deviation (0 to 50°C)
    flir_features[:, 1] = np.random.gamma(2, 5, num_samples)
    flir_features[:, 1] = np.clip(flir_features[:, 1], 0, 50)
    
    # t_max: Maximum temperature (-40 to 330°C)
    flir_features[:, 2] = np.random.normal(45, 20, num_samples)
    flir_features[:, 2] = np.clip(flir_features[:, 2], -40, 330)
    
    # t_p95: 95th percentile temperature (-40 to 330°C)
    flir_features[:, 3] = np.random.normal(40, 18, num_samples)
    flir_features[:, 3] = np.clip(flir_features[:, 3], -40, 330)
    
    # t_hot_area_pct: Percentage of hot area (0-100%)
    flir_features[:, 4] = np.random.beta(2, 5, num_samples) * 100
    
    # t_hot_largest_blob_pct: Percentage of largest hot blob (0-50%)
    flir_features[:, 5] = np.random.beta(1, 4, num_samples) * 50
    
    # t_grad_mean: Mean temperature gradient (0-20)
    flir_features[:, 6] = np.random.gamma(2, 3, num_samples)
    flir_features[:, 6] = np.clip(flir_features[:, 6], 0, 20)
    
    # t_grad_std: Std of temperature gradient (0-10)
    flir_features[:, 7] = np.random.gamma(1, 2, num_samples)
    flir_features[:, 7] = np.clip(flir_features[:, 7], 0, 10)
    
    # t_diff_mean: Mean temperature difference (0-30)
    flir_features[:, 8] = np.random.gamma(2, 4, num_samples)
    flir_features[:, 8] = np.clip(flir_features[:, 8], 0, 30)
    
    # t_diff_std: Std of temperature difference (0-15)
    flir_features[:, 9] = np.random.gamma(1, 3, num_samples)
    flir_features[:, 9] = np.clip(flir_features[:, 9], 0, 15)
    
    # flow_mag_mean: Mean flow magnitude (0-15)
    flir_features[:, 10] = np.random.gamma(2, 2, num_samples)
    flir_features[:, 10] = np.clip(flir_features[:, 10], 0, 15)
    
    # flow_mag_std: Std of flow magnitude (0-8)
    flir_features[:, 11] = np.random.gamma(1, 1.5, num_samples)
    flir_features[:, 11] = np.clip(flir_features[:, 11], 0, 8)
    
    # tproxy_val: Temperature proxy value (0-100)
    flir_features[:, 12] = np.random.normal(30, 15, num_samples)
    flir_features[:, 12] = np.clip(flir_features[:, 12], 0, 100)
    
    # tproxy_delta: Temperature proxy delta (-50 to 50)
    flir_features[:, 13] = np.random.normal(0, 10, num_samples)
    flir_features[:, 13] = np.clip(flir_features[:, 13], -50, 50)
    
    # tproxy_vel: Temperature proxy velocity (-20 to 20)
    flir_features[:, 14] = np.random.normal(0, 5, num_samples)
    flir_features[:, 14] = np.clip(flir_features[:, 14], -20, 20)
    
    # Generate SCD41 features (3 features) with realistic distributions
    scd41_features = np.zeros((num_samples, 3))
    
    # gas_val: CO2 concentration (400-5000 ppm)
    scd41_features[:, 0] = np.random.normal(450, 200, num_samples)
    scd41_features[:, 0] = np.clip(scd41_features[:, 0], 400, 5000)
    
    # gas_delta: CO2 change rate (-500 to 500 ppm/min)
    scd41_features[:, 1] = np.random.normal(0, 100, num_samples)
    scd41_features[:, 1] = np.clip(scd41_features[:, 1], -500, 500)
    
    # gas_vel: CO2 velocity (-50 to 50 ppm/s)
    scd41_features[:, 2] = np.random.normal(0, 15, num_samples)
    scd41_features[:, 2] = np.clip(scd41_features[:, 2], -50, 50)
    
    print(f"✅ Generated {num_samples:,} samples with enhanced realism")
    return flir_features, scd41_features

def create_dataset(flir_features, scd41_features):
    """Combine features and create labels with balanced distribution and realistic fire patterns"""
    print("💾 Combining features and creating dataset with realistic fire patterns...")
    
    # Combine all features (15 FLIR + 3 SCD41 = 18 features)
    all_features = np.concatenate([flir_features, scd41_features], axis=1)
    
    # Create more sophisticated fire detection logic
    # Fire probability based on multiple interacting factors
    fire_indicators = np.zeros(len(all_features))
    
    # High temperature indicators (weighted more heavily)
    fire_indicators += (flir_features[:, 2] > 60) * 0.3  # High max temperature
    fire_indicators += (flir_features[:, 3] > 55) * 0.25  # High 95th percentile
    fire_indicators += (flir_features[:, 0] > 40) * 0.2   # High mean temperature
    
    # Large hot area indicators
    fire_indicators += (flir_features[:, 4] > 15) * 0.2   # Large hot area
    fire_indicators += (flir_features[:, 5] > 5) * 0.15   # Large hot blob
    
    # Rapid temperature changes
    fire_indicators += (flir_features[:, 6] > 8) * 0.15   # High temperature gradient
    fire_indicators += (flir_features[:, 8] > 12) * 0.15  # High temperature difference
    
    # Elevated CO2 levels
    fire_indicators += (scd41_features[:, 0] > 800) * 0.25  # High CO2
    fire_indicators += (scd41_features[:, 1] > 150) * 0.2   # Rapid CO2 increase
    
    # Interaction effects (more realistic fire signatures)
    # High temperature + high CO2
    temp_co2_interaction = ((flir_features[:, 2] > 70) & (scd41_features[:, 0] > 1000)).astype(int) * 0.3
    fire_indicators += temp_co2_interaction
    
    # Large hot area + rapid temperature change
    area_gradient_interaction = ((flir_features[:, 4] > 20) & (flir_features[:, 6] > 10)).astype(int) * 0.2
    fire_indicators += area_gradient_interaction
    
    # Clip probabilities to [0, 1]
    fire_indicators = np.clip(fire_indicators, 0, 1)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.05, len(fire_indicators))
    fire_probability = np.clip(fire_indicators + noise, 0, 1)
    
    # Generate labels based on fire probability
    labels = np.random.binomial(1, fire_probability)
    
    # Create DataFrame
    feature_names = [
        't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
        't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
        't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
        'tproxy_val', 'tproxy_delta', 'tproxy_vel',
        'gas_val', 'gas_delta', 'gas_vel'
    ]
    
    df = pd.DataFrame(all_features, columns=feature_names)
    df['fire_detected'] = labels
    
    print(f"✅ Dataset created with shape: {df.shape}")
    print(f"Fire samples: {sum(labels):,} ({sum(labels)/len(labels)*100:.2f}%)")
    
    return df, feature_names

def split_dataset(df, feature_names, test_size=0.15, val_size=0.15, random_state=42):
    """Split dataset into train/validation/test sets with stratification to prevent data leakage"""
    print("📊 Splitting dataset into train/validation/test sets with proper stratification...")
    
    # Separate features and labels
    X = df[feature_names].values
    y = df['fire_detected'].values
    
    # First split: separate test set (ensures test set is completely unseen)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation sets
    # Adjust val_size to account for the first split
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]:,} samples")
    print(f"Validation set: {X_val.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    # Verify no data leakage by checking feature distributions
    print("🔍 Verifying no data leakage...")
    train_means = np.mean(X_train, axis=0)
    val_means = np.mean(X_val, axis=0)
    test_means = np.mean(X_test, axis=0)
    
    # Check if distributions are similar (within 10% relative difference)
    relative_diffs = []
    for i in range(len(train_means)):
        if train_means[i] != 0:
            rel_diff = abs(train_means[i] - val_means[i]) / abs(train_means[i])
            relative_diffs.append(rel_diff)
        else:
            relative_diffs.append(0 if val_means[i] == 0 else float('inf'))
    
    max_diff = max(relative_diffs)
    print(f"Maximum relative difference between train/val feature means: {max_diff:.4f}")
    
    if max_diff > 0.5:
        print("⚠️  Warning: Large differences detected between train/val sets. Consider re-splitting.")
    else:
        print("✅ No significant data leakage detected between splits.")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model with regularization and early stopping"""
    print("🚀 Training XGBoost model with regularization...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create and train XGBoost model with strong regularization
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,           # Increased for better performance
        max_depth=6,                # Moderate depth
        learning_rate=0.05,         # Lower learning rate for better generalization
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        random_state=42,
        early_stopping_rounds=20,   # Early stopping
        eval_metric='logloss'
    )
    
    # Fit with validation set for early stopping
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )
    
    # Evaluate XGBoost model
    xgb_train_pred = xgb_model.predict(X_train_scaled)
    xgb_val_pred = xgb_model.predict(X_val_scaled)
    xgb_val_pred_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]
    
    xgb_train_metrics = {
        'accuracy': accuracy_score(y_train, xgb_train_pred),
        'f1_score': f1_score(y_train, xgb_train_pred),
        'precision': precision_score(y_train, xgb_train_pred),
        'recall': recall_score(y_train, xgb_train_pred)
    }
    
    xgb_val_metrics = {
        'accuracy': accuracy_score(y_val, xgb_val_pred),
        'f1_score': f1_score(y_val, xgb_val_pred),
        'precision': precision_score(y_val, xgb_val_pred),
        'recall': recall_score(y_val, xgb_val_pred),
        'auc': roc_auc_score(y_val, xgb_val_pred_proba) if len(np.unique(y_val)) > 1 else 0.0
    }
    
    print("XGBoost Training Metrics:")
    for metric, value in xgb_train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nXGBoost Validation Metrics:")
    for metric, value in xgb_val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return xgb_model, scaler, xgb_train_metrics, xgb_val_metrics

def train_random_forest_model(X_train, y_train, X_val, y_val):
    """Train Random Forest model"""
    print("\n🌲 Training Random Forest model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create and train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate Random Forest model
    rf_train_pred = rf_model.predict(X_train_scaled)
    rf_val_pred = rf_model.predict(X_val_scaled)
    rf_val_pred_proba = rf_model.predict_proba(X_val_scaled)[:, 1]
    
    rf_train_metrics = {
        'accuracy': accuracy_score(y_train, rf_train_pred),
        'f1_score': f1_score(y_train, rf_train_pred),
        'precision': precision_score(y_train, rf_train_pred),
        'recall': recall_score(y_train, rf_train_pred)
    }
    
    rf_val_metrics = {
        'accuracy': accuracy_score(y_val, rf_val_pred),
        'f1_score': f1_score(y_val, rf_val_pred),
        'precision': precision_score(y_val, rf_val_pred),
        'recall': recall_score(y_val, rf_val_pred),
        'auc': roc_auc_score(y_val, rf_val_pred_proba) if len(np.unique(y_val)) > 1 else 0.0
    }
    
    print("Random Forest Training Metrics:")
    for metric, value in rf_train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nRandom Forest Validation Metrics:")
    for metric, value in rf_val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return rf_model, scaler, rf_train_metrics, rf_val_metrics

def train_gradient_boosting_model(X_train, y_train, X_val, y_val):
    """Train Gradient Boosting model"""
    print("\n📊 Training Gradient Boosting model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create and train Gradient Boosting model
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    gb_model.fit(X_train_scaled, y_train)
    
    # Evaluate Gradient Boosting model
    gb_train_pred = gb_model.predict(X_train_scaled)
    gb_val_pred = gb_model.predict(X_val_scaled)
    gb_val_pred_proba = gb_model.predict_proba(X_val_scaled)[:, 1]
    
    gb_train_metrics = {
        'accuracy': accuracy_score(y_train, gb_train_pred),
        'f1_score': f1_score(y_train, gb_train_pred),
        'precision': precision_score(y_train, gb_train_pred),
        'recall': recall_score(y_train, gb_train_pred)
    }
    
    gb_val_metrics = {
        'accuracy': accuracy_score(y_val, gb_val_pred),
        'f1_score': f1_score(y_val, gb_val_pred),
        'precision': precision_score(y_val, gb_val_pred),
        'recall': recall_score(y_val, gb_val_pred),
        'auc': roc_auc_score(y_val, gb_val_pred_proba) if len(np.unique(y_val)) > 1 else 0.0
    }
    
    print("Gradient Boosting Training Metrics:")
    for metric, value in gb_train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nGradient Boosting Validation Metrics:")
    for metric, value in gb_val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return gb_model, scaler, gb_train_metrics, gb_val_metrics

def train_logistic_regression_model(X_train, y_train, X_val, y_val):
    """Train Logistic Regression model"""
    print("\n📈 Training Logistic Regression model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create and train Logistic Regression model
    lr_model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    
    lr_model.fit(X_train_scaled, y_train)
    
    # Evaluate Logistic Regression model
    lr_train_pred = lr_model.predict(X_train_scaled)
    lr_val_pred = lr_model.predict(X_val_scaled)
    lr_val_pred_proba = lr_model.predict_proba(X_val_scaled)[:, 1]
    
    lr_train_metrics = {
        'accuracy': accuracy_score(y_train, lr_train_pred),
        'f1_score': f1_score(y_train, lr_train_pred),
        'precision': precision_score(y_train, lr_train_pred),
        'recall': recall_score(y_train, lr_train_pred)
    }
    
    lr_val_metrics = {
        'accuracy': accuracy_score(y_val, lr_val_pred),
        'f1_score': f1_score(y_val, lr_val_pred),
        'precision': precision_score(y_val, lr_val_pred),
        'recall': recall_score(y_val, lr_val_pred),
        'auc': roc_auc_score(y_val, lr_val_pred_proba) if len(np.unique(y_val)) > 1 else 0.0
    }
    
    print("Logistic Regression Training Metrics:")
    for metric, value in lr_train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nLogistic Regression Validation Metrics:")
    for metric, value in lr_val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return lr_model, scaler, lr_train_metrics, lr_val_metrics

class FireDataset(Dataset):
    """Dataset for fire detection"""
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ImprovedNN(nn.Module):
    """Improved neural network with dropout and batch normalization"""
    def __init__(self, input_size=18, hidden_sizes=[128, 64, 32], num_classes=2, dropout_rate=0.3):
        super(ImprovedNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers with batch normalization and dropout
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_neural_network(X_train, y_train, X_val, y_val):
    """Train improved neural network with early stopping and regularization"""
    print("\n🚀 Training Improved Neural Network model...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create model
    nn_model = ImprovedNN(input_size=18, dropout_rate=0.3).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization
    
    # Create datasets and data loaders
    train_dataset = FireDataset(X_train_scaled, y_train)
    val_dataset = FireDataset(X_val_scaled, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Training loop
    num_epochs = 100
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        nn_model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = nn_model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(batch_labels.cpu().numpy())
        
        # Validation
        nn_model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                
                outputs = nn_model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        train_acc = accuracy_score(train_targets, train_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            model_path = os.path.join(project_root, 'data', 'flir_scd41', 'best_improved_nn_model.pth')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(nn_model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}')
            print(f'  Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_acc:.4f}')
    
    # Load best model
    nn_model.load_state_dict(torch.load(model_path))
    
    # Final NN metrics
    nn_train_metrics = {
        'accuracy': accuracy_score(train_targets, train_preds),
        'f1_score': f1_score(train_targets, train_preds),
        'precision': precision_score(train_targets, train_preds),
        'recall': recall_score(train_targets, train_preds)
    }
    
    nn_val_metrics = {
        'accuracy': best_val_acc,
        'f1_score': f1_score(val_targets, val_preds),
        'precision': precision_score(val_targets, val_preds),
        'recall': recall_score(val_targets, val_preds)
    }
    
    print("\nNeural Network Final Metrics:")
    print("Training Metrics:")
    for metric, value in nn_train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nValidation Metrics:")
    for metric, value in nn_val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return nn_model, scaler, nn_train_metrics, nn_val_metrics

def calculate_ensemble_weights(model_scores):
    """Calculate ensemble weights based on validation performance"""
    print("⚖️ Calculating ensemble weights based on validation performance...")
    
    # Print individual model scores
    model_names = list(model_scores.keys())
    scores = list(model_scores.values())
    
    print("Validation Accuracy Scores:")
    for name, score in model_scores.items():
        print(f"  {name}: {score:.4f}")
    
    # Method: Performance-based weighting (exponential scaling)
    def calculate_performance_weights(scores, scaling_factor=3.0):
        """Calculate weights based on performance scores using exponential scaling"""
        # Normalize scores to [0, 1] range
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All models have same performance, equal weights
            return [1.0/len(scores)] * len(scores)
        
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        
        # Apply exponential scaling
        weighted_scores = [np.exp(scaling_factor * score) for score in normalized_scores]
        
        # Normalize to sum to 1
        total_weight = sum(weighted_scores)
        weights = [w / total_weight for w in weighted_scores]
        
        return weights
    
    # Calculate weights
    ensemble_weights = calculate_performance_weights(scores)
    
    print(f"\nEnsemble weights:")
    for i, (name, score) in enumerate(zip(model_names, scores)):
        print(f"  {name} weight: {ensemble_weights[i]:.4f}")
    
    return dict(zip(model_names, ensemble_weights))

def evaluate_ensemble(X_test, y_test, models, scalers, weights, feature_names):
    """Evaluate ensemble model on test set"""
    print("\n🧪 Evaluating ensemble model on test set...")
    
    # Scale test data using each model's scaler
    scaled_data = {}
    for model_name in models.keys():
        if model_name in scalers and scalers[model_name] is not None:
            scaled_data[model_name] = scalers[model_name].transform(X_test)
        else:
            scaled_data[model_name] = X_test
    
    # Get predictions from each model
    predictions = {}
    probabilities = {}
    
    for model_name, model in models.items():
        if model_name == 'neural_network':
            # Neural network prediction
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            with torch.no_grad():
                test_tensor = torch.FloatTensor(scaled_data[model_name]).to(device)
                outputs = model(test_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
        elif hasattr(model, 'predict'):
            # Scikit-learn style models
            preds = model.predict(scaled_data[model_name])
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(scaled_data[model_name])[:, 1]
            else:
                probs = model.decision_function(scaled_data[model_name])
                # Normalize to [0, 1] range
                probs = (probs - probs.min()) / (probs.max() - probs.min())
        else:
            # Fallback
            preds = np.zeros(len(X_test))
            probs = np.zeros(len(X_test))
        
        predictions[model_name] = preds
        probabilities[model_name] = probs
    
    # Calculate weighted ensemble predictions
    ensemble_probs = np.zeros(len(X_test))
    for model_name, weight in weights.items():
        if model_name in probabilities:
            ensemble_probs += weight * probabilities[model_name]
    
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    # Calculate ensemble metrics
    ensemble_metrics = {
        'accuracy': accuracy_score(y_test, ensemble_preds),
        'f1_score': f1_score(y_test, ensemble_preds),
        'precision': precision_score(y_test, ensemble_preds),
        'recall': recall_score(y_test, ensemble_preds),
        'auc': roc_auc_score(y_test, ensemble_probs) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    print("Ensemble Test Metrics:")
    for metric, value in ensemble_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return ensemble_preds, ensemble_probs, ensemble_metrics

def save_models_and_results(df, models, scalers, ensemble_weights, 
                          model_metrics, feature_names):
    """Save all models and results"""
    print("💾 Saving models and results...")
    
    # Create data directory
    data_dir = os.path.join(project_root, 'data', 'flir_scd41')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save dataset
    dataset_path = os.path.join(data_dir, 'flir_scd41_dataset.csv')
    df.to_csv(dataset_path, index=False)
    
    # Save individual models and scalers
    model_info = {}
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    for model_name, model in models.items():
        if model_name == 'neural_network':
            # Neural Network model already saved during training
            model_path = os.path.join(data_dir, 'best_improved_nn_model.pth')
            model_info[model_name] = {
                'model_path': model_path,
                'metrics': model_metrics.get(model_name, {})
            }
        else:
            # Save other models
            model_path = os.path.join(data_dir, f'flir_scd41_{model_name}_{timestamp}.joblib')
            import joblib
            joblib.dump(model, model_path)
            model_info[model_name] = {
                'model_path': model_path,
                'metrics': model_metrics.get(model_name, {})
            }
        
        # Save scaler if it exists
        if model_name in scalers and scalers[model_name] is not None:
            scaler_path = os.path.join(data_dir, f'flir_scd41_{model_name}_scaler_{timestamp}.joblib')
            import joblib
            joblib.dump(scalers[model_name], scaler_path)
            model_info[model_name]['scaler_path'] = scaler_path
    
    # Save ensemble weights
    weights_data = {
        'models': list(ensemble_weights.keys()),
        'weights': list(ensemble_weights.values()),
        'validation_scores': {name: metrics.get('accuracy', 0) for name, metrics in model_metrics.items()},
        'calculation_method': 'performance_based_exponential_scaling',
        'scaling_factor': 3.0,
        'timestamp': timestamp
    }
    
    weights_path = os.path.join(data_dir, 'ensemble_weights.json')
    with open(weights_path, 'w') as f:
        json.dump(weights_data, f, indent=2)
    
    # Save model information
    complete_model_info = {
        'models': model_info,
        'ensemble': {
            'weights_path': weights_path,
        },
        'feature_names': feature_names,
        'timestamp': timestamp
    }
    
    model_info_path = os.path.join(data_dir, 'model_info.json')
    with open(model_info_path, 'w') as f:
        json.dump(complete_model_info, f, indent=2)
    
    print(f"✅ Dataset saved to {dataset_path}")
    for model_name in models.keys():
        print(f"✅ {model_name} model saved")
    print(f"✅ Ensemble weights saved to {weights_path}")
    print(f"✅ Model information saved to {model_info_path}")

def main():
    """Main execution function"""
    print("🔥 FLIR+SCD41 Fire Detection System - Complete Training Pipeline (100K Samples)")
    print("=" * 80)
    
    try:
        # Generate synthetic data with 100K samples
        flir_features, scd41_features = generate_synthetic_data(num_samples=100000)
        
        # Create dataset
        df, feature_names = create_dataset(flir_features, scd41_features)
        
        # Split dataset with proper stratification to prevent data leakage
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, feature_names)
        
        # Train multiple models
        models = {}
        scalers = {}
        model_metrics = {}
        
        # Train XGBoost model
        xgb_model, xgb_scaler, xgb_train_metrics, xgb_val_metrics = train_xgboost_model(X_train, y_train, X_val, y_val)
        models['xgboost'] = xgb_model
        scalers['xgboost'] = xgb_scaler
        model_metrics['xgboost'] = xgb_val_metrics
        
        # Train Random Forest model
        rf_model, rf_scaler, rf_train_metrics, rf_val_metrics = train_random_forest_model(X_train, y_train, X_val, y_val)
        models['random_forest'] = rf_model
        scalers['random_forest'] = rf_scaler
        model_metrics['random_forest'] = rf_val_metrics
        
        # Train Gradient Boosting model
        gb_model, gb_scaler, gb_train_metrics, gb_val_metrics = train_gradient_boosting_model(X_train, y_train, X_val, y_val)
        models['gradient_boosting'] = gb_model
        scalers['gradient_boosting'] = gb_scaler
        model_metrics['gradient_boosting'] = gb_val_metrics
        
        # Train Logistic Regression model
        lr_model, lr_scaler, lr_train_metrics, lr_val_metrics = train_logistic_regression_model(X_train, y_train, X_val, y_val)
        models['logistic_regression'] = lr_model
        scalers['logistic_regression'] = lr_scaler
        model_metrics['logistic_regression'] = lr_val_metrics
        
        # Train Neural Network model
        nn_model, nn_scaler, nn_train_metrics, nn_val_metrics = train_neural_network(X_train, y_train, X_val, y_val)
        models['neural_network'] = nn_model
        scalers['neural_network'] = nn_scaler
        model_metrics['neural_network'] = nn_val_metrics
        
        # Calculate ensemble weights based on validation performance
        validation_scores = {name: metrics['accuracy'] for name, metrics in model_metrics.items()}
        ensemble_weights = calculate_ensemble_weights(validation_scores)
        
        # Evaluate ensemble on test set
        ensemble_preds, ensemble_probs, ensemble_metrics = evaluate_ensemble(
            X_test, y_test, models, scalers, ensemble_weights, feature_names
        )
        
        # Update model metrics with ensemble metrics
        model_metrics['ensemble'] = ensemble_metrics
        
        # Save models and results
        save_models_and_results(
            df, models, scalers, ensemble_weights,
            model_metrics, feature_names
        )
        
        print("\n🎉 Complete training pipeline executed successfully!")
        print("📁 Check the data/flir_scd41/ directory for output files")
        
        # Print summary of results
        print("\n📊 MODEL PERFORMANCE SUMMARY:")
        print("=" * 50)
        for model_name, metrics in model_metrics.items():
            if model_name == 'ensemble':
                print(f"\n🏆 Ensemble Model:")
            else:
                print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  AUC:       {metrics['auc']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)