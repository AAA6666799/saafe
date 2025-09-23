#!/usr/bin/env python3
"""
Improved Training Pipeline for FLIR+SCD41 Fire Detection System

This script implements best practices to prevent overfitting and ensure proper learning:
1. Early stopping
2. Cross-validation
3. Regularization
4. Data augmentation
5. Proper train/validation/test splits
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def generate_synthetic_data(num_samples=10000):
    """Generate synthetic FLIR+SCD41 dataset with controlled noise"""
    print("🔄 Generating synthetic FLIR+SCD41 dataset...")
    
    # Generate FLIR features (15 features)
    np.random.seed(42)
    flir_features = np.random.normal(25, 10, (num_samples, 15))
    flir_features[:, 0] = np.clip(flir_features[:, 0], -40, 330)  # t_mean: -40 to 330°C
    flir_features[:, 2] = np.clip(flir_features[:, 2], -40, 330)  # t_max: -40 to 330°C
    flir_features[:, 4] = np.clip(flir_features[:, 4], 0, 100)    # t_hot_area_pct: 0-100%
    
    # Generate SCD41 features (3 features)
    scd41_features = np.random.normal(450, 100, (num_samples, 3))
    scd41_features[:, 0] = np.clip(scd41_features[:, 0], 400, 40000)  # gas_val: 400-40000 ppm
    
    print(f"✅ Generated {num_samples} samples")
    print(f"FLIR features shape: {flir_features.shape}")
    print(f"SCD41 features shape: {scd41_features.shape}")
    
    return flir_features, scd41_features

def create_dataset(flir_features, scd41_features):
    """Combine features and create labels with balanced distribution"""
    print("💾 Combining features and creating dataset...")
    
    # Combine all features (15 FLIR + 3 SCD41 = 18 features)
    all_features = np.concatenate([flir_features, scd41_features], axis=1)
    
    # Create labels (fire detected or not) with more realistic patterns
    # Fire probability based on multiple factors
    fire_probability = (
        (flir_features[:, 2] > 60).astype(int) * 0.3 +  # High max temperature
        (scd41_features[:, 0] > 1000).astype(int) * 0.3 +  # High CO2
        (flir_features[:, 4] > 10).astype(int) * 0.2 +  # Large hot area
        (flir_features[:, 12] > 50).astype(int) * 0.2  # High temperature proxy
    )
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, len(fire_probability))
    fire_probability = np.clip(fire_probability + noise, 0, 1)
    
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
    print(f"Fire samples: {sum(labels)} ({sum(labels)/len(labels)*100:.2f}%)")
    
    return df, feature_names

def augment_data(X, y, factor=0.1):
    """Simple data augmentation by adding noise"""
    print("📈 Augmenting training data...")
    
    n_samples = X.shape[0]
    n_augment = int(n_samples * factor)
    
    # Randomly select samples to augment
    indices = np.random.choice(n_samples, n_augment, replace=True)
    
    # Add gaussian noise
    noise = np.random.normal(0, 0.01, (n_augment, X.shape[1]))
    X_augmented = X[indices] + noise
    y_augmented = y[indices]
    
    # Combine original and augmented data
    X_combined = np.vstack([X, X_augmented])
    y_combined = np.hstack([y, y_augmented])
    
    print(f"   Original samples: {n_samples}")
    print(f"   Augmented samples: {n_augment}")
    print(f"   Total samples: {X_combined.shape[0]}")
    
    return X_combined, y_combined

def split_dataset(df, feature_names, test_size=0.15, val_size=0.15):
    """Split dataset into train/validation/test sets with stratification"""
    print("📊 Splitting dataset into train/validation/test sets...")
    
    # Separate features and labels
    X = df.drop('fire_detected', axis=1).values
    y = df['fire_detected'].values
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: separate train and validation sets
    # Adjust val_size to account for the first split
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model with regularization and early stopping"""
    print("🚀 Training XGBoost model with regularization...")
    
    # Create and train XGBoost model with strong regularization
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,          # Increased for early stopping
        max_depth=4,               # Reduced depth to prevent overfitting
        learning_rate=0.1,         # Lower learning rate
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,             # L1 regularization
        reg_lambda=1.0,            # L2 regularization
        random_state=42,
        early_stopping_rounds=10,  # Early stopping
        eval_metric='logloss'
    )
    
    # Fit with validation set for early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate XGBoost model
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_val_pred_proba = xgb_model.predict_proba(X_val)[:, 1]
    
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
    
    # Check for overfitting
    accuracy_gap = xgb_train_metrics['accuracy'] - xgb_val_metrics['accuracy']
    if accuracy_gap > 0.1:
        print("⚠️  Warning: XGBoost may be overfitting!")
        print(f"   Training-Validation accuracy gap: {accuracy_gap:.4f}")
    
    return xgb_model, xgb_train_metrics, xgb_val_metrics

class ImprovedNN(nn.Module):
    """Improved neural network with dropout and batch normalization"""
    def __init__(self, input_size=18, hidden_sizes=[64, 32, 16], num_classes=2, dropout_rate=0.3):
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

class FireDataset(Dataset):
    """Dataset for fire detection"""
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_neural_network(X_train, y_train, X_val, y_val):
    """Train improved neural network with early stopping and regularization"""
    print("\n🚀 Training Improved Neural Network model...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data augmentation
    X_train_aug, y_train_aug = augment_data(X_train, y_train, factor=0.2)
    
    # Create model
    nn_model = ImprovedNN(input_size=18, dropout_rate=0.3).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Create datasets and data loaders
    train_dataset = FireDataset(X_train_aug, y_train_aug)
    val_dataset = FireDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    # Training loop
    num_epochs = 100  # Increased epochs with early stopping
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    
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
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(project_root, 'data', 'flir_scd41', 'best_improved_nn_model.pth')
            torch.save(nn_model.state_dict(), model_path)
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Store losses for plotting
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}')
            print(f'  Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_acc:.4f}')
    
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
    
    # Check for overfitting
    accuracy_gap = nn_train_metrics['accuracy'] - nn_val_metrics['accuracy']
    if accuracy_gap > 0.1:
        print("⚠️  Warning: Neural Network may be overfitting!")
        print(f"   Training-Validation accuracy gap: {accuracy_gap:.4f}")
    
    return nn_model, nn_train_metrics, nn_val_metrics

def cross_validate_models(X, y):
    """Perform cross-validation to get more robust performance estimates"""
    print("\n🔍 Performing Cross-Validation...")
    
    # XGBoost cross-validation
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    
    # Stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_scores = cross_val_score(xgb_model, X, y, cv=skf, scoring='accuracy')
    
    print(f"XGBoost Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

def calculate_ensemble_weights(xgb_score, nn_score):
    """Calculate ensemble weights based on validation performance"""
    print("⚖️ Calculating ensemble weights...")
    
    print(f"XGBoost validation accuracy: {xgb_score:.4f}")
    print(f"Neural Network validation accuracy: {nn_score:.4f}")
    
    # Method: Performance-based weighting (exponential scaling)
    def calculate_performance_weights(scores, scaling_factor=2.0):
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
    model_scores = [xgb_score, nn_score]
    ensemble_weights = calculate_performance_weights(model_scores)
    
    print(f"\nEnsemble weights:")
    print(f"  XGBoost weight: {ensemble_weights[0]:.4f}")
    print(f"  Neural Network weight: {ensemble_weights[1]:.4f}")
    
    return ensemble_weights

def evaluate_models(xgb_model, nn_model, X_test, y_test, ensemble_weights):
    """Evaluate models on test set"""
    print("🧪 Evaluating models on test set...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # XGBoost predictions
    xgb_test_pred = xgb_model.predict(X_test)
    xgb_test_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Neural Network predictions
    nn_model.eval()
    with torch.no_grad():
        test_data = torch.FloatTensor(X_test).to(device)
        nn_outputs = nn_model(test_data)
        nn_test_pred_proba = torch.softmax(nn_outputs, dim=1)[:, 1].cpu().numpy()
        nn_test_pred = (nn_test_pred_proba > 0.5).astype(int)
    
    # Ensemble predictions (weighted average)
    ensemble_pred_proba = (
        ensemble_weights[0] * xgb_test_pred_proba + 
        ensemble_weights[1] * nn_test_pred_proba
    )
    ensemble_test_pred = (ensemble_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    xgb_test_metrics = {
        'accuracy': accuracy_score(y_test, xgb_test_pred),
        'f1_score': f1_score(y_test, xgb_test_pred),
        'precision': precision_score(y_test, xgb_test_pred),
        'recall': recall_score(y_test, xgb_test_pred),
        'auc': roc_auc_score(y_test, xgb_test_pred_proba)
    }
    
    nn_test_metrics = {
        'accuracy': accuracy_score(y_test, nn_test_pred),
        'f1_score': f1_score(y_test, nn_test_pred),
        'precision': precision_score(y_test, nn_test_pred),
        'recall': recall_score(y_test, nn_test_pred),
        'auc': roc_auc_score(y_test, nn_test_pred_proba)
    }
    
    ensemble_test_metrics = {
        'accuracy': accuracy_score(y_test, ensemble_test_pred),
        'f1_score': f1_score(y_test, ensemble_test_pred),
        'precision': precision_score(y_test, ensemble_test_pred),
        'recall': recall_score(y_test, ensemble_test_pred),
        'auc': roc_auc_score(y_test, ensemble_pred_proba)
    }
    
    print("Test Set Performance:")
    print("\nXGBoost:")
    for metric, value in xgb_test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nNeural Network:")
    for metric, value in nn_test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nEnsemble:")
    for metric, value in ensemble_test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return xgb_test_metrics, nn_test_metrics, ensemble_test_metrics

def save_models_and_results(df, xgb_model, nn_model, ensemble_weights, 
                          xgb_test_metrics, nn_test_metrics, ensemble_test_metrics, feature_names):
    """Save all models and results"""
    print("💾 Saving models and results...")
    
    # Create data directory
    data_dir = os.path.join(project_root, 'data', 'flir_scd41')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save dataset
    dataset_path = os.path.join(data_dir, 'flir_scd41_dataset.csv')
    df.to_csv(dataset_path, index=False)
    
    # Save XGBoost model
    xgb_model_path = os.path.join(data_dir, 'flir_scd41_xgboost_model_improved.json')
    xgb_model.save_model(xgb_model_path)
    
    # Neural Network model already saved during training
    nn_model_path = os.path.join(data_dir, 'best_improved_nn_model.pth')
    
    # Save ensemble weights
    weights_data = {
        'models': ['xgboost', 'neural_network'],
        'weights': ensemble_weights,
        'validation_scores': {
            'xgboost': xgb_test_metrics['accuracy'],
            'neural_network': nn_test_metrics['accuracy']
        },
        'calculation_method': 'performance_based_exponential_scaling',
        'scaling_factor': 2.0
    }
    
    weights_path = os.path.join(data_dir, 'ensemble_weights_improved.json')
    with open(weights_path, 'w') as f:
        json.dump(weights_data, f, indent=2)
    
    # Save model information
    model_info = {
        'xgboost': {
            'model_path': xgb_model_path,
            'metrics': xgb_test_metrics
        },
        'neural_network': {
            'model_path': nn_model_path,
            'metrics': nn_test_metrics
        },
        'ensemble': {
            'weights_path': weights_path,
            'metrics': ensemble_test_metrics
        },
        'feature_names': feature_names,
        'training_date': datetime.now().isoformat()
    }
    
    model_info_path = os.path.join(data_dir, 'model_info_improved.json')
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Dataset saved to {dataset_path}")
    print(f"✅ XGBoost model saved to {xgb_model_path}")
    print(f"✅ Neural Network model saved to {nn_model_path}")
    print(f"✅ Ensemble weights saved to {weights_path}")
    print(f"✅ Model information saved to {model_info_path}")

def main():
    """Main training pipeline with improved techniques"""
    print("🔥 FLIR+SCD41 Fire Detection System - Improved Training Pipeline")
    print("="*70)
    
    # Step 1: Generate synthetic data
    flir_features, scd41_features = generate_synthetic_data(num_samples=10000)
    
    # Step 2: Create dataset
    df, feature_names = create_dataset(flir_features, scd41_features)
    
    # Step 3: Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, feature_names)
    
    # Step 4: Cross-validation
    cv_scores = cross_validate_models(X_train, y_train)
    
    # Step 5: Train models with regularization
    xgb_model, xgb_train_metrics, xgb_val_metrics = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # Load the NN model that was saved during training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn_model = ImprovedNN(input_size=18).to(device)
    model_path = os.path.join(project_root, 'data', 'flir_scd41', 'best_improved_nn_model.pth')
    if os.path.exists(model_path):
        nn_model.load_state_dict(torch.load(model_path, map_location=device))
    
    # For simplicity, we'll retrain the NN model
    nn_model, nn_train_metrics, nn_val_metrics = train_neural_network(X_train, y_train, X_val, y_val)
    
    # Step 6: Calculate ensemble weights
    ensemble_weights = calculate_ensemble_weights(
        xgb_val_metrics['accuracy'], 
        nn_val_metrics['accuracy']
    )
    
    # Step 7: Evaluate on test set
    xgb_test_metrics, nn_test_metrics, ensemble_test_metrics = evaluate_models(
        xgb_model, nn_model, X_test, y_test, ensemble_weights
    )
    
    # Step 8: Save models and results
    save_models_and_results(
        df, xgb_model, nn_model, ensemble_weights,
        xgb_test_metrics, nn_test_metrics, ensemble_test_metrics, feature_names
    )
    
    # Print summary
    print("\n🏁 Training Process Summary")
    print("="*50)
    print(f"Dataset Size: {len(df):,} samples")
    print(f"Features: {len(feature_names)} (15 FLIR + 3 SCD41)")
    print(f"Fire Samples: {sum(df['fire_detected'])} ({sum(df['fire_detected'])/len(df)*100:.2f}%)")
    print(f"Training Samples: {len(X_train):,}")
    print(f"Validation Samples: {len(X_val):,}")
    print(f"Test Samples: {len(X_test):,}")
    print()
    print("Model Performance (Test Set):")
    print(f"  XGBoost Accuracy: {xgb_test_metrics['accuracy']:.4f}")
    print(f"  Neural Network Accuracy: {nn_test_metrics['accuracy']:.4f}")
    print(f"  Ensemble Accuracy: {ensemble_test_metrics['accuracy']:.4f}")
    print()
    print("Ensemble Weights:")
    print(f"  XGBoost: {ensemble_weights[0]:.4f}")
    print(f"  Neural Network: {ensemble_weights[1]:.4f}")
    print()
    print("✅ Improved end-to-end training pipeline completed successfully!")

if __name__ == "__main__":
    main()