#!/usr/bin/env python3
"""
Model Diagnostics for FLIR+SCD41 Fire Detection System

This script helps diagnose common machine learning problems:
1. Model not learning
2. Overfitting (remembering patterns instead of learning)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def diagnose_model_learning(X_train, y_train, X_val, y_val):
    """Diagnose if model is learning properly"""
    print("🔬 Diagnosing model learning...")
    
    # Check for data issues
    print("\n1. Data Quality Check:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Class distribution - Train: {np.bincount(y_train)}")
    print(f"   Class distribution - Validation: {np.bincount(y_val)}")
    
    # Check for data leakage
    common_rows = np.sum([np.any(np.all(X_train == row, axis=1)) for row in X_val])
    if common_rows > 0:
        print(f"   ⚠️  Warning: {common_rows} identical rows found in train and validation sets (possible data leakage)")
    else:
        print("   ✅ No data leakage detected")
    
    # Check for feature variance
    feature_variances = np.var(X_train, axis=0)
    low_variance_features = np.sum(feature_variances < 1e-6)
    if low_variance_features > 0:
        print(f"   ⚠️  Warning: {low_variance_features} features have very low variance")
    else:
        print("   ✅ All features have sufficient variance")
    
    return {
        'samples': X_train.shape[0],
        'features': X_train.shape[1],
        'train_class_dist': np.bincount(y_train),
        'val_class_dist': np.bincount(y_val),
        'data_leakage': common_rows,
        'low_variance_features': low_variance_features
    }

def plot_learning_curves(model_type, X, y, cv=5):
    """Plot learning curves to diagnose underfitting/overfitting"""
    print(f"\n2. Generating {model_type} Learning Curves...")
    
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        scoring = 'accuracy'
    else:  # neural_network
        # For learning curves, we'll use a simpler approach
        print("   Learning curves for neural networks require custom implementation")
        print("   Please use validation curves instead")
        return
    
    # Generate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=scoring
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title(f'Learning Curves ({model_type.capitalize()})')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save plot
    data_dir = os.path.join(project_root, 'data', 'flir_scd41')
    os.makedirs(data_dir, exist_ok=True)
    plot_path = os.path.join(data_dir, f'{model_type}_learning_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   📊 Learning curves saved to {plot_path}")
    
    # Diagnose based on learning curves
    final_train_score = train_mean[-1]
    final_val_score = val_mean[-1]
    gap = final_train_score - final_val_score
    
    print(f"   Final Training Score: {final_train_score:.4f}")
    print(f"   Final Validation Score: {final_val_score:.4f}")
    print(f"   Gap: {gap:.4f}")
    
    if gap > 0.1:
        print("   🤔 Diagnosis: Model may be overfitting (high variance)")
        print("   💡 Suggestions:")
        print("      - Add regularization")
        print("      - Reduce model complexity")
        print("      - Get more training data")
    elif final_val_score < 0.7:
        print("   🤔 Diagnosis: Model may be underfitting (high bias)")
        print("   💡 Suggestions:")
        print("      - Increase model complexity")
        print("      - Add more features")
        print("      - Train for longer")
    else:
        print("   ✅ Model appears to be learning properly")

def plot_validation_curves(X_train, y_train, param_name='max_depth', param_range=range(1, 11)):
    """Plot validation curves to find optimal hyperparameters"""
    print(f"\n3. Generating Validation Curves for {param_name}...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    train_scores, val_scores = validation_curve(
        model, X_train, y_train, 
        param_name=param_name, param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot validation curves
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(param_range, val_mean, 'o-', color='red', label='Validation score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel(param_name)
    plt.ylabel('Accuracy Score')
    plt.title(f'Validation Curves ({param_name})')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save plot
    data_dir = os.path.join(project_root, 'data', 'flir_scd41')
    os.makedirs(data_dir, exist_ok=True)
    plot_path = os.path.join(data_dir, f'validation_curves_{param_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   📊 Validation curves saved to {plot_path}")
    
    # Find optimal parameter
    optimal_idx = np.argmax(val_mean)
    optimal_param = param_range[optimal_idx]
    print(f"   Optimal {param_name}: {optimal_param}")
    print(f"   Best validation score: {val_mean[optimal_idx]:.4f}")

def detect_overfitting(xgb_train_metrics, xgb_val_metrics, nn_train_metrics=None, nn_val_metrics=None):
    """Detect overfitting by comparing training and validation metrics"""
    print("\n4. Overfitting Detection:")
    
    # XGBoost overfitting detection
    xgb_train_acc = xgb_train_metrics['accuracy']
    xgb_val_acc = xgb_val_metrics['accuracy']
    xgb_gap = xgb_train_acc - xgb_val_acc
    
    print(f"   XGBoost - Training Accuracy: {xgb_train_acc:.4f}")
    print(f"   XGBoost - Validation Accuracy: {xgb_val_acc:.4f}")
    print(f"   XGBoost - Gap: {xgb_gap:.4f}")
    
    if xgb_gap > 0.1:
        print("   ⚠️  XGBoost may be overfitting!")
        return True
    else:
        print("   ✅ XGBoost does not appear to be overfitting")
    
    # Neural Network overfitting detection (if provided)
    if nn_train_metrics and nn_val_metrics:
        nn_train_acc = nn_train_metrics['accuracy']
        nn_val_acc = nn_val_metrics['accuracy']
        nn_gap = nn_train_acc - nn_val_acc
        
        print(f"   Neural Network - Training Accuracy: {nn_train_acc:.4f}")
        print(f"   Neural Network - Validation Accuracy: {nn_val_acc:.4f}")
        print(f"   Neural Network - Gap: {nn_gap:.4f}")
        
        if nn_gap > 0.1:
            print("   ⚠️  Neural Network may be overfitting!")
            return True
        else:
            print("   ✅ Neural Network does not appear to be overfitting")
    
    return False

def suggest_solutions(overfitting_detected, learning_issues=False):
    """Suggest solutions based on detected problems"""
    print("\n5. Recommended Solutions:")
    
    if overfitting_detected:
        print("   🔧 Overfitting Solutions:")
        print("      1. Add regularization:")
        print("         - XGBoost: Increase reg_alpha, reg_lambda")
        print("         - Neural Network: Increase dropout, L1/L2 regularization")
        print("      2. Reduce model complexity:")
        print("         - XGBoost: Decrease max_depth, n_estimators")
        print("         - Neural Network: Reduce layers/neurons")
        print("      3. Get more training data")
        print("      4. Use cross-validation for better evaluation")
        print("      5. Apply early stopping")
        print("      6. Use data augmentation techniques")
    
    if learning_issues:
        print("   🔧 Learning Issues Solutions:")
        print("      1. Check data quality:")
        print("         - Remove duplicates")
        print("         - Handle missing values")
        print("         - Check for data leakage")
        print("      2. Feature engineering:")
        print("         - Normalize/standardize features")
        print("         - Remove irrelevant features")
        print("         - Create new meaningful features")
        print("      3. Model adjustments:")
        print("         - Try different algorithms")
        print("         - Tune hyperparameters")
        print("         - Adjust learning rate")
        print("      4. Training process:")
        print("         - Increase training epochs")
        print("         - Use different optimizers")
        print("         - Implement learning rate scheduling")

def run_comprehensive_diagnostics(X_train, y_train, X_val, y_val, 
                                xgb_train_metrics, xgb_val_metrics,
                                nn_train_metrics=None, nn_val_metrics=None):
    """Run all diagnostics"""
    print("🔍 Running Comprehensive Model Diagnostics")
    print("="*50)
    
    # 1. Basic diagnostics
    data_info = diagnose_model_learning(X_train, y_train, X_val, y_val)
    
    # 2. Overfitting detection
    overfitting_detected = detect_overfitting(
        xgb_train_metrics, xgb_val_metrics, 
        nn_train_metrics, nn_val_metrics
    )
    
    # 3. Suggest solutions
    suggest_solutions(overfitting_detected)
    
    print("\n✅ Diagnostics completed!")
    return overfitting_detected

# Example usage function
def example_usage():
    """Example of how to use the diagnostics"""
    print("Example Usage of Model Diagnostics")
    print("="*35)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 18)
    y = np.random.randint(0, 2, 1000)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Mock metrics (in practice, these would come from actual model training)
    xgb_train_metrics = {'accuracy': 0.95}
    xgb_val_metrics = {'accuracy': 0.85}
    
    # Run diagnostics
    run_comprehensive_diagnostics(
        X_train, y_train, X_val, y_val,
        xgb_train_metrics, xgb_val_metrics
    )

if __name__ == "__main__":
    example_usage()