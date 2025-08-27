#!/usr/bin/env python3
"""
XGBoost Training Script for SageMaker
Handles XGBoost model training with fire detection data.
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib

def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--eta', type=float, default=0.3)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--num_round', type=int, default=100)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--colsample_bytree', type=float, default=1.0)
    
    # SageMaker environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', './data/val'))
    
    args = parser.parse_args()
    
    print(f'Training XGBoost model: {args.model_type}')
    
    # Load data - find CSV files in directory
    train_files = [f for f in os.listdir(args.train) if f.endswith('.csv')]
    val_files = [f for f in os.listdir(args.validation) if f.endswith('.csv')]
    
    if not train_files or not val_files:
        raise FileNotFoundError("No CSV files found in training or validation directories")
    
    train_data = pd.read_csv(os.path.join(args.train, train_files[0]))
    val_data = pd.read_csv(os.path.join(args.validation, val_files[0]))
    
    # Prepare features and labels
    feature_cols = [col for col in train_data.columns if col != 'fire_detected']
    X_train = train_data[feature_cols].values
    y_train = train_data['fire_detected'].values
    X_val = val_data[feature_cols].values
    y_val = val_data['fire_detected'].values
    
    print(f'Training data shape: {X_train.shape}')
    print(f'Validation data shape: {X_val.shape}')
    print(f'Fire samples in training: {sum(y_train)}')
    print(f'Fire samples in validation: {sum(y_val)}')
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dval = xgb.DMatrix(X_val_scaled, label=y_val)
    
    # Set parameters
    params = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'objective': args.objective,
        'eval_metric': 'logloss',
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'random_state': 42,
        'verbosity': 1
    }
    
    # Add class weight for imbalanced data
    scale_pos_weight = len(y_train) / (2 * sum(y_train))
    params['scale_pos_weight'] = scale_pos_weight
    
    print(f'XGBoost parameters: {params}')
    
    # Train model with validation
    watchlist = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_round,
        evals=watchlist,
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # Make predictions
    train_pred_proba = model.predict(dtrain)
    val_pred_proba = model.predict(dval)
    
    # Convert probabilities to binary predictions
    train_pred = (train_pred_proba > 0.5).astype(int)
    val_pred = (val_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, train_pred),
        'f1_score': f1_score(y_train, train_pred),
        'precision': precision_score(y_train, train_pred),
        'recall': recall_score(y_train, train_pred),
        'auc': roc_auc_score(y_train, train_pred_proba)
    }
    
    val_metrics = {
        'accuracy': accuracy_score(y_val, val_pred),
        'f1_score': f1_score(y_val, val_pred),
        'precision': precision_score(y_val, val_pred),
        'recall': recall_score(y_val, val_pred),
        'auc': roc_auc_score(y_val, val_pred_proba)
    }
    
    print('Training Metrics:')
    for metric, value in train_metrics.items():
        print(f'  {metric}: {value:.4f}')
    
    print('Validation Metrics:')
    for metric, value in val_metrics.items():
        print(f'  {metric}: {value:.4f}')
    
    # Save model
    model_path = os.path.join(args.model_dir, 'model.bst')
    model.save_model(model_path)
    
    # Save scaler
    scaler_path = os.path.join(args.model_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    # Save metrics
    metrics_path = os.path.join(args.model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'model_type': args.model_type,
            'parameters': params
        }, f, indent=2)
    
    # Save feature importance
    importance = model.get_fscore()
    if importance:
        importance_path = os.path.join(args.model_dir, 'feature_importance.json')
        with open(importance_path, 'w') as f:
            json.dump(importance, f, indent=2)
    
    print(f'Model saved to {model_path}')
    print(f'Scaler saved to {scaler_path}')
    print(f'Metrics saved to {metrics_path}')
    
    # Print final validation metrics for SageMaker
    print(f'Final validation accuracy: {val_metrics["accuracy"]:.4f}')
    print(f'Final validation F1-score: {val_metrics["f1_score"]:.4f}')

if __name__ == '__main__':
    main()