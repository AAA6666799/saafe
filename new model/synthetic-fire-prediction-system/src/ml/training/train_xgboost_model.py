"""
Training script for XGBoost models compatible with SageMaker.
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load training data from CSV file."""
    logger.info(f"Loading data from {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop('fire_detected', axis=1)
    y = df['fire_detected']
    
    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

def train_model(X_train, y_train, hyperparameters):
    """Train an XGBoost model."""
    logger.info(f"Training XGBoost model with hyperparameters: {hyperparameters}")
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Set parameters
    params = {
        'max_depth': hyperparameters.get('max_depth', 6),
        'eta': hyperparameters.get('eta', 0.3),
        'objective': hyperparameters.get('objective', 'binary:logistic'),
        'eval_metric': hyperparameters.get('eval_metric', 'logloss'),
        'subsample': hyperparameters.get('subsample', 1.0),
        'colsample_bytree': hyperparameters.get('colsample_bytree', 1.0),
        'min_child_weight': hyperparameters.get('min_child_weight', 1),
        'gamma': hyperparameters.get('gamma', 0),
        'seed': hyperparameters.get('seed', 42)
    }
    
    # Number of boosting rounds
    num_rounds = hyperparameters.get('num_round', 100)
    
    # Train the model
    model = xgb.train(params, dtrain, num_rounds)
    logger.info("XGBoost model training completed")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained XGBoost model."""
    logger.info("Evaluating XGBoost model performance")
    
    # Create DMatrix for testing
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Make predictions
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }
    
    logger.info("Model evaluation completed:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  AUC: {auc:.4f}")
    
    return metrics

def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    
    # XGBoost hyperparameters
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--eta', type=float, default=0.3)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--eval-metric', type=str, default='logloss')
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--colsample-bytree', type=float, default=1.0)
    parser.add_argument('--min-child-weight', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--num-round', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    logger.info("Starting XGBoost model training")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Training data path: {args.train}")
    logger.info(f"Validation data path: {args.validation}")
    
    try:
        # Load training data
        train_files = [os.path.join(args.train, f) for f in os.listdir(args.train) if f.endswith('.csv')]
        if not train_files:
            raise ValueError(f"No CSV files found in training directory: {args.train}")
        
        X_train_list, y_train_list = [], []
        for train_file in train_files:
            X, y = load_data(train_file)
            X_train_list.append(X)
            y_train_list.append(y)
        
        X_train = pd.concat(X_train_list, ignore_index=True)
        y_train = pd.concat(y_train_list, ignore_index=True)
        
        # Load validation data
        val_files = [os.path.join(args.validation, f) for f in os.listdir(args.validation) if f.endswith('.csv')]
        if not val_files:
            raise ValueError(f"No CSV files found in validation directory: {args.validation}")
        
        X_val_list, y_val_list = [], []
        for val_file in val_files:
            X, y = load_data(val_file)
            X_val_list.append(X)
            y_val_list.append(y)
        
        X_val = pd.concat(X_val_list, ignore_index=True)
        y_val = pd.concat(y_val_list, ignore_index=True)
        
        # Prepare hyperparameters
        hyperparameters = {
            'max_depth': args.max_depth,
            'eta': args.eta,
            'objective': args.objective,
            'eval_metric': args.eval_metric,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'min_child_weight': args.min_child_weight,
            'gamma': args.gamma,
            'num_round': args.num_round,
            'seed': args.seed
        }
        
        # Train model
        model = train_model(X_train, y_train, hyperparameters)
        
        # Evaluate model
        metrics = evaluate_model(model, X_val, y_val)
        
        # Save model
        model_path = os.path.join(args.model_dir, 'model.bin')
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        metrics_path = os.path.join(args.model_dir, 'evaluation.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        logger.info("XGBoost training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()