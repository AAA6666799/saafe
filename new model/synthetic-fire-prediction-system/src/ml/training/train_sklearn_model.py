"""
Training script for scikit-learn models compatible with SageMaker.
This script can be used to train Random Forest, XGBoost, and other scikit-learn models.
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
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

def train_model(X_train, y_train, model_type, hyperparameters):
    """Train a scikit-learn model."""
    logger.info(f"Training {model_type} model with hyperparameters: {hyperparameters}")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=hyperparameters.get('n_estimators', 100),
            max_depth=hyperparameters.get('max_depth', None),
            min_samples_split=hyperparameters.get('min_samples_split', 2),
            min_samples_leaf=hyperparameters.get('min_samples_leaf', 1),
            max_features=hyperparameters.get('max_features', 'sqrt'),
            bootstrap=hyperparameters.get('bootstrap', True),
            class_weight=hyperparameters.get('class_weight', None),
            random_state=hyperparameters.get('random_state', 42),
            n_jobs=hyperparameters.get('n_jobs', -1)
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            C=hyperparameters.get('C', 1.0),
            penalty=hyperparameters.get('penalty', 'l2'),
            solver=hyperparameters.get('solver', 'lbfgs'),
            max_iter=hyperparameters.get('max_iter', 1000),
            class_weight=hyperparameters.get('class_weight', None),
            random_state=hyperparameters.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    logger.info("Evaluating model performance")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
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
    parser.add_argument('--model-type', type=str, default='random_forest')
    
    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--min-samples-split', type=int, default=2)
    parser.add_argument('--min-samples-leaf', type=int, default=1)
    parser.add_argument('--max-features', type=str, default='sqrt')
    parser.add_argument('--bootstrap', type=bool, default=True)
    parser.add_argument('--class-weight', type=str, default=None)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--n-jobs', type=int, default=-1)
    
    args = parser.parse_args()
    
    logger.info("Starting model training")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Training data path: {args.train}")
    logger.info(f"Validation data path: {args.validation}")
    logger.info(f"Model type: {args.model_type}")
    
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
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf': args.min_samples_leaf,
            'max_features': args.max_features,
            'bootstrap': args.bootstrap,
            'class_weight': args.class_weight,
            'random_state': args.random_state,
            'n_jobs': args.n_jobs
        }
        
        # Train model
        model = train_model(X_train, y_train, args.model_type, hyperparameters)
        
        # Evaluate model
        metrics = evaluate_model(model, X_val, y_val)
        
        # Save model
        model_path = os.path.join(args.model_dir, 'model.joblib')
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        metrics_path = os.path.join(args.model_dir, 'evaluation.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()