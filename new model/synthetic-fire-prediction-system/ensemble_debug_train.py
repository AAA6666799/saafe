#!/usr/bin/env python3
"""
Ensemble debug training script - closely matches the full ensemble approach
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(data_path):
    """Load training data"""
    print("Loading training data...")
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        print(f"ERROR: Data directory {data_path} does not exist")
        sys.exit(1)
    
    # Find JSON files
    data_files = [f for f in os.listdir(data_path) if f.endswith(".json")]
    print(f"Found {len(data_files)} JSON files")
    
    if not data_files:
        print("ERROR: No JSON files found")
        sys.exit(1)
    
    # Load data from all files
    all_samples = []
    for data_file in data_files:
        file_path = os.path.join(data_path, data_file)
        print(f"Loading data from {data_file}")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        samples = data.get('samples', [])
        all_samples.extend(samples)
        print(f"Loaded {len(samples)} samples from {data_file}")
    
    print(f"Total samples loaded: {len(all_samples)}")
    return all_samples

def prepare_features(samples):
    """Prepare features from samples"""
    print("Preparing features...")
    
    # Extract features and labels
    features = []
    labels = []
    
    valid_samples = 0
    invalid_samples = 0
    
    for sample in samples:
        try:
            # Check if sample has the expected structure
            if 'features' not in sample:
                print(f"Warning: Sample missing 'features' key, skipping")
                invalid_samples += 1
                continue
                
            # Check if all required features are present
            # Updated to match the actual feature names in the data
            required_features = ['t_mean', 't_std', 't_max', 'gas_val', 'tproxy_val', 'flow_mag_mean', 'flow_mag_std']
            missing_features = [f for f in required_features if f not in sample['features']]
            
            if missing_features:
                print(f"Warning: Sample missing features {missing_features}, skipping")
                invalid_samples += 1
                continue
            
            # Extract features with the correct names
            feature_vector = [
                sample['features']['t_mean'],        # Temperature mean
                sample['features']['t_std'],         # Temperature std
                sample['features']['t_max'],         # Temperature max
                sample['features']['gas_val'],       # Gas value (CO2 proxy)
                sample['features']['tproxy_val'],    # Temperature proxy value
                sample['features']['flow_mag_mean'], # Flow magnitude mean
                sample['features']['flow_mag_std']   # Flow magnitude std
            ]
            
            features.append(feature_vector)
            labels.append(sample['label'])
            valid_samples += 1
            
        except Exception as e:
            print(f"Warning: Error processing sample: {e}, skipping")
            invalid_samples += 1
    
    print(f"Valid samples: {valid_samples}")
    print(f"Invalid samples: {invalid_samples}")
    
    if valid_samples == 0:
        print("ERROR: No valid samples found")
        sys.exit(1)
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y

def train_ensemble_models(X_train, y_train, X_test, y_test):
    """Train ensemble models"""
    print("Training ensemble models...")
    
    models = {}
    results = {}
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    models['random_forest'] = rf_model
    results['random_forest'] = {
        'accuracy': rf_accuracy,
        'predictions': rf_pred.tolist()
    }
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    models['gradient_boosting'] = gb_model
    results['gradient_boosting'] = {
        'accuracy': gb_accuracy,
        'predictions': gb_pred.tolist()
    }
    print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    models['logistic_regression'] = lr_model
    results['logistic_regression'] = {
        'accuracy': lr_accuracy,
        'predictions': lr_pred.tolist()
    }
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    
    return models, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--output-data-dir", type=str, default="/opt/ml/output/data")
    args = parser.parse_args()
    
    print("=== Ensemble Debug Training Script ===")
    print(f"Data path: {args.data_path}")
    print(f"Model dir: {args.model_dir}")
    print(f"Output data dir: {args.output_data_dir}")
    
    try:
        # Load data
        samples = load_data(args.data_path)
        
        if len(samples) == 0:
            print("ERROR: No samples loaded")
            sys.exit(1)
        
        # Prepare features
        X, y = prepare_features(samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train ensemble models
        models, results = train_ensemble_models(X_train, y_train, X_test, y_test)
        
        # Save models
        print("Saving models...")
        os.makedirs(args.model_dir, exist_ok=True)
        
        for name, model in models.items():
            model_path = os.path.join(args.model_dir, f"{name}_model.joblib")
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save results
        results_path = os.path.join(args.model_dir, "training_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved training results to {results_path}")
        
        # Create a simple summary file
        summary_path = os.path.join(args.model_dir, "model_summary.txt")
        with open(summary_path, "w") as f:
            f.write("Ensemble Debug Model Summary\n")
            f.write("============================\n")
            f.write(f"Total samples: {len(samples)}\n")
            f.write(f"Valid samples: {X.shape[0]}\n")
            f.write(f"Features: {X.shape[1]}\n")
            f.write(f"Random Forest Accuracy: {results['random_forest']['accuracy']:.4f}\n")
            f.write(f"Gradient Boosting Accuracy: {results['gradient_boosting']['accuracy']:.4f}\n")
            f.write(f"Logistic Regression Accuracy: {results['logistic_regression']['accuracy']:.4f}\n")
        
        print("✅ Ensemble debug training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()