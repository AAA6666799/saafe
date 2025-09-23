#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Model Evaluation
This script evaluates the trained model's performance using a validation dataset.
"""

import boto3
import pandas as pd
import numpy as np
import json
import tempfile
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# AWS Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET = 'fire-detection-training-691595239825'
S3_PREFIX = 'flir_scd41_training'

def download_model():
    """Download the trained model from S3."""
    print("Downloading trained model...")
    
    # Initialize S3 client
    s3 = boto3.client('s3', region_name=AWS_REGION)
    
    # Model artifacts URI
    model_key = f"{S3_PREFIX}/models/flir-scd41-xgboost-simple-20250828-154649/output/model.tar.gz"
    local_model_path = "/tmp/model.tar.gz"
    
    try:
        # Download model
        s3.download_file(S3_BUCKET, model_key, local_model_path)
        print(f"Model downloaded to {local_model_path}")
        
        # Extract model
        import tarfile
        with tarfile.open(local_model_path, 'r:gz') as tar:
            tar.extractall(path="/tmp/model")
        
        print("Model extracted successfully")
        return "/tmp/model"
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def load_validation_data():
    """Load validation data for evaluation."""
    print("Loading validation data...")
    
    # Initialize S3 client
    s3 = boto3.client('s3', region_name=AWS_REGION)
    
    # Download JSON data from S3
    json_key = f"{S3_PREFIX}/data/demo_data_50000.json"
    local_json_path = "/tmp/demo_data_50000.json"
    
    try:
        s3.download_file(S3_BUCKET, json_key, local_json_path)
        print(f"Downloaded JSON data from s3://{S3_BUCKET}/{json_key}")
    except Exception as e:
        print(f"Error downloading JSON data: {e}")
        return None
    
    # Load JSON data
    with open(local_json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    samples = data["samples"]
    df = pd.DataFrame([{
        **sample["features"],
        "label": sample["label"]
    } for sample in samples])
    
    # For XGBoost, the label should be the first column
    columns = list(df.columns)
    label_col = columns.pop()  # Remove 'label' from the end
    columns.insert(0, label_col)  # Insert 'label' at the beginning
    df = df[columns]
    
    # Split into features and labels
    y = df.iloc[:, 0]  # First column is label
    X = df.iloc[:, 1:]  # Remaining columns are features
    
    print(f"Loaded {len(df)} samples for evaluation")
    return X, y

def evaluate_model(model_path, X_test, y_test):
    """Evaluate the model performance."""
    print("Evaluating model performance...")
    
    try:
        # Load model
        model = xgb.Booster()
        model.load_model(f"{model_path}/xgboost-model")
        
        # Convert to DMatrix for prediction
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Make predictions
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Print results
        print("\n" + "=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")
        print("=" * 50)
        
        # Save results
        results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc)
        }
        
        with open("/tmp/model_evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("Evaluation results saved to /tmp/model_evaluation_results.json")
        
        return results
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None

def main():
    """Main function to evaluate the trained model."""
    print("FLIR+SCD41 Fire Detection - Model Evaluation")
    print("=" * 45)
    
    # Download model
    model_path = download_model()
    if not model_path:
        print("Failed to download model")
        return
    
    # Load validation data
    data = load_validation_data()
    if not data:
        print("Failed to load validation data")
        return
    
    X_test, y_test = data
    
    # Evaluate model
    results = evaluate_model(model_path, X_test, y_test)
    
    if results:
        print("\n✅ Model evaluation completed successfully!")
        print("\nBased on these results, you can decide whether to proceed with deployment.")
        print("Generally, an AUC above 0.7 is considered acceptable for many applications.")
    else:
        print("\n❌ Model evaluation failed.")

if __name__ == "__main__":
    main()