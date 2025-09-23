#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - SageMaker Training Script (Fixed)
This script is compatible with SageMaker's training environment and converts JSON data to CSV.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_json_data(data_dir):
    """Load JSON data from directory."""
    data_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    if not data_files:
        raise ValueError(f"No JSON files found in {data_dir}")
    
    # Load the first data file
    data_file = data_files[0]
    with open(os.path.join(data_dir, data_file), "r") as f:
        data = json.load(f)
    
    return data

def convert_to_csv(data, output_path):
    """Convert JSON data to CSV format."""
    # Extract samples
    samples = data["samples"]
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        **sample["features"],
        "label": sample["label"]
    } for sample in samples])
    
    # Save as CSV
    df.to_csv(output_path, index=False)
    print(f"Data converted to CSV and saved to {output_path}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--output-data-dir", type=str, default="/opt/ml/output/data")
    args = parser.parse_args()
    
    print("Starting FLIR+SCD41 training...")
    
    try:
        # Load data
        print(f"Loading data from {args.data_path}")
        data = load_json_data(args.data_path)
        print(f"Loaded {len(data['samples'])} samples")
        
        # Convert to CSV for algorithm mode compatibility
        csv_path = "/opt/ml/processing/train_data.csv"
        df = convert_to_csv(data, csv_path)
        
        # Prepare features and target
        X = df.drop("label", axis=1)
        y = df["label"]
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        print("Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Save model
        os.makedirs(args.model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
        print(f"Model saved to {args.model_dir}")
        
        # Save evaluation results
        with open(os.path.join(args.model_dir, "evaluation.json"), "w") as f:
            json.dump({
                "accuracy": float(accuracy),
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            }, f)
        print("Evaluation results saved")
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise