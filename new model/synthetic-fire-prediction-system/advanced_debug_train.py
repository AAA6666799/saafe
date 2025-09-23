#!/usr/bin/env python3
"""
Advanced debug training script - tests data loading, processing, and simple model training
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_json_data(data_dir):
    """Load JSON data from directory."""
    print(f"Looking for JSON files in {data_dir}")
    try:
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")
            
        data_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        print(f"Found {len(data_files)} JSON files: {data_files}")
        
        if not data_files:
            raise ValueError(f"No JSON files found in {data_dir}")
        
        # Load the first data file
        data_file = data_files[0]
        full_path = os.path.join(data_dir, data_file)
        print(f"Loading data from {full_path}")
        
        with open(full_path, "r") as f:
            data = json.load(f)
        
        print(f"Successfully loaded data with {len(data.get('samples', []))} samples")
        return data
    except Exception as e:
        print(f"Error in load_json_data: {e}")
        raise

def convert_to_dataframe(data):
    """Convert JSON data to DataFrame."""
    print("Converting JSON data to DataFrame")
    try:
        # Extract samples
        samples = data["samples"]
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            **sample["features"],
            "label": sample["label"]
        } for sample in samples])
        
        print(f"DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error in convert_to_dataframe: {e}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()
    
    print("=== Advanced Debug Training Script ===")
    print(f"Data path: {args.data_path}")
    print(f"Model dir: {args.model_dir}")
    print()
    
    try:
        # Check if data directory exists
        if not os.path.exists(args.data_path):
            print(f"ERROR: Data directory {args.data_path} does not exist")
            sys.exit(1)
        
        # List contents of data directory
        print(f"Contents of data directory:")
        contents = os.listdir(args.data_path)
        for item in contents:
            print(f"  {item}")
        print()
        
        # Try to load JSON data
        print("Attempting to load JSON data...")
        data = load_json_data(args.data_path)
        
        # Try to convert to DataFrame
        print("Attempting to convert data to DataFrame...")
        df = convert_to_dataframe(data)
        
        # Show some basic info
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Label distribution:")
        print(df['label'].value_counts())
        print()
        
        # Prepare features and target
        print("Preparing features and target...")
        feature_columns = [col for col in df.columns if col != 'label']
        X = df[feature_columns]
        y = df['label']
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print()
        
        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print()
        
        # Train a simple model
        print("Training a simple Random Forest model...")
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=1)
        model.fit(X_train, y_train)
        
        # Make predictions
        print("Making predictions...")
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        print()
        
        # Show classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print()
        
        print("✅ Advanced debug training completed successfully!")
        
        # Save the model
        print("Saving model...")
        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(args.model_dir, "advanced_debug_model.joblib")
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Also save a simple text file with results
        results_path = os.path.join(args.model_dir, "training_results.txt")
        with open(results_path, "w") as f:
            f.write(f"Advanced Debug Training Results\n")
            f.write(f"===============================\n")
            f.write(f"Data shape: {df.shape}\n")
            f.write(f"Train samples: {X_train.shape[0]}\n")
            f.write(f"Test samples: {X_test.shape[0]}\n")
            f.write(f"Model accuracy: {accuracy:.4f}\n")
            f.write(f"Features used: {len(feature_columns)}\n")
        print(f"✅ Results saved to {results_path}")
        
    except Exception as e:
        print(f"❌ ERROR during advanced debug training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()