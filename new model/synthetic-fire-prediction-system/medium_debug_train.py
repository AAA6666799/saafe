#!/usr/bin/env python3
"""
Medium complexity debug training script - tests data loading
"""

import argparse
import json
import os
import sys
import pandas as pd

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

def convert_to_csv(data):
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
        print(f"Error in convert_to_csv: {e}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()
    
    print("=== Medium Debug Training Script ===")
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
        df = convert_to_csv(data)
        
        # Show some basic info
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Label distribution:")
        print(df['label'].value_counts())
        
        print("\n✅ Medium debug training completed successfully!")
        
        # Create a simple model file to satisfy SageMaker
        os.makedirs(args.model_dir, exist_ok=True)
        with open(os.path.join(args.model_dir, "medium_debug_model.txt"), "w") as f:
            f.write(f"Debug model created successfully\nData shape: {df.shape}")
        print(f"✅ Debug model saved to {os.path.join(args.model_dir, 'medium_debug_model.txt')}")
        
    except Exception as e:
        print(f"❌ ERROR during medium debug training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()