#!/usr/bin/env python3
"""
Simple debug training script to test data loading
"""

import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()
    
    print("=== Simple Debug Training Script ===")
    print(f"Data path: {args.data_path}")
    print(f"Model dir: {args.model_dir}")
    
    # Check if data directory exists
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data directory {args.data_path} does not exist")
        sys.exit(1)
    
    # List contents of data directory
    print(f"Contents of data directory: {os.listdir(args.data_path)}")
    
    # Try to load JSON data
    try:
        data_files = [f for f in os.listdir(args.data_path) if f.endswith(".json")]
        print(f"Found {len(data_files)} JSON files")
        
        if not data_files:
            print("ERROR: No JSON files found")
            sys.exit(1)
            
        # Load the first data file
        data_file = data_files[0]
        print(f"Loading data from {data_file}")
        
        with open(os.path.join(args.data_path, data_file), "r") as f:
            data = json.load(f)
        
        print(f"Successfully loaded data with {len(data.get('samples', []))} samples")
        print("Training would continue here...")
        
        # Create a simple model file to satisfy SageMaker
        os.makedirs(args.model_dir, exist_ok=True)
        with open(os.path.join(args.model_dir, "model.txt"), "w") as f:
            f.write("Debug model - data loading successful")
        
        print("✅ Simple debug training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during data loading: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()