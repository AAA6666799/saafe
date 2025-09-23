#!/usr/bin/env python3
"""
Debug script to test data loading specifically
"""

import argparse
import json
import os
import sys

def load_json_data(data_dir):
    """Load JSON data from directory."""
    print(f"Looking for JSON files in {data_dir}")
    data_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    print(f"Found {len(data_files)} JSON files: {data_files}")
    
    if not data_files:
        raise ValueError(f"No JSON files found in {data_dir}")
    
    # Load the first data file
    data_file = data_files[0]
    print(f"Loading data from {data_file}")
    
    with open(os.path.join(data_dir, data_file), "r") as f:
        data = json.load(f)
    
    print(f"Successfully loaded data with {len(data.get('samples', []))} samples")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()
    
    print("=== Data Loading Debug Script ===")
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
        print("✅ Data loading successful!")
        
        # Test accessing a sample
        if 'samples' in data and len(data['samples']) > 0:
            sample = data['samples'][0]
            print(f"First sample keys: {list(sample.keys())}")
            if 'features' in sample:
                print(f"Features keys: {list(sample['features'].keys())}")
            if 'label' in sample:
                print(f"Label: {sample['label']}")
        
        # Create a simple model file to satisfy SageMaker
        os.makedirs(args.model_dir, exist_ok=True)
        with open(os.path.join(args.model_dir, "data_loading_test.txt"), "w") as f:
            f.write("Data loading test successful")
        
        print("\n✅ Data loading debug completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during data loading: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()