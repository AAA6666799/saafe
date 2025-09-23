#!/usr/bin/env python3
"""
Very simple debug training script to test basic functionality
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
    
    print("=== Very Simple Debug Training Script ===")
    print(f"Data path: {args.data_path}")
    print(f"Model dir: {args.model_dir}")
    print()
    
    # Check if data directory exists
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data directory {args.data_path} does not exist")
        sys.exit(1)
    
    # List contents of data directory
    print(f"Contents of data directory:")
    try:
        contents = os.listdir(args.data_path)
        for item in contents:
            print(f"  {item}")
    except Exception as e:
        print(f"Error listing data directory: {e}")
        sys.exit(1)
    
    print()
    print("✅ Very simple debug training completed successfully!")
    
    # Create a simple model file to satisfy SageMaker
    try:
        os.makedirs(args.model_dir, exist_ok=True)
        with open(os.path.join(args.model_dir, "debug_model.txt"), "w") as f:
            f.write("Debug model created successfully")
        print(f"✅ Debug model saved to {os.path.join(args.model_dir, 'debug_model.txt')}")
    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)