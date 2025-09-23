#!/usr/bin/env python3
"""
Minimal training script for testing
"""

import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--output-data-dir", type=str, default="/opt/ml/output/data")
    args = parser.parse_args()
    
    print("Minimal training script running successfully!")
    print(f"Data path: {args.data_path}")
    print(f"Model dir: {args.model_dir}")
    print(f"Output data dir: {args.output_data_dir}")
    
    # Create a minimal model file
    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, "minimal_model.txt"), "w") as f:
        f.write("This is a minimal model")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()