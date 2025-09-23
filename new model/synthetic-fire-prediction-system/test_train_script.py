#!/usr/bin/env python3
"""
Test script to verify our training script can run
"""

import subprocess
import tempfile
import os

def test_train_script():
    """Test our training script"""
    # Create a temporary directory for our code
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy our new ensemble training script
        os.system(f"cp /Volumes/Ajay/saafe\\ copy\\ 3/new\\ model/synthetic-fire-prediction-system/flir_scd41_sagemaker_training_100k_ensemble.py {temp_dir}/train")
        
        # Make script executable
        os.chmod(f"{temp_dir}/train", 0o755)
        
        # Verify file exists
        print(f"Train file exists: {os.path.exists(f'{temp_dir}/train')}")
        if os.path.exists(f'{temp_dir}/train'):
            print(f"Train file size: {os.path.getsize(f'{temp_dir}/train')} bytes")
            
            # Show first 10 lines
            with open(f'{temp_dir}/train', 'r') as f:
                print("First 10 lines of train script:")
                for i, line in enumerate(f):
                    if i < 10:
                        print(f"  {line.rstrip()}")
                    else:
                        break
        
        # Try to run the script with --help to see if it can start
        try:
            print("\nTrying to run train script with --help...")
            result = subprocess.run([f"{temp_dir}/train", "--help"], 
                                  capture_output=True, text=True, timeout=10)
            print(f"Return code: {result.returncode}")
            print(f"Stdout: {result.stdout}")
            if result.stderr:
                print(f"Stderr: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("Script timed out (this might be expected)")
        except Exception as e:
            print(f"Error running script: {e}")
        
        # Try to run the script without arguments
        try:
            print("\nTrying to run train script without arguments...")
            result = subprocess.run([f"{temp_dir}/train"], 
                                  capture_output=True, text=True, timeout=5)
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print(f"Stdout: {result.stdout}")
            if result.stderr:
                print(f"Stderr: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("Script timed out (this might be expected)")
        except Exception as e:
            print(f"Error running script: {e}")
        
        print("\n✅ Test completed!")

if __name__ == "__main__":
    test_train_script()