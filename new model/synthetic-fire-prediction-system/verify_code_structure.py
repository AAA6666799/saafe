#!/usr/bin/env python3
"""
Verify the structure of our code package
"""

import tarfile
import tempfile
import os

def verify_code_structure():
    """Verify the structure of our code package"""
    # Create a temporary directory for our code
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy our new ensemble training and inference scripts
        os.system(f"cp /Volumes/Ajay/saafe\\ copy\\ 3/new\\ model/synthetic-fire-prediction-system/flir_scd41_sagemaker_training_100k_ensemble.py {temp_dir}/train")
        os.system(f"cp /Volumes/Ajay/saafe\\ copy\\ 3/new\\ model/synthetic-fire-prediction-system/flir_scd41_inference_100k_ensemble.py {temp_dir}/serve")
        
        # Make scripts executable
        os.chmod(f"{temp_dir}/train", 0o755)
        os.chmod(f"{temp_dir}/serve", 0o755)
        
        # Create tar.gz file
        code_tar_path = f"{temp_dir}/code_verification.tar.gz"
        with tarfile.open(code_tar_path, "w:gz") as tar:
            # Add files with correct names for SageMaker
            tar.add(f"{temp_dir}/train", arcname="train")
            tar.add(f"{temp_dir}/serve", arcname="serve")
        
        # Verify tar file was created
        print(f"Tar file exists: {os.path.exists(code_tar_path)}")
        if os.path.exists(code_tar_path):
            print(f"Tar file size: {os.path.getsize(code_tar_path)} bytes")
        
        # Debug: List contents of tar file
        print("\nTar file contents:")
        with tarfile.open(code_tar_path, "r:gz") as tar:
            members = tar.getnames()
            for member in members:
                print(f"  {member}")
        
        # Extract and verify structure
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        with tarfile.open(code_tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        
        print(f"\nExtracted structure:")
        for root, dirs, files in os.walk(extract_dir):
            level = root.replace(extract_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        # Check if files are executable
        for file in ['train', 'serve']:
            file_path = os.path.join(extract_dir, file)
            if os.path.exists(file_path):
                mode = os.stat(file_path).st_mode
                is_executable = bool(mode & 0o111)
                print(f"\n{file} is executable: {is_executable}")
                print(f"{file} mode: {oct(mode)}")
        
        print("\n✅ Verification completed!")

if __name__ == "__main__":
    verify_code_structure()