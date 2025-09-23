#!/usr/bin/env python3
"""
Test script to verify tar file contents
"""

import tarfile
import tempfile
import os

def test_tar_contents():
    """Test the tar file contents"""
    # Create a temporary directory for our code
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy the working training and inference scripts like in the working example
        os.system(f"cp /Volumes/Ajay/saafe\\ copy\\ 3/new\\ model/synthetic-fire-prediction-system/flir_scd41_sagemaker_training_fixed.py {temp_dir}/train")
        os.system(f"cp /Volumes/Ajay/saafe\\ copy\\ 3/new\\ model/synthetic-fire-prediction-system/flir_scd41_inference_fixed.py {temp_dir}/serve")
        
        # Make scripts executable
        os.chmod(f"{temp_dir}/train", 0o755)
        os.chmod(f"{temp_dir}/serve", 0o755)
        
        # Verify files exist before packaging
        print(f"Train file exists: {os.path.exists(f'{temp_dir}/train')}")
        print(f"Serve file exists: {os.path.exists(f'{temp_dir}/serve')}")
        
        if os.path.exists(f"{temp_dir}/train"):
            print(f"Train file size: {os.path.getsize(f'{temp_dir}/train')} bytes")
        
        if os.path.exists(f"{temp_dir}/serve"):
            print(f"Serve file size: {os.path.getsize(f'{temp_dir}/serve')} bytes")
        
        # Create tar.gz file
        code_tar_path = f"{temp_dir}/test_code.tar.gz"
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
                # Get file info
                member_info = tar.getmember(member)
                print(f"    Size: {member_info.size} bytes")
                print(f"    Mode: {oct(member_info.mode)}")
        
        # Extract and verify
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        with tarfile.open(code_tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        
        print(f"\nExtracted contents:")
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, extract_dir)
                print(f"  {rel_path}")
                if os.path.exists(file_path):
                    print(f"    Size: {os.path.getsize(file_path)} bytes")
                    print(f"    Mode: {oct(os.stat(file_path).st_mode)}")
        
        print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_tar_contents()