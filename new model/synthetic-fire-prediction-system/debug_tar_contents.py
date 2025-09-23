#!/usr/bin/env python3
"""
Debug script to verify tar file contents in detail
"""

import tarfile
import tempfile
import os

def debug_tar_contents():
    """Debug the tar file contents in detail"""
    # Create a temporary directory for our code
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy our new ensemble training and inference scripts
        os.system(f"cp /Volumes/Ajay/saafe\\ copy\\ 3/new\\ model/synthetic-fire-prediction-system/flir_scd41_sagemaker_training_100k_ensemble.py {temp_dir}/train")
        os.system(f"cp /Volumes/Ajay/saafe\\ copy\\ 3/new\\ model/synthetic-fire-prediction-system/flir_scd41_inference_100k_ensemble.py {temp_dir}/serve")
        
        # Make scripts executable
        os.chmod(f"{temp_dir}/train", 0o755)
        os.chmod(f"{temp_dir}/serve", 0o755)
        
        # Verify files exist before packaging
        print(f"Train file exists: {os.path.exists(f'{temp_dir}/train')}")
        print(f"Serve file exists: {os.path.exists(f'{temp_dir}/serve')}")
        
        if os.path.exists(f"{temp_dir}/train"):
            print(f"Train file size: {os.path.getsize(f'{temp_dir}/train')} bytes")
            # Show first few lines of train file
            with open(f"{temp_dir}/train", "r") as f:
                print("First 5 lines of train file:")
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"  {line.rstrip()}")
                    else:
                        break
        
        if os.path.exists(f"{temp_dir}/serve"):
            print(f"Serve file size: {os.path.getsize(f'{temp_dir}/serve')} bytes")
            # Show first few lines of serve file
            with open(f"{temp_dir}/serve", "r") as f:
                print("First 5 lines of serve file:")
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"  {line.rstrip()}")
                    else:
                        break
        
        # Create tar.gz file
        code_tar_path = f"{temp_dir}/debug_code.tar.gz"
        with tarfile.open(code_tar_path, "w:gz") as tar:
            # Add files with correct names for SageMaker
            tar.add(f"{temp_dir}/train", arcname="train")
            tar.add(f"{temp_dir}/serve", arcname="serve")
        
        # Verify tar file was created
        print(f"Tar file exists: {os.path.exists(code_tar_path)}")
        if os.path.exists(code_tar_path):
            print(f"Tar file size: {os.path.getsize(code_tar_path)} bytes")
        
        # Debug: List contents of tar file with detailed info
        print("\nTar file contents (detailed):")
        with tarfile.open(code_tar_path, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                print(f"  Name: {member.name}")
                print(f"    Size: {member.size} bytes")
                print(f"    Mode: {oct(member.mode)}")
                print(f"    Type: {member.type}")
                print(f"    Linkname: {member.linkname}")
                print(f"    Uid: {member.uid}")
                print(f"    Gid: {member.gid}")
                print(f"    Uname: {member.uname}")
                print(f"    Gname: {member.gname}")
        
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
                    # Show first few lines of extracted files
                    with open(file_path, "r") as f:
                        print("    First 3 lines:")
                        for i, line in enumerate(f):
                            if i < 3:
                                print(f"      {line.rstrip()}")
                            else:
                                break
        
        print("\n✅ Debug completed successfully!")

if __name__ == "__main__":
    debug_tar_contents()