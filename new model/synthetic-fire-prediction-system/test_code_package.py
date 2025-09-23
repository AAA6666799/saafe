#!/usr/bin/env python3
"""
Test script to verify code package contents
"""

import tarfile
import tempfile
import os

def test_code_package():
    """Test the code packaging process"""
    # Create training script content
    training_script = '''#!/usr/bin/env python3
print("Training script")
'''
    
    # Create inference script content
    inference_script = '''#!/usr/bin/env python3
print("Inference script")
'''
    
    # Create a temporary directory for our code
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Write training and inference scripts to files with correct names
        train_path = os.path.join(temp_dir, 'train')
        serve_path = os.path.join(temp_dir, 'serve')
        
        # Write the training script
        with open(train_path, 'w') as f:
            f.write(training_script)
        
        # Write the inference script
        with open(serve_path, 'w') as f:
            f.write(inference_script)
        
        # Make scripts executable
        os.chmod(train_path, 0o755)
        os.chmod(serve_path, 0o755)
        
        # Verify files exist
        print(f"Train file exists: {os.path.exists(train_path)}")
        print(f"Serve file exists: {os.path.exists(serve_path)}")
        
        if os.path.exists(train_path):
            print(f"Train file size: {os.path.getsize(train_path)} bytes")
        
        if os.path.exists(serve_path):
            print(f"Serve file size: {os.path.getsize(serve_path)} bytes")
        
        # Create tar.gz file
        code_tar_path = f"{temp_dir}/test_code.tar.gz"
        with tarfile.open(code_tar_path, "w:gz") as tar:
            # Add files with correct names for SageMaker
            tar.add(train_path, arcname="train")
            tar.add(serve_path, arcname="serve")
        
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
        
        print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_code_package()