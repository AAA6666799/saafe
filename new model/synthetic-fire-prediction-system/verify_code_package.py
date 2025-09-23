#!/usr/bin/env python3
"""
Verify the contents of the code package
"""

import boto3
import tarfile
import tempfile
import os

def verify_code_package():
    """Download and verify the code package"""
    s3 = boto3.client('s3', region_name='us-east-1')
    
    bucket = 'fire-detection-training-691595239825'
    key = 'flir_scd41_training/code/fixed_ensemble_code.tar.gz'
    
    print(f"Verifying code package: s3://{bucket}/{key}")
    print("=" * 50)
    
    try:
        # Download the code package
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = os.path.join(temp_dir, 'code_package.tar.gz')
            print(f"Downloading code package...")
            s3.download_file(bucket, key, local_path)
            print(f"✅ Downloaded successfully")
            
            # Extract and list contents
            print(f"\nContents of code package:")
            with tarfile.open(local_path, 'r:gz') as tar:
                members = tar.getnames()
                for member in members:
                    print(f"  {member}")
            
            # Verify required files exist
            required_files = ['train', 'serve']
            missing_files = [f for f in required_files if f not in members]
            
            if not missing_files:
                print(f"\n✅ All required files present: {required_files}")
            else:
                print(f"\n❌ Missing files: {missing_files}")
                
            # Check file permissions
            with tarfile.open(local_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        print(f"  {member.name}: {oct(member.mode)} permissions")
                        
    except Exception as e:
        print(f"❌ Error verifying code package: {e}")

if __name__ == "__main__":
    verify_code_package()