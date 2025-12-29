#!/usr/bin/env python3
"""
Script to test S3 connection and permissions for Fire Detection AI training.
This script verifies that you have the necessary permissions to upload and download from S3.
"""

import boto3
import argparse
import os
import uuid
import json
from datetime import datetime

def test_s3_permissions(bucket_name, prefix):
    """Test S3 permissions by creating a test file, uploading it, and downloading it."""
    print(f"Testing S3 permissions for bucket: {bucket_name}")
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # Generate a unique test file name
    test_file_name = f"test_file_{uuid.uuid4().hex[:8]}.json"
    local_path = f"/tmp/{test_file_name}"
    s3_key = f"{prefix}/test/{test_file_name}"
    
    # Create a test file
    test_data = {
        "test_id": uuid.uuid4().hex,
        "timestamp": datetime.now().isoformat(),
        "message": "This is a test file for Fire Detection AI training on SageMaker."
    }
    
    with open(local_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Created test file: {local_path}")
    
    # Check if bucket exists
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✅ Bucket {bucket_name} exists and is accessible.")
    except Exception as e:
        print(f"❌ Error accessing bucket {bucket_name}: {e}")
        print("Please check if the bucket exists and you have the necessary permissions.")
        return False
    
    # Upload test file
    try:
        s3_client.upload_file(local_path, bucket_name, s3_key)
        print(f"✅ Successfully uploaded test file to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"❌ Error uploading to S3: {e}")
        print("Please check your write permissions for this bucket.")
        return False
    
    # Download test file
    download_path = f"/tmp/downloaded_{test_file_name}"
    try:
        s3_client.download_file(bucket_name, s3_key, download_path)
        print(f"✅ Successfully downloaded test file from S3 to {download_path}")
    except Exception as e:
        print(f"❌ Error downloading from S3: {e}")
        print("Please check your read permissions for this bucket.")
        return False
    
    # Verify file contents
    try:
        with open(download_path, 'r') as f:
            downloaded_data = json.load(f)
        
        if downloaded_data["test_id"] == test_data["test_id"]:
            print("✅ File contents verified successfully.")
        else:
            print("❌ File contents do not match.")
            return False
    except Exception as e:
        print(f"❌ Error verifying file contents: {e}")
        return False
    
    # Clean up
    try:
        os.remove(local_path)
        os.remove(download_path)
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        print("✅ Cleaned up test files.")
    except Exception as e:
        print(f"⚠️ Warning: Could not clean up all test files: {e}")
    
    # Test listing objects
    try:
        s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        print(f"✅ Successfully listed objects in s3://{bucket_name}/{prefix}/")
    except Exception as e:
        print(f"❌ Error listing objects in S3: {e}")
        print("Please check your list permissions for this bucket.")
        return False
    
    print("\n✅ All S3 permission tests passed successfully!")
    print(f"You have the necessary permissions to use bucket: {bucket_name}")
    return True

def check_aws_identity():
    """Check AWS identity and account information."""
    try:
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        
        print("\nAWS Identity Information:")
        print(f"Account ID: {identity['Account']}")
        print(f"User/Role ARN: {identity['Arn']}")
        
        return True
    except Exception as e:
        print(f"\n❌ Error retrieving AWS identity: {e}")
        print("Please check your AWS credentials.")
        return False

def check_sagemaker_quota():
    """Check SageMaker service quotas for ml.p3.16xlarge instances."""
    try:
        servicequotas_client = boto3.client('service-quotas')
        
        # Get quota for ml.p3.16xlarge instances
        response = servicequotas_client.get_service_quota(
            ServiceCode='sagemaker',
            QuotaCode='L-3D4B9AF0'  # This is the quota code for ml.p3.16xlarge instances
        )
        
        quota_value = response['Quota']['Value']
        print(f"\nSageMaker ml.p3.16xlarge instance quota: {quota_value}")
        
        if quota_value < 1:
            print("⚠️ Warning: Your quota for ml.p3.16xlarge instances is less than 1.")
            print("You may need to request a quota increase to use this instance type.")
        else:
            print(f"✅ Your quota allows for {int(quota_value)} ml.p3.16xlarge instances.")
        
        return True
    except Exception as e:
        print("\n⚠️ Could not check SageMaker quota for ml.p3.16xlarge instances.")
        print("This is not critical, but you may want to check your quota in the AWS console.")
        print(f"Error: {e}")
        return True  # Return True as this is not critical

def main():
    parser = argparse.ArgumentParser(description='Test S3 connection and permissions')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', default='fire-detection-notebooks', help='S3 prefix (folder)')
    args = parser.parse_args()
    
    print("============================================")
    print("Fire Detection AI - S3 Connection Test")
    print("============================================\n")
    
    # Check AWS identity
    if not check_aws_identity():
        return
    
    # Test S3 permissions
    if not test_s3_permissions(args.bucket, args.prefix):
        print("\n❌ S3 permission test failed.")
        print("Please check your AWS credentials and bucket permissions.")
        return
    
    # Check SageMaker quota
    try:
        check_sagemaker_quota()
    except:
        pass
    
    print("\n============================================")
    print("✅ All tests completed successfully!")
    print("You are ready to upload and download notebooks for Fire Detection AI training.")
    print("============================================")

if __name__ == "__main__":
    main()