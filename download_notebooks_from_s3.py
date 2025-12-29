#!/usr/bin/env python3
"""
Script to download Fire Detection AI training notebooks from an S3 bucket to a SageMaker notebook instance.
This script should be run on the SageMaker notebook instance.
"""

import boto3
import os
import argparse
from tqdm import tqdm

def download_file_from_s3(bucket_name, s3_key, local_path, s3_client):
    """Download a file from an S3 bucket."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket_name, s3_key, local_path)
        return True
    except Exception as e:
        print(f"Error downloading s3://{bucket_name}/{s3_key}: {e}")
        return False

def list_objects_in_bucket(bucket_name, prefix, s3_client):
    """List objects in an S3 bucket with the given prefix."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            return [item['Key'] for item in response['Contents']]
        return []
    except Exception as e:
        print(f"Error listing objects in s3://{bucket_name}/{prefix}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Download notebooks from S3 bucket to SageMaker')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', default='notebooks', help='S3 prefix (folder) for notebooks')
    parser.add_argument('--output-dir', default='fire_detection_notebooks', help='Local directory to save notebooks')
    args = parser.parse_args()
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List all objects in the bucket with the given prefix
    print(f"Listing objects in s3://{args.bucket}/{args.prefix}/")
    s3_objects = list_objects_in_bucket(args.bucket, args.prefix, s3_client)
    
    if not s3_objects:
        print(f"No objects found in s3://{args.bucket}/{args.prefix}/")
        return
    
    # Filter for notebook files and supporting files
    notebook_files = [key for key in s3_objects if key.endswith('.ipynb')]
    supporting_files = [key for key in s3_objects if not key.endswith('.ipynb') and not key.endswith('/')]
    
    # Download notebooks
    print(f"Downloading {len(notebook_files)} notebooks to {args.output_dir}/")
    successful_downloads = 0
    
    for s3_key in tqdm(notebook_files, desc="Downloading notebooks"):
        filename = os.path.basename(s3_key)
        local_path = os.path.join(args.output_dir, filename)
        
        if download_file_from_s3(args.bucket, s3_key, local_path, s3_client):
            successful_downloads += 1
            print(f"✅ Downloaded {s3_key} to {local_path}")
        else:
            print(f"❌ Failed to download {s3_key}")
    
    # Download supporting files
    if supporting_files:
        print(f"\nDownloading {len(supporting_files)} supporting files...")
        for s3_key in tqdm(supporting_files, desc="Downloading supporting files"):
            filename = os.path.basename(s3_key)
            local_path = os.path.join(args.output_dir, "supporting_files", filename)
            
            if download_file_from_s3(args.bucket, s3_key, local_path, s3_client):
                successful_downloads += 1
                print(f"✅ Downloaded {s3_key} to {local_path}")
            else:
                print(f"❌ Failed to download {s3_key}")
    
    print(f"\nDownload complete! Successfully downloaded {successful_downloads} files to {args.output_dir}/")
    print("You can now open these notebooks in JupyterLab.")

if __name__ == "__main__":
    main()