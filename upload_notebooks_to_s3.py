#!/usr/bin/env python3
"""
Script to upload Fire Detection AI training notebooks to an S3 bucket.
"""

import boto3
import os
import argparse
from tqdm import tqdm

def upload_file_to_s3(file_path, bucket_name, s3_key, s3_client):
    """Upload a file to an S3 bucket."""
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        return True
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Upload notebooks to S3 bucket')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', default='notebooks', help='S3 prefix (folder) for notebooks')
    args = parser.parse_args()
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # List of notebooks to upload
    notebooks = [
        # 5M training notebooks (simplified and parts)
        "fire_detection_5m_training_simplified.ipynb",
        "fire_detection_5m_training_part1.ipynb",
        "fire_detection_5m_training_part2.ipynb",
        "fire_detection_5m_training_part3.ipynb",
        "fire_detection_5m_training_part4.ipynb",
        "fire_detection_5m_training_part5.ipynb",
        "fire_detection_5m_training_part6.ipynb",
        "fire_detection_5m_training.ipynb",
        
        # Additional relevant notebooks
        "AWS_SageMaker_50M_Training.ipynb",
        "Fire_Detection_SageMaker_50M.ipynb",
        "SageMaker_Fire_Detection_50M_Clean.ipynb",
        "data_cleaning_sagemaker.ipynb",
        "model_training_sagemaker.ipynb",
        "fire_prediction_training_sagemaker.ipynb"
    ]
    
    # Check if bucket exists, create if it doesn't
    try:
        s3_client.head_bucket(Bucket=args.bucket)
        print(f"Bucket {args.bucket} exists.")
    except:
        print(f"Bucket {args.bucket} does not exist. Creating...")
        try:
            s3_client.create_bucket(Bucket=args.bucket)
            print(f"Bucket {args.bucket} created successfully.")
        except Exception as e:
            print(f"Error creating bucket: {e}")
            return
    
    # Upload notebooks
    print(f"Uploading notebooks to s3://{args.bucket}/{args.prefix}/")
    successful_uploads = 0
    
    for notebook in tqdm(notebooks, desc="Uploading notebooks"):
        if os.path.exists(notebook):
            s3_key = f"{args.prefix}/{notebook}"
            if upload_file_to_s3(notebook, args.bucket, s3_key, s3_client):
                successful_uploads += 1
                print(f"✅ Uploaded {notebook} to s3://{args.bucket}/{s3_key}")
            else:
                print(f"❌ Failed to upload {notebook}")
        else:
            print(f"⚠️ Notebook {notebook} not found in current directory")
    
    # Upload supporting files
    supporting_files = [
        "requirements_gpu.txt",
        "fire_detection_50m_config.json",
        "minimal_fire_ensemble_config.json"
    ]
    
    print("\nUploading supporting files...")
    for file in tqdm(supporting_files, desc="Uploading supporting files"):
        if os.path.exists(file):
            s3_key = f"{args.prefix}/supporting_files/{file}"
            if upload_file_to_s3(file, args.bucket, s3_key, s3_client):
                successful_uploads += 1
                print(f"✅ Uploaded {file} to s3://{args.bucket}/{s3_key}")
            else:
                print(f"❌ Failed to upload {file}")
        else:
            print(f"⚠️ File {file} not found in current directory")
    
    print(f"\nUpload complete! Successfully uploaded {successful_uploads} files.")
    print(f"Notebooks are available at: s3://{args.bucket}/{args.prefix}/")

if __name__ == "__main__":
    main()