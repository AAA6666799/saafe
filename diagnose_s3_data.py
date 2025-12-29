#!/usr/bin/env python3
import boto3
import json

def diagnose_s3_buckets():
    """Diagnose S3 bucket contents to find datasets"""
    s3_client = boto3.client('s3')
    
    print("=== S3 BUCKET DIAGNOSIS ===\n")
    
    # List all buckets
    try:
        buckets = s3_client.list_buckets()
        print("Available S3 Buckets:")
        for bucket in buckets['Buckets']:
            print(f"  - {bucket['Name']}")
        print()
    except Exception as e:
        print(f"Error listing buckets: {e}")
        return
    
    # Ask user for input bucket
    input_bucket = input("Enter your input bucket name (where raw data is stored): ").strip()
    
    if not input_bucket:
        print("No bucket name provided!")
        return
    
    # Check bucket contents
    try:
        print(f"\nContents of s3://{input_bucket}:")
        
        # List all objects
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=input_bucket)
        
        csv_files = []
        all_files = []
        
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    size = obj['Size']
                    all_files.append(key)
                    
                    print(f"  {key} ({size:,} bytes)")
                    
                    if key.endswith('.csv'):
                        csv_files.append(key)
        
        print(f"\nSummary:")
        print(f"Total files: {len(all_files)}")
        print(f"CSV files found: {len(csv_files)}")
        
        if csv_files:
            print(f"\nCSV files:")
            for csv_file in csv_files:
                print(f"  - {csv_file}")
            
            # Check if they're in a specific folder
            folders = set()
            for csv_file in csv_files:
                if '/' in csv_file:
                    folder = '/'.join(csv_file.split('/')[:-1])
                    folders.add(folder)
            
            if folders:
                print(f"\nCSV files are in these folders:")
                for folder in folders:
                    print(f"  - {folder}/")
                
                # Suggest the correct prefix
                most_common_folder = max(folders, key=lambda f: sum(1 for csv in csv_files if csv.startswith(f)))
                print(f"\nSuggested dataset_prefix: '{most_common_folder}/'")
        else:
            print("\nNo CSV files found! Please check:")
            print("1. Are your datasets uploaded to this bucket?")
            print("2. Are they in CSV format?")
            print("3. Do you have the correct bucket name?")
    
    except Exception as e:
        print(f"Error accessing bucket {input_bucket}: {e}")
        print("Make sure:")
        print("1. The bucket name is correct")
        print("2. You have permission to access this bucket")
        print("3. The bucket exists in your current AWS region")

if __name__ == "__main__":
    diagnose_s3_buckets()