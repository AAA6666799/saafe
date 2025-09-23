#!/usr/bin/env python3
"""
Debug script to check what data is available in the S3 bucket
"""

import boto3
import json
from datetime import datetime, timedelta
import pytz

def debug_s3_data():
    """Debug what data is available in the S3 bucket"""
    print("🔍 Debugging S3 data...")
    print("=" * 50)
    
    try:
        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'data-collector-of-first-device'
        
        # List all objects in the bucket
        print(f"Listing objects in bucket: {bucket_name}")
        response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=20)
        
        if 'Contents' in response:
            print(f"Found {len(response['Contents'])} objects (showing up to 20):")
            print("-" * 50)
            
            # Sort by last modified (most recent first)
            objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
            
            for i, obj in enumerate(objects[:20]):
                key = obj['Key']
                size = obj['Size']
                modified = obj['LastModified']
                print(f"{i+1:2d}. {key}")
                print(f"     Size: {size} bytes, Last Modified: {modified}")
                
                # Try to download and parse a few sample files
                if i < 3:  # Only try to parse the first 3 files
                    try:
                        file_response = s3.get_object(Bucket=bucket_name, Key=key)
                        content = file_response['Body'].read().decode('utf-8')
                        
                        # Try to parse as JSON
                        try:
                            data = json.loads(content)
                            print(f"     Content: {str(data)[:100]}...")
                        except json.JSONDecodeError:
                            # If not JSON, show first 100 characters
                            print(f"     Content (text): {content[:100]}...")
                    except Exception as e:
                        print(f"     Error reading file: {e}")
                
                print()
        else:
            print("No objects found in the bucket")
            
    except Exception as e:
        print(f"❌ Error accessing S3 bucket: {e}")

if __name__ == "__main__":
    debug_s3_data()