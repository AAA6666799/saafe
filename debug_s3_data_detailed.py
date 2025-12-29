#!/usr/bin/env python3
"""
Detailed debug script to check what data is available in the S3 bucket
"""

import boto3
import json
import csv
from io import StringIO
from datetime import datetime, timedelta
import pytz

def debug_s3_data_detailed():
    """Debug what data is available in the S3 bucket in detail"""
    print("🔍 Detailed S3 data debugging...")
    print("=" * 50)
    
    try:
        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'data-collector-of-first-device'
        
        # Check for thermal data files
        print("Checking for thermal data files...")
        thermal_response = s3.list_objects_v2(
            Bucket=bucket_name, 
            Prefix='thermal-data/',
            MaxKeys=10
        )
        
        if 'Contents' in thermal_response:
            print(f"Found {len(thermal_response['Contents'])} thermal data files:")
            for obj in thermal_response['Contents'][:5]:  # Show first 5
                key = obj['Key']
                size = obj['Size']
                modified = obj['LastModified']
                print(f"  {key} ({size} bytes, {modified})")
        else:
            print("No thermal data files found")
        
        print("\n" + "-" * 50)
        
        # Check for gas data files
        print("Checking for gas data files...")
        gas_response = s3.list_objects_v2(
            Bucket=bucket_name, 
            Prefix='gas-data/',
            MaxKeys=10
        )
        
        if 'Contents' in gas_response:
            print(f"Found {len(gas_response['Contents'])} gas data files:")
            for obj in gas_response['Contents'][:5]:  # Show first 5
                key = obj['Key']
                size = obj['Size']
                modified = obj['LastModified']
                print(f"  {key} ({size} bytes, {modified})")
        else:
            print("No gas data files found")
        
        print("\n" + "-" * 50)
        
        # Check the most recent files regardless of type
        print("Checking most recent files (any type)...")
        all_response = s3.list_objects_v2(
            Bucket=bucket_name, 
            MaxKeys=10
        )
        
        if 'Contents' in all_response:
            # Sort by last modified (most recent first)
            objects = sorted(all_response['Contents'], key=lambda x: x['LastModified'], reverse=True)
            print("Most recent files:")
            for i, obj in enumerate(objects[:10]):
                key = obj['Key']
                size = obj['Size']
                modified = obj['LastModified']
                print(f"  {i+1}. {key} ({size} bytes, {modified})")
                
                # Try to download and parse the most recent file
                if i == 0:
                    print(f"\n  Contents of most recent file ({key}):")
                    try:
                        file_response = s3.get_object(Bucket=bucket_name, Key=key)
                        content = file_response['Body'].read().decode('utf-8')
                        
                        # Show first 200 characters
                        print(f"    First 200 characters: {content[:200]}")
                        
                        # Try to parse as CSV
                        if '.csv' in key:
                            try:
                                csv_reader = csv.reader(StringIO(content))
                                rows = list(csv_reader)
                                print(f"    CSV has {len(rows)} rows")
                                if len(rows) > 0:
                                    print(f"    Headers: {rows[0]}")
                                if len(rows) > 1:
                                    print(f"    First data row: {rows[1]}")
                            except Exception as e:
                                print(f"    Error parsing CSV: {e}")
                    except Exception as e:
                        print(f"    Error reading file: {e}")
        else:
            print("No files found in the bucket")
            
    except Exception as e:
        print(f"❌ Error accessing S3 bucket: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_s3_data_detailed()