#!/usr/bin/env python3
"""
Simple test script to verify S3 access and data parsing
"""

import boto3
import csv
import json
from io import StringIO
from datetime import datetime, timedelta
import pytz

def test_s3_access():
    """Test S3 access and data parsing"""
    print("🔍 Testing S3 access and data parsing...")
    print("=" * 50)
    
    try:
        # Initialize S3 client
        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'data-collector-of-first-device'
        
        # Test if we can access the bucket
        s3.head_bucket(Bucket=bucket_name)
        print(f"✅ S3 connection successful - Bucket '{bucket_name}' accessible")
        
        # List some objects
        response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=5)
        
        if 'Contents' in response:
            print(f"✅ Found {len(response['Contents'])} objects in bucket")
            
            # Get the most recent object
            objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
            latest_obj = objects[0]
            key = latest_obj['Key']
            print(f"Latest object: {key}")
            
            # Try to download and parse it
            file_response = s3.get_object(Bucket=bucket_name, Key=key)
            content = file_response['Body'].read().decode('utf-8')
            
            print(f"Content (first 200 chars): {content[:200]}")
            
            # Try to parse as CSV
            if '.csv' in key:
                csv_reader = csv.reader(StringIO(content))
                rows = list(csv_reader)
                print(f"✅ CSV parsing successful: {len(rows)} rows")
                
                if len(rows) > 0:
                    print(f"Headers: {rows[0]}")
                if len(rows) > 1:
                    print(f"First data row: {rows[1]}")
                    
                    # Try to convert values to appropriate types
                    headers = rows[0]
                    data_row = rows[1]
                    
                    data_dict = {}
                    for i, header in enumerate(headers):
                        if i < len(data_row):
                            try:
                                # Try to convert to float if possible
                                data_dict[header] = float(data_row[i])
                            except ValueError:
                                # Keep as string if not a number
                                data_dict[header] = data_row[i]
                    
                    print(f"Parsed data: {data_dict}")
            
        else:
            print("❌ No objects found in bucket")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_s3_access()