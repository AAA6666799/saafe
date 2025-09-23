#!/usr/bin/env python3
"""
Real-time check for the most recent files in S3 bucket
"""

import boto3
from datetime import datetime, timedelta
import pytz

def real_time_check():
    """Check for the most recent files in real-time."""
    print("🔍 Real-Time S3 Bucket Check")
    print("=" * 40)
    
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Get current time in different timezones
        utc_now = datetime.now(pytz.UTC)
        local_now = datetime.now()
        
        print(f"Current Times:")
        print(f"  UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Local: {local_now.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # List objects in the bucket with sorting by last modified
        print("Checking S3 bucket: data-collector-of-first-device")
        response = s3_client.list_objects_v2(
            Bucket='data-collector-of-first-device'
        )
        
        if 'Contents' not in response:
            print("❌ No files found in the bucket")
            return
        
        print(f"✅ Found {response['KeyCount']} files in the bucket")
        
        # Sort files by last modified (most recent first)
        sorted_files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
        
        print(f"\n🔥 Most Recent Files:")
        print("-" * 30)
        
        # Show top 10 most recent files
        for i, obj in enumerate(sorted_files[:10]):
            key = obj['Key']
            last_modified = obj['LastModified']
            size = obj['Size']
            
            print(f"{i+1:2d}. {last_modified.strftime('%Y-%m-%d %H:%M:%S %Z')} | {key} ({size} bytes)")
            
            # Check if this file is from today and within the last hour
            time_diff = utc_now - last_modified
            if time_diff < timedelta(hours=1):
                print(f"    🟢 Within last hour ({time_diff})")
            elif time_diff < timedelta(days=1):
                print(f"    🟡 Within last 24 hours ({time_diff})")
            else:
                print(f"    🔴 Older than 24 hours ({time_diff})")
        
        # Check specifically for files from today with "20250909" in the name
        print(f"\n📅 Files from Today (20250909):")
        print("-" * 30)
        today_files = [obj for obj in sorted_files if '20250909' in obj['Key']]
        
        if today_files:
            for i, obj in enumerate(today_files[:10]):
                key = obj['Key']
                last_modified = obj['LastModified']
                size = obj['Size']
                
                print(f"{i+1:2d}. {last_modified.strftime('%Y-%m-%d %H:%M:%S %Z')} | {key} ({size} bytes)")
                
                # Check if within last hour
                time_diff = utc_now - last_modified
                if time_diff < timedelta(hours=1):
                    print(f"    🟢 Within last hour ({time_diff})")
        else:
            print("No files from today found")
            
        # Check for the specific file you mentioned
        print(f"\n🎯 Checking for Specific File:")
        print("-" * 30)
        target_file = "gas_data_20250909_132755.csv"
        
        matching_files = [obj for obj in sorted_files if target_file in obj['Key']]
        
        if matching_files:
            obj = matching_files[0]
            key = obj['Key']
            last_modified = obj['LastModified']
            size = obj['Size']
            
            print(f"Found file: {key}")
            print(f"Last modified: {last_modified.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # Check if within last hour
            time_diff = utc_now - last_modified
            print(f"Age: {time_diff}")
            
            if time_diff < timedelta(hours=1):
                print("🟢 This file is within the last hour - LIVE DATA DETECTED!")
            else:
                print("🔴 This file is older than one hour")
        else:
            print(f"File {target_file} not found in the bucket")
            
    except Exception as e:
        print(f"❌ Error during real-time check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    real_time_check()