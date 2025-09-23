#!/usr/bin/env python3
"""
Script to check S3 bucket contents and verify live data
"""

import boto3
from datetime import datetime, timedelta
import json

def check_s3_contents():
    """Check S3 bucket contents and categorize files by age."""
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name='us-east-1')
    
    try:
        # List objects in the bucket
        response = s3_client.list_objects_v2(
            Bucket='data-collector-of-first-device',
            MaxKeys=100
        )
        
        if 'Contents' not in response:
            print("No files found in the bucket")
            return
        
        # Get current time
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)
        
        # Categorize files
        recent_files = []  # Last hour
        today_files = []   # Last 24 hours
        old_files = []     # Older than 24 hours
        
        thermal_count = 0
        gas_count = 0
        
        print("Files in S3 bucket 'data-collector-of-first-device':")
        print("=" * 60)
        
        for obj in response['Contents']:
            key = obj['Key']
            last_modified = obj['LastModified']
            
            # Count file types
            if 'thermal_data' in key:
                thermal_count += 1
            elif 'gas_data' in key:
                gas_count += 1
            
            # Categorize by age
            if last_modified > one_hour_ago.replace(tzinfo=last_modified.tzinfo):
                recent_files.append(obj)
            elif last_modified > one_day_ago.replace(tzinfo=last_modified.tzinfo):
                today_files.append(obj)
            else:
                old_files.append(obj)
            
            # Print file info
            print(f"{last_modified.strftime('%Y-%m-%d %H:%M:%S')} | {key}")
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(f"Total files: {response['KeyCount']}")
        print(f"Thermal data files: {thermal_count}")
        print(f"Gas data files: {gas_count}")
        print(f"Files from last hour: {len(recent_files)}")
        print(f"Files from last 24 hours: {len(today_files)}")
        print(f"Older files: {len(old_files)}")
        
        if recent_files:
            print("\n🔥 RECENT FILES (last hour) - LIVE DATA:")
            for obj in recent_files:
                print(f"  {obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')} | {obj['Key']}")
        else:
            print("\n⚠️  No recent files found in the last hour")
            if today_files:
                print("📅 Most recent files from today:")
                # Show the 5 most recent files
                sorted_today = sorted(today_files, key=lambda x: x['LastModified'], reverse=True)
                for obj in sorted_today[:5]:
                    print(f"  {obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')} | {obj['Key']}")
        
    except Exception as e:
        print(f"Error checking S3 contents: {e}")

if __name__ == "__main__":
    check_s3_contents()