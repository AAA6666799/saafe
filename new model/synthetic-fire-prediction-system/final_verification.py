#!/usr/bin/env python3
"""
Final verification that the dashboard should detect live data
"""

import boto3
from datetime import datetime, timedelta
import pytz

def final_verification():
    """Final verification of live data detection."""
    print("✅ Final Live Data Verification")
    print("=" * 40)
    
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Get current time
        utc_now = datetime.now(pytz.UTC)
        one_hour_ago = utc_now - timedelta(hours=1)
        
        print(f"Current UTC Time: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"One Hour Ago: {one_hour_ago.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print()
        
        # Use pagination to get all files
        print("Checking S3 bucket: data-collector-of-first-device")
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket='data-collector-of-first-device',
            PaginationConfig={'MaxItems': 1000}  # Check up to 1000 most recent files
        )
        
        all_files = []
        for page in pages:
            if 'Contents' in page:
                all_files.extend(page['Contents'])
        
        print(f"✅ Found {len(all_files)} files in the bucket")
        
        # Sort files by last modified (most recent first)
        sorted_files = sorted(all_files, key=lambda x: x['LastModified'], reverse=True)
        
        # Filter for recent files (last hour)
        recent_files = []
        thermal_files = 0
        gas_files = 0
        
        for obj in sorted_files:
            # Check if file was modified in the last hour
            file_time = obj['LastModified']
            if file_time.tzinfo is None:
                # If no timezone info, assume it's UTC
                file_time = file_time.replace(tzinfo=pytz.UTC)
            
            # Compare with one hour ago
            if file_time > one_hour_ago.replace(tzinfo=file_time.tzinfo):
                recent_files.append(obj)
                
                # Count file types
                if 'thermal_data' in obj['Key']:
                    thermal_files += 1
                elif 'gas_data' in obj['Key']:
                    gas_files += 1
            else:
                # Since files are sorted by date, we can break early
                break
        
        print(f"\n📊 Live Data Detection Results:")
        print(f"  Recent files (last hour): {len(recent_files)}")
        print(f"  Recent thermal files: {thermal_files}")
        print(f"  Recent gas files: {gas_files}")
        
        if len(recent_files) > 0:
            print("✅ LIVE DATA DETECTED!")
            print("   The dashboard should now show 'LIVE DATA DETECTED'")
            
            print(f"\n🔥 Most Recent Files:")
            for i, obj in enumerate(recent_files[:5]):
                key = obj['Key']
                last_modified = obj['LastModified']
                time_diff = utc_now - last_modified
                print(f"  {i+1}. {last_modified.strftime('%Y-%m-%d %H:%M:%S %Z')} | {key} | {time_diff} ago")
        else:
            print("⚠️ NO RECENT LIVE DATA")
            print("   This is unexpected given what you've observed")
            
        # Show the most recent file overall
        if sorted_files:
            most_recent = sorted_files[0]
            key = most_recent['Key']
            last_modified = most_recent['LastModified']
            time_diff = utc_now - last_modified
            print(f"\n📄 Most Recent File Overall:")
            print(f"  {last_modified.strftime('%Y-%m-%d %H:%M:%S %Z')} | {key} | {time_diff} ago")
            
    except Exception as e:
        print(f"❌ Error during final verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    final_verification()