#!/usr/bin/env python3
"""
Simple script to check live data status
"""

import boto3
from datetime import datetime, timedelta
import pytz

def check_live_status():
    """Check if there's live data in the S3 bucket."""
    print("🔥 Fire Detection System - Live Data Status Check")
    print("=" * 50)
    
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Get current time
        utc_now = datetime.now(pytz.UTC)
        print(f"Current UTC Time: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # One hour ago
        one_hour_ago = utc_now - timedelta(hours=1)
        print(f"One Hour Ago: {one_hour_ago.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print()
        
        # List objects in the bucket
        print("Checking S3 bucket: data-collector-of-first-device")
        response = s3_client.list_objects_v2(
            Bucket='data-collector-of-first-device',
            MaxKeys=100
        )
        
        if 'Contents' not in response:
            print("❌ No files found in the bucket")
            return
        
        print(f"✅ Found {response['KeyCount']} files in the bucket")
        
        # Sort files by last modified
        sorted_files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
        
        # Get the most recent file
        most_recent = sorted_files[0]
        last_modified = most_recent['LastModified']
        
        print(f"\n📄 Most Recent File:")
        print(f"  Key: {most_recent['Key']}")
        print(f"  Last Modified: {last_modified.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Check if within last hour
        time_diff = utc_now - last_modified
        within_last_hour = time_diff < timedelta(hours=1)
        
        print(f"  Age: {time_diff}")
        print(f"  Within Last Hour: {within_last_hour}")
        
        if within_last_hour:
            print("\n✅ LIVE DATA DETECTED")
            print("   The system is receiving live data from deployed devices.")
        else:
            print("\n⚠️  NO RECENT LIVE DATA")
            print("   Devices are not currently sending data to the S3 bucket.")
            print(f"   The most recent file is {time_diff} old.")
        
        # Show recent files (last 24 hours)
        one_day_ago = utc_now - timedelta(days=1)
        recent_files = [f for f in sorted_files if f['LastModified'] > one_day_ago]
        print(f"\n📅 Files from Last 24 Hours: {len(recent_files)}")
        
        # Show files from today
        today = datetime.now().date()
        today_files = [f for f in response['Contents'] if f['LastModified'].date() == today]
        print(f"📅 Files from Today: {len(today_files)}")
        
    except Exception as e:
        print(f"❌ Error checking live status: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_live_status()