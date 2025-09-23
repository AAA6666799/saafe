#!/usr/bin/env python3
"""
Verify if live data should be detected by the dashboard
"""

import boto3
from datetime import datetime, timedelta
import pytz

def verify_live_data():
    """Verify if live data should be detected."""
    print("🔍 Live Data Verification")
    print("=" * 30)
    
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Get current times
        utc_now = datetime.now(pytz.UTC)
        local_now = datetime.now()
        
        print(f"Current Times:")
        print(f"  UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Local: {local_now.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Calculate time thresholds
        one_hour_ago = utc_now - timedelta(hours=1)
        ninety_minutes_ago = utc_now - timedelta(minutes=90)
        
        print(f"Time Thresholds:")
        print(f"  One Hour Ago: {one_hour_ago.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  90 Minutes Ago: {ninety_minutes_ago.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print()
        
        # List objects in the bucket
        print("Checking S3 bucket: data-collector-of-first-device")
        response = s3_client.list_objects_v2(
            Bucket='data-collector-of-first-device',
            MaxKeys=50  # Check more files to be sure
        )
        
        if 'Contents' not in response:
            print("❌ No files found in the bucket")
            return
        
        print(f"✅ Found {response['KeyCount']} files in the bucket")
        
        # Sort files by last modified (most recent first)
        sorted_files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
        
        print(f"\n🔥 Most Recent Files:")
        print("-" * 30)
        
        recent_count_60 = 0
        recent_count_90 = 0
        
        # Check the most recent files
        for i, obj in enumerate(sorted_files[:20]):
            key = obj['Key']
            last_modified = obj['LastModified']
            size = obj['Size']
            
            print(f"{i+1:2d}. {last_modified.strftime('%Y-%m-%d %H:%M:%S %Z')} | {key} ({size} bytes)")
            
            # Check time differences
            time_diff = utc_now - last_modified
            
            # Check if within 60 minutes (old dashboard logic)
            if last_modified > one_hour_ago.replace(tzinfo=last_modified.tzinfo):
                recent_count_60 += 1
                print(f"    🟢 Within 60 min ({time_diff})")
            # Check if within 90 minutes (new dashboard logic)
            elif last_modified > ninety_minutes_ago.replace(tzinfo=last_modified.tzinfo):
                recent_count_90 += 1
                print(f"    🟡 Within 90 min ({time_diff})")
            else:
                print(f"    🔴 Older ({time_diff})")
        
        print(f"\n📊 Summary:")
        print(f"  Files within 60 minutes: {recent_count_60}")
        print(f"  Files within 90 minutes: {recent_count_90}")
        print(f"  Total recent files: {recent_count_60 + recent_count_90}")
        
        if recent_count_60 > 0:
            print("✅ LIVE DATA DETECTED (60-minute threshold)")
        elif recent_count_90 > 0:
            print("🟡 Live data detected (90-minute threshold, timezone difference likely)")
        else:
            print("⚠️ NO RECENT LIVE DATA")
            
        # Check specifically for files from today
        today = datetime.now().date()
        today_files = [obj for obj in sorted_files if obj['LastModified'].date() == today]
        print(f"\n📅 Files from today ({today}): {len(today_files)}")
        
        if today_files:
            print("✅ Today's files found - devices are active!")
            
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_live_data()