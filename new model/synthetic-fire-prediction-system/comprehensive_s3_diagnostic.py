#!/usr/bin/env python3
"""
Comprehensive S3 diagnostic tool to check for live data issues
"""

import boto3
from datetime import datetime, timedelta
import pytz

def comprehensive_s3_diagnostic():
    """Perform comprehensive analysis of S3 bucket contents."""
    print("🔍 Comprehensive S3 Bucket Diagnostic")
    print("=" * 50)
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name='us-east-1')
    
    try:
        # Get current times in different timezones
        utc_now = datetime.now(pytz.UTC)
        local_now = datetime.now()
        
        print("🕐 Current Times:")
        print(f"  UTC Time: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Local Time: {local_now.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # List objects in the bucket
        print("📂 Analyzing S3 Bucket: data-collector-of-first-device")
        response = s3_client.list_objects_v2(
            Bucket='data-collector-of-first-device'
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
        print(f"  Last Modified: {last_modified}")
        print(f"  Last Modified (UTC): {last_modified.astimezone(pytz.UTC)}")
        print(f"  Size: {most_recent['Size']} bytes")
        
        # Calculate time differences
        print(f"\n⏱️  Time Analysis:")
        time_diff_utc = utc_now - last_modified
        print(f"  Age (UTC calculation): {time_diff_utc}")
        
        # Check if within last hour
        within_last_hour = time_diff_utc < timedelta(hours=1)
        print(f"  Within last hour: {within_last_hour}")
        
        # Check timezone information
        print(f"\n🌍 Timezone Information:")
        print(f"  Last Modified TZ Info: {last_modified.tzinfo}")
        print(f"  UTC Now TZ Info: {utc_now.tzinfo}")
        
        # Check if there's a timezone mismatch
        if last_modified.tzinfo is None and utc_now.tzinfo is not None:
            print("  ⚠️  Potential timezone mismatch detected!")
            # Try assuming the timestamp is UTC
            assumed_utc = last_modified.replace(tzinfo=pytz.UTC)
            time_diff_assumed = utc_now - assumed_utc
            print(f"  Age (assuming timestamp is UTC): {time_diff_assumed}")
            within_last_hour_assumed = time_diff_assumed < timedelta(hours=1)
            print(f"  Within last hour (assumed UTC): {within_last_hour_assumed}")
        
        # Show recent files (last 24 hours)
        print(f"\n📅 Files from Last 24 Hours:")
        one_day_ago = utc_now - timedelta(days=1)
        recent_files = [f for f in sorted_files if f['LastModified'] > one_day_ago]
        print(f"  Found {len(recent_files)} files in the last 24 hours")
        
        for i, file_obj in enumerate(recent_files[:10]):  # Show top 10
            print(f"    {i+1:2d}. {file_obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S %Z')} | {file_obj['Key']}")
        
        # Check file naming patterns
        print(f"\n📝 File Naming Analysis:")
        thermal_count = sum(1 for f in response['Contents'] if 'thermal_data' in f['Key'])
        gas_count = sum(1 for f in response['Contents'] if 'gas_data' in f['Key'])
        print(f"  Thermal data files: {thermal_count}")
        print(f"  Gas data files: {gas_count}")
        
        # Check for today's files
        today = datetime.now().date()
        today_files = [f for f in response['Contents'] if f['LastModified'].date() == today]
        print(f"  Files from today: {len(today_files)}")
        
        if today_files:
            print("  Today's files:")
            sorted_today = sorted(today_files, key=lambda x: x['LastModified'], reverse=True)
            for i, file_obj in enumerate(sorted_today[:5]):  # Show top 5
                print(f"    {i+1}. {file_obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S %Z')} | {file_obj['Key']}")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"  Most recent file age: {time_diff_utc}")
        print(f"  Within last hour: {within_last_hour}")
        if within_last_hour:
            print("  ✅ LIVE DATA DETECTED")
        else:
            print("  ⚠️  NO RECENT LIVE DATA")
            print("     This indicates that devices are not currently sending data")
        
    except Exception as e:
        print(f"❌ Error during diagnostic: {e}")
        import traceback
        traceback.print_exc()

def check_s3_permissions():
    """Check S3 permissions and access."""
    print("\n🔐 S3 Permissions Check:")
    print("-" * 25)
    
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Test basic access
        response = s3_client.head_bucket(Bucket='data-collector-of-first-device')
        print("✅ Basic S3 access: OK")
        
        # Test listing objects
        response = s3_client.list_objects_v2(Bucket='data-collector-of-first-device', MaxKeys=1)
        print("✅ List objects permission: OK")
        
        # Test getting object metadata (if files exist)
        if 'Contents' in response and response['Contents']:
            first_key = response['Contents'][0]['Key']
            response = s3_client.head_object(Bucket='data-collector-of-first-device', Key=first_key)
            print("✅ Read object metadata permission: OK")
        
        print("✅ All S3 permissions checks passed")
        
    except Exception as e:
        print(f"❌ S3 permissions error: {e}")

def main():
    """Main diagnostic function."""
    print("🔥 Fire Detection System - Comprehensive S3 Diagnostic")
    print("=" * 55)
    
    # Run comprehensive S3 analysis
    comprehensive_s3_diagnostic()
    
    # Check S3 permissions
    check_s3_permissions()
    
    print(f"\n" + "=" * 55)
    print("🔧 Diagnostic Complete")
    print("=" * 55)

if __name__ == "__main__":
    main()