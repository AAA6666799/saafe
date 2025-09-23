#!/usr/bin/env python3
"""
Script to verify device connectivity and live data transmission
"""

import boto3
from datetime import datetime, timedelta
import json

def verify_device_connectivity():
    """Verify that devices are sending live data to S3."""
    print("🔍 Verifying Device Connectivity and Live Data Transmission")
    print("=" * 60)
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name='us-east-1')
    
    try:
        # List objects in the bucket
        response = s3_client.list_objects_v2(
            Bucket='data-collector-of-first-device',
            MaxKeys=100
        )
        
        if 'Contents' not in response:
            print("❌ No files found in the bucket")
            print("   This indicates devices are not sending data or there's a configuration issue")
            return False
        
        # Get current time and calculate time windows
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)
        
        # Filter for recent files
        recent_files = []
        old_files = []
        
        for obj in response['Contents']:
            last_modified = obj['LastModified']
            
            # Categorize by age
            if last_modified > one_hour_ago.replace(tzinfo=last_modified.tzinfo):
                recent_files.append(obj)
            else:
                old_files.append(obj)
        
        print(f"📊 Bucket Analysis:")
        print(f"   Total files: {response['KeyCount']}")
        print(f"   Recent files (last hour): {len(recent_files)}")
        print(f"   Old files: {len(old_files)}")
        
        if recent_files:
            print("\n✅ LIVE DATA DETECTED")
            print("   Devices are successfully sending data to the cloud")
            
            # Show details of recent files
            print(f"\n📋 Recent Files (Last Hour):")
            sorted_recent = sorted(recent_files, key=lambda x: x['LastModified'], reverse=True)
            for i, obj in enumerate(sorted_recent[:10]):  # Show top 10
                print(f"   {i+1}. {obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')} | {obj['Key']}")
            
            # Analyze file types
            thermal_count = sum(1 for obj in sorted_recent if 'thermal_data' in obj['Key'])
            gas_count = sum(1 for obj in sorted_recent if 'gas_data' in obj['Key'])
            
            print(f"\n📈 Data Composition:")
            print(f"   Thermal data files: {thermal_count}")
            print(f"   Gas data files: {gas_count}")
            
            # Calculate data frequency
            if len(sorted_recent) > 1:
                # Calculate average time between files
                timestamps = [obj['LastModified'] for obj in sorted_recent]
                if len(timestamps) > 1:
                    time_diffs = [(timestamps[i] - timestamps[i+1]).total_seconds() 
                                 for i in range(len(timestamps)-1)]
                    avg_interval = sum(time_diffs) / len(time_diffs)
                    print(f"   Average interval: {avg_interval:.1f} seconds")
                    
                    if avg_interval < 60:
                        print("   🚀 High-frequency data collection confirmed (< 1 minute intervals)")
                    elif avg_interval < 300:
                        print("   ⚡ Medium-frequency data collection (~1-5 minute intervals)")
                    else:
                        print("   🐢 Low-frequency data collection (> 5 minute intervals)")
            
            return True
        else:
            print("\n⚠️  NO LIVE DATA DETECTED")
            print("   No files have been uploaded in the last hour")
            
            # Show the most recent old file
            if old_files:
                sorted_old = sorted(old_files, key=lambda x: x['LastModified'], reverse=True)
                most_recent_old = sorted_old[0]
                print(f"\n📅 Most Recent File (May be old):")
                print(f"   {most_recent_old['LastModified'].strftime('%Y-%m-%d %H:%M:%S')} | {most_recent_old['Key']}")
                print(f"   Age: {(now - most_recent_old['LastModified'].replace(tzinfo=None)).days} days old")
            
            print("\n🔧 Troubleshooting Steps:")
            print("   1. Verify devices are powered on and connected to the internet")
            print("   2. Check device S3 credentials and configuration")
            print("   3. Review device logs for upload errors")
            print("   4. Confirm the device is uploading to the correct S3 bucket:")
            print("      Bucket name: data-collector-of-first-device")
            print("   5. Check device sensor functionality")
            print("   6. Verify device has sufficient storage space")
            print("   7. Ensure device clock is synchronized")
            
            return False
        
    except Exception as e:
        print(f"❌ Error checking S3 contents: {e}")
        return False

def check_lambda_processing():
    """Check if Lambda function is processing data."""
    print("\n🔍 Checking Lambda Processing")
    print("=" * 30)
    
    try:
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        logs_client = boto3.client('logs', region_name='us-east-1')
        
        # Check if function exists
        response = lambda_client.get_function(FunctionName='saafe-s3-data-processor')
        print("✅ Lambda function exists")
        print(f"   Function name: {response['Configuration']['FunctionName']}")
        print(f"   Runtime: {response['Configuration']['Runtime']}")
        
        # Check CloudWatch logs for recent activity
        try:
            log_group = '/aws/lambda/saafe-s3-data-processor'
            logs_response = logs_client.describe_log_streams(
                logGroupName=log_group,
                orderBy='LastEventTime',
                descending=True,
                limit=5
            )
            
            if 'logStreams' in logs_response and logs_response['logStreams']:
                print("✅ Log streams found")
                latest_stream = logs_response['logStreams'][0]
                print(f"   Most recent log stream: {latest_stream['logStreamName']}")
                print(f"   Last event time: {datetime.fromtimestamp(latest_stream['lastEventTimestamp']/1000).strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print("⚠️  No log streams found - function may not have been triggered")
                
        except Exception as log_error:
            print(f"⚠️  Could not check logs: {log_error}")
            
    except Exception as e:
        print(f"❌ Error checking Lambda function: {e}")

def main():
    """Main verification function."""
    print("🔥 Fire Detection System - Device Connectivity Verification")
    print("=" * 60)
    
    # Verify device connectivity
    live_data_detected = verify_device_connectivity()
    
    # Check Lambda processing
    check_lambda_processing()
    
    print("\n" + "=" * 60)
    if live_data_detected:
        print("🎉 VERIFICATION COMPLETE: Devices are sending live data")
        print("   The system is ready for real-time fire detection")
    else:
        print("⚠️  VERIFICATION COMPLETE: No live data detected")
        print("   Please check device connectivity and configuration")
    
    print("=" * 60)

if __name__ == "__main__":
    main()