#!/usr/bin/env python3
"""
Debug S3 file listing issue
"""

import boto3
from datetime import datetime
import pytz

def debug_s3():
    """Debug S3 file listing."""
    print("🔍 S3 Debugging")
    print("=" * 20)
    
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        utc_now = datetime.now(pytz.UTC)
        today_str = utc_now.strftime('%Y%m%d')
        
        print(f"Current date: {today_str}")
        print(f"Current UTC time: {utc_now}")
        print()
        
        # Try different approaches to list files
        print("1. Listing first 100 files:")
        response = s3_client.list_objects_v2(
            Bucket='data-collector-of-first-device',
            MaxKeys=100
        )
        
        if 'Contents' in response:
            print(f"   Found {response['KeyCount']} files")
            # Check for today's files
            today_files = [obj for obj in response['Contents'] if today_str in obj['Key']]
            print(f"   Today's files in first 100: {len(today_files)}")
            
            if today_files:
                sorted_today = sorted(today_files, key=lambda x: x['LastModified'], reverse=True)
                most_recent = sorted_today[0]
                print(f"   Most recent today file: {most_recent['Key']}")
                print(f"   Last modified: {most_recent['LastModified']}")
        else:
            print("   No files found")
        
        print()
        print("2. Listing with prefix for gas data:")
        try:
            response = s3_client.list_objects_v2(
                Bucket='data-collector-of-first-device',
                Prefix=f'gas-data/gas_data_{today_str}',
                MaxKeys=50
            )
            
            if 'Contents' in response:
                print(f"   Found {response['KeyCount']} gas files from today")
                if response['Contents']:
                    sorted_files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                    most_recent = sorted_files[0]
                    time_diff = utc_now - most_recent['LastModified']
                    print(f"   Most recent: {most_recent['Key']}")
                    print(f"   Last modified: {most_recent['LastModified']}")
                    print(f"   Age: {time_diff}")
            else:
                print("   No gas files from today")
        except Exception as e:
            print(f"   Error: {e}")
            
        print()
        print("3. Listing with prefix for thermal data:")
        try:
            response = s3_client.list_objects_v2(
                Bucket='data-collector-of-first-device',
                Prefix=f'thermal-data/thermal_data_{today_str}',
                MaxKeys=50
            )
            
            if 'Contents' in response:
                print(f"   Found {response['KeyCount']} thermal files from today")
                if response['Contents']:
                    sorted_files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                    most_recent = sorted_files[0]
                    time_diff = utc_now - most_recent['LastModified']
                    print(f"   Most recent: {most_recent['Key']}")
                    print(f"   Last modified: {most_recent['LastModified']}")
                    print(f"   Age: {time_diff}")
            else:
                print("   No thermal files from today")
        except Exception as e:
            print(f"   Error: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_s3()