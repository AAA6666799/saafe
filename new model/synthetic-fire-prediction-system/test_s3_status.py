#!/usr/bin/env python3
"""
Test S3 status checking code
"""

import boto3
from datetime import datetime, timedelta
import pytz

def test_s3_status():
    """Test S3 status checking."""
    print("🔍 Testing S3 Status Checking")
    print("=" * 30)
    
    try:
        # Initialize S3 client
        s3 = boto3.client('s3', region_name='us-east-1')
        utc_now = datetime.now(pytz.UTC)
        one_hour_ago = utc_now - timedelta(hours=1)
        
        # Use a more efficient approach to find recent files
        # Instead of listing all files, we'll search for today's files by prefix
        today_str = utc_now.strftime('%Y%m%d')
        recent_files = []
        thermal_files = 0
        gas_files = 0
        
        print(f"Current UTC time: {utc_now}")
        print(f"One hour ago: {one_hour_ago}")
        print(f"Today's date string: {today_str}")
        print()
        
        # Search for gas data files from today
        print("Searching for gas data files from today...")
        try:
            paginator = s3.get_paginator('list_objects_v2')
            gas_pages = paginator.paginate(
                Bucket='data-collector-of-first-device',
                Prefix=f'gas-data/gas_data_{today_str}'
            )
            
            for page in gas_pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        file_time = obj['LastModified']
                        if file_time.tzinfo is None:
                            file_time = file_time.replace(tzinfo=pytz.UTC)
                        
                        # Check if file was modified in the last hour
                        if file_time > one_hour_ago.replace(tzinfo=file_time.tzinfo):
                            recent_files.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'modified': obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S UTC')
                            })
                            gas_files += 1
        except Exception as e:
            print(f"Error searching for gas files: {e}")
        
        # Search for thermal data files from today
        print("Searching for thermal data files from today...")
        try:
            paginator = s3.get_paginator('list_objects_v2')
            thermal_pages = paginator.paginate(
                Bucket='data-collector-of-first-device',
                Prefix=f'thermal-data/thermal_data_{today_str}'
            )
            
            for page in thermal_pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        file_time = obj['LastModified']
                        if file_time.tzinfo is None:
                            file_time = file_time.replace(tzinfo=pytz.UTC)
                        
                        # Check if file was modified in the last hour
                        if file_time > one_hour_ago.replace(tzinfo=file_time.tzinfo):
                            recent_files.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'modified': obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S UTC')
                            })
                            thermal_files += 1
        except Exception as e:
            print(f"Error searching for thermal files: {e}")
        
        # Sort recent files by last modified (most recent first)
        recent_files.sort(key=lambda x: x['modified'], reverse=True)
        
        # Also get total file count for display
        try:
            response = s3.list_objects_v2(
                Bucket='data-collector-of-first-device',
                MaxKeys=1
            )
            total_files = response.get('KeyCount', 0)
            if response.get('IsTruncated', False):
                # If truncated, there are more files
                total_files = '1000+'  # Approximate
        except:
            total_files = 'Unknown'
        
        print(f"\n📊 Results:")
        print(f"  Total files: {total_files}")
        print(f"  Recent files (last hour): {len(recent_files)}")
        print(f"  Recent thermal files: {thermal_files}")
        print(f"  Recent gas files: {gas_files}")
        
        if len(recent_files) > 0:
            print("✅ LIVE DATA DETECTED!")
            print("\n🔥 Most Recent Files:")
            for i, file_info in enumerate(recent_files[:5]):
                key = file_info['key']
                modified = file_info['modified']
                size = file_info['size']
                print(f"  {i+1}. {modified} | {key} ({size} bytes)")
        else:
            print("⚠️ NO RECENT LIVE DATA")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_s3_status()