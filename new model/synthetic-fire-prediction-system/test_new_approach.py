#!/usr/bin/env python3
"""
Test the new approach for finding recent files
"""

import boto3
from datetime import datetime, timedelta
import pytz

def test_new_approach():
    """Test the new approach for finding recent files."""
    print("🧪 Testing New Approach for Finding Recent Files")
    print("=" * 50)
    
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        utc_now = datetime.now(pytz.UTC)
        one_hour_ago = utc_now - timedelta(hours=1)
        today_str = utc_now.strftime('%Y%m%d')
        
        print(f"Current UTC time: {utc_now}")
        print(f"One hour ago: {one_hour_ago}")
        print(f"Today's date string: {today_str}")
        print()
        
        recent_files = []
        thermal_files = 0
        gas_files = 0
        
        # Search for gas data files from today
        print("Searching for gas data files from today...")
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
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
            paginator = s3_client.get_paginator('list_objects_v2')
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
        
        print(f"\n📊 Results:")
        print(f"  Recent files (last hour): {len(recent_files)}")
        print(f"  Recent thermal files: {thermal_files}")
        print(f"  Recent gas files: {gas_files}")
        
        if len(recent_files) > 0:
            print("✅ LIVE DATA DETECTED!")
            print("\n🔥 Most Recent Files:")
            for i, file_info in enumerate(recent_files[:10]):
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
    test_new_approach()