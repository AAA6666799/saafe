#!/usr/bin/env python3
"""
Test script to verify the dashboard data retrieval function works correctly
"""

import boto3
import csv
import json
from io import StringIO
from datetime import datetime, timedelta
import pytz

def get_recent_sensor_data_test():
    """Test version of the get_recent_sensor_data function"""
    print("🔍 Testing get_recent_sensor_data function...")
    print("=" * 50)
    
    try:
        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'data-collector-of-first-device'
        
        # Get current time in UTC
        utc_now = datetime.now(pytz.UTC)
        one_day_ago = utc_now - timedelta(days=1)  # Extended to 1 day for testing
        
        # Lists to store data
        thermal_data = []
        gas_data = []
        
        # Get recent thermal data files
        today_str = utc_now.strftime('%Y%m%d')
        try:
            paginator = s3.get_paginator('list_objects_v2')
            thermal_pages = paginator.paginate(
                Bucket=bucket_name,
                Prefix=f'thermal-data/thermal_data_{today_str}'
            )
            
            for page in thermal_pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        file_time = obj['LastModified']
                        if file_time.tzinfo is None:
                            file_time = file_time.replace(tzinfo=pytz.UTC)
                        
                        # Check if file was modified in the last day (for testing)
                        if file_time > one_day_ago.replace(tzinfo=file_time.tzinfo):
                            # Download and parse the CSV file
                            try:
                                response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                                content = response['Body'].read().decode('utf-8')
                                
                                # Parse CSV content
                                csv_reader = csv.reader(StringIO(content))
                                rows = list(csv_reader)
                                
                                if len(rows) > 1:
                                    # Get headers and first data row
                                    headers = rows[0]
                                    data_row = rows[1]
                                    
                                    # Create a dictionary with headers as keys
                                    data_dict = {}
                                    for i, header in enumerate(headers):
                                        if i < len(data_row):
                                            data_dict[header] = data_row[i]
                                    
                                    data_dict['timestamp'] = file_time
                                    thermal_data.append(data_dict)
                                    print(f"Added thermal data: {obj['Key']}")
                            except Exception as e:
                                print(f"Error parsing thermal file {obj['Key']}: {e}")
                                continue
        except Exception as e:
            print(f"Error listing thermal data: {e}")
            pass
        
        # Get recent gas data files
        try:
            paginator = s3.get_paginator('list_objects_v2')
            gas_pages = paginator.paginate(
                Bucket=bucket_name,
                Prefix=f'gas-data/gas_data_{today_str}'
            )
            
            for page in gas_pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        file_time = obj['LastModified']
                        if file_time.tzinfo is None:
                            file_time = file_time.replace(tzinfo=pytz.UTC)
                        
                        # Check if file was modified in the last day (for testing)
                        if file_time > one_day_ago.replace(tzinfo=file_time.tzinfo):
                            # Download and parse the CSV file
                            try:
                                response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                                content = response['Body'].read().decode('utf-8')
                                
                                # Parse CSV content
                                csv_reader = csv.reader(StringIO(content))
                                rows = list(csv_reader)
                                
                                if len(rows) > 1:
                                    # Get headers and first data row
                                    headers = rows[0]
                                    data_row = rows[1]
                                    
                                    # Create a dictionary with headers as keys
                                    data_dict = {}
                                    for i, header in enumerate(headers):
                                        if i < len(data_row):
                                            data_dict[header] = data_row[i]
                                    
                                    data_dict['timestamp'] = file_time
                                    gas_data.append(data_dict)
                                    print(f"Added gas data: {obj['Key']}")
                            except Exception as e:
                                print(f"Error parsing gas file {obj['Key']}: {e}")
                                continue
        except Exception as e:
            print(f"Error listing gas data: {e}")
            pass
        
        # If no data found for today, try to get some recent data regardless of date
        if not thermal_data or not gas_data:
            print("No data found for today, checking recent files...")
            try:
                response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=20)
                if 'Contents' in response:
                    objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                    for obj in objects[:20]:
                        file_time = obj['LastModified']
                        if file_time.tzinfo is None:
                            file_time = file_time.replace(tzinfo=pytz.UTC)
                        
                        key = obj['Key']
                        # Download and parse the file
                        try:
                            response = s3.get_object(Bucket=bucket_name, Key=key)
                            content = response['Body'].read().decode('utf-8')
                            
                            # Parse CSV content
                            csv_reader = csv.reader(StringIO(content))
                            rows = list(csv_reader)
                            
                            if len(rows) > 1:
                                # Get headers and first data row
                                headers = rows[0]
                                data_row = rows[1]
                                
                                # Create a dictionary with headers as keys
                                data_dict = {}
                                for i, header in enumerate(headers):
                                    if i < len(data_row):
                                        data_dict[header] = data_row[i]
                                
                                data_dict['timestamp'] = file_time
                                
                                # Add to appropriate list based on key
                                if 'thermal' in key:
                                    thermal_data.append(data_dict)
                                    print(f"Added older thermal data: {key}")
                                elif 'gas' in key:
                                    gas_data.append(data_dict)
                                    print(f"Added older gas data: {key}")
                        except Exception as e:
                            print(f"Error parsing file {key}: {e}")
                            continue
            except Exception as e:
                print(f"Error listing all files: {e}")
                pass
        
        # Sort by timestamp and get most recent
        thermal_data.sort(key=lambda x: x['timestamp'], reverse=True)
        gas_data.sort(key=lambda x: x['timestamp'], reverse=True)
        
        result = {
            'thermal': thermal_data[:10],  # Last 10 thermal readings
            'gas': gas_data[:10],          # Last 10 gas readings
            'latest_thermal': thermal_data[0] if thermal_data else None,
            'latest_gas': gas_data[0] if gas_data else None
        }
        
        print(f"\n✅ Function completed successfully")
        print(f"Thermal data entries: {len(result['thermal'])}")
        print(f"Gas data entries: {len(result['gas'])}")
        
        if result['latest_thermal']:
            print(f"Latest thermal timestamp: {result['latest_thermal']['timestamp']}")
            print(f"Latest thermal data keys: {list(result['latest_thermal'].keys())}")
            
        if result['latest_gas']:
            print(f"Latest gas timestamp: {result['latest_gas']['timestamp']}")
            print(f"Latest gas data keys: {list(result['latest_gas'].keys())}")
        
        return result
    except Exception as e:
        print(f"❌ Error in get_recent_sensor_data: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = get_recent_sensor_data_test()
    if result:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed!")