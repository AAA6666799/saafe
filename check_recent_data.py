#!/usr/bin/env python3
"""
Check for the most recent sensor data in the S3 bucket
"""

import boto3
import csv
from io import StringIO
from datetime import datetime, timedelta
import pytz

def check_recent_data():
    """Check for the most recent sensor data"""
    print("🔍 Checking for recent sensor data...")
    print("=" * 50)
    
    try:
        # Initialize S3 client
        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'data-collector-of-first-device'
        
        # Get current time
        utc_now = datetime.now(pytz.UTC)
        one_day_ago = utc_now - timedelta(days=1)
        one_week_ago = utc_now - timedelta(days=7)
        
        print(f"Current time (UTC): {utc_now}")
        print(f"One day ago: {one_day_ago}")
        print(f"One week ago: {one_week_ago}")
        
        # Check for recent thermal data
        print("\n🌡️  Checking for recent thermal data...")
        thermal_found = False
        
        # Check with today's date prefix
        today_str = utc_now.strftime('%Y%m%d')
        try:
            thermal_response = s3.list_objects_v2(
                Bucket=bucket_name, 
                Prefix=f'thermal-data/thermal_data_{today_str}'
            )
            
            if 'Contents' in thermal_response:
                objects = sorted(thermal_response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                for obj in objects[:3]:  # Check first 3 recent files
                    file_time = obj['LastModified']
                    if file_time.tzinfo is None:
                        file_time = file_time.replace(tzinfo=pytz.UTC)
                    
                    print(f"  Found thermal file: {obj['Key']} (Modified: {file_time})")
                    thermal_found = True
                    
                    # Analyze the file
                    file_response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                    content = file_response['Body'].read().decode('utf-8')
                    
                    csv_reader = csv.reader(StringIO(content))
                    rows = list(csv_reader)
                    
                    if len(rows) > 1:
                        headers = rows[0]
                        data_row = rows[1]
                        print(f"    Columns: {len(headers)}")
                        print(f"    Sample data: {dict(zip(headers[:5], data_row[:5]))}")
                    break
        except Exception as e:
            print(f"  Error checking today's thermal data: {e}")
        
        # If not found, check more broadly
        if not thermal_found:
            try:
                thermal_response = s3.list_objects_v2(
                    Bucket=bucket_name, 
                    Prefix='thermal-data/'
                )
                
                if 'Contents' in thermal_response:
                    objects = sorted(thermal_response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                    for obj in objects[:3]:  # Check first 3 recent files
                        file_time = obj['LastModified']
                        if file_time.tzinfo is None:
                            file_time = file_time.replace(tzinfo=pytz.UTC)
                        
                        if file_time > one_week_ago.replace(tzinfo=file_time.tzinfo):
                            print(f"  Recent thermal file: {obj['Key']} (Modified: {file_time})")
                            
                            # Analyze the file
                            file_response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                            content = file_response['Body'].read().decode('utf-8')
                            
                            csv_reader = csv.reader(StringIO(content))
                            rows = list(csv_reader)
                            
                            if len(rows) > 1:
                                headers = rows[0]
                                data_row = rows[1]
                                print(f"    Columns: {len(headers)}")
                                print(f"    Sample data: {dict(zip(headers[:5], data_row[:5]))}")
                        else:
                            print(f"  No recent thermal data found (most recent: {file_time})")
            except Exception as e:
                print(f"  Error checking thermal data: {e}")
        
        # Check for recent gas data
        print("\n💨 Checking for recent gas data...")
        gas_found = False
        
        # Check with today's date prefix
        try:
            gas_response = s3.list_objects_v2(
                Bucket=bucket_name, 
                Prefix=f'gas-data/gas_data_{today_str}'
            )
            
            if 'Contents' in gas_response:
                objects = sorted(gas_response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                for obj in objects[:3]:  # Check first 3 recent files
                    file_time = obj['LastModified']
                    if file_time.tzinfo is None:
                        file_time = file_time.replace(tzinfo=pytz.UTC)
                    
                    print(f"  Found gas file: {obj['Key']} (Modified: {file_time})")
                    gas_found = True
                    
                    # Analyze the file
                    file_response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                    content = file_response['Body'].read().decode('utf-8')
                    
                    csv_reader = csv.reader(StringIO(content))
                    rows = list(csv_reader)
                    
                    if len(rows) > 1:
                        headers = rows[0]
                        data_row = rows[1]
                        print(f"    Columns: {len(headers)}")
                        print(f"    Data: {dict(zip(headers, data_row))}")
                    break
        except Exception as e:
            print(f"  Error checking today's gas data: {e}")
        
        # If not found, check more broadly
        if not gas_found:
            try:
                gas_response = s3.list_objects_v2(
                    Bucket=bucket_name, 
                    Prefix='gas-data/'
                )
                
                if 'Contents' in gas_response:
                    objects = sorted(gas_response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                    for obj in objects[:3]:  # Check first 3 recent files
                        file_time = obj['LastModified']
                        if file_time.tzinfo is None:
                            file_time = file_time.replace(tzinfo=pytz.UTC)
                        
                        if file_time > one_week_ago.replace(tzinfo=file_time.tzinfo):
                            print(f"  Recent gas file: {obj['Key']} (Modified: {file_time})")
                            
                            # Analyze the file
                            file_response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                            content = file_response['Body'].read().decode('utf-8')
                            
                            csv_reader = csv.reader(StringIO(content))
                            rows = list(csv_reader)
                            
                            if len(rows) > 1:
                                headers = rows[0]
                                data_row = rows[1]
                                print(f"    Columns: {len(headers)}")
                                print(f"    Data: {dict(zip(headers, data_row))}")
                        else:
                            print(f"  No recent gas data found (most recent: {file_time})")
            except Exception as e:
                print(f"  Error checking gas data: {e}")
                
    except Exception as e:
        print(f"❌ Error checking recent data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_recent_data()