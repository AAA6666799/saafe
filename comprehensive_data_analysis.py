#!/usr/bin/env python3
"""
Comprehensive analysis of all sensor data in the S3 bucket
"""

import boto3
import csv
from io import StringIO
from datetime import datetime
import pytz

def comprehensive_data_analysis():
    """Comprehensive analysis of sensor data"""
    print("🔍 Comprehensive sensor data analysis...")
    print("=" * 50)
    
    try:
        # Initialize S3 client
        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'data-collector-of-first-device'
        
        # Count different types of files
        thermal_count = 0
        gas_count = 0
        other_count = 0
        
        # Check for thermal data with specific prefix
        try:
            thermal_response = s3.list_objects_v2(
                Bucket=bucket_name, 
                Prefix='thermal-data/',
                MaxKeys=5
            )
            
            if 'Contents' in thermal_response:
                thermal_count = len(thermal_response['Contents'])
                print(f"Found {thermal_count} thermal data files")
                
                if thermal_response['Contents']:
                    latest_thermal = thermal_response['Contents'][0]
                    print(f"Latest thermal file: {latest_thermal['Key']}")
                    print(f"Last modified: {latest_thermal['LastModified']}")
                    
                    # Analyze structure
                    file_response = s3.get_object(Bucket=bucket_name, Key=latest_thermal['Key'])
                    content = file_response['Body'].read().decode('utf-8')
                    
                    csv_reader = csv.reader(StringIO(content))
                    rows = list(csv_reader)
                    
                    if len(rows) > 0:
                        headers = rows[0]
                        print(f"Thermal data columns: {len(headers)}")
                        if len(headers) > 5:
                            print("Header sample:", headers[:5], "...", headers[-5:] if len(headers) > 10 else "")
                        else:
                            print("Headers:", headers)
            else:
                print("No thermal data files found with 'thermal-data/' prefix")
        except Exception as e:
            print(f"Error checking thermal data: {e}")
        
        # Check for gas data with specific prefix
        try:
            gas_response = s3.list_objects_v2(
                Bucket=bucket_name, 
                Prefix='gas-data/',
                MaxKeys=5
            )
            
            if 'Contents' in gas_response:
                gas_count = len(gas_response['Contents'])
                print(f"Found {gas_count} gas data files")
                
                if gas_response['Contents']:
                    latest_gas = gas_response['Contents'][0]
                    print(f"Latest gas file: {latest_gas['Key']}")
                    print(f"Last modified: {latest_gas['LastModified']}")
                    
                    # Analyze structure
                    file_response = s3.get_object(Bucket=bucket_name, Key=latest_gas['Key'])
                    content = file_response['Body'].read().decode('utf-8')
                    
                    csv_reader = csv.reader(StringIO(content))
                    rows = list(csv_reader)
                    
                    if len(rows) > 0:
                        headers = rows[0]
                        print(f"Gas data columns: {len(headers)}")
                        print("Headers:", headers)
                        
                        if len(rows) > 1:
                            data_row = rows[1]
                            print("Data sample:", dict(zip(headers, data_row)))
            else:
                print("No gas data files found with 'gas-data/' prefix")
        except Exception as e:
            print(f"Error checking gas data: {e}")
        
        # Check overall bucket contents
        try:
            all_response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=50)
            
            if 'Contents' in all_response:
                objects = sorted(all_response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                print(f"\nTotal files in bucket: {len(objects)}")
                print("Most recent files:")
                
                file_types = {}
                for obj in objects[:20]:  # Check first 20 files
                    key = obj['Key']
                    print(f"  {obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')} - {key}")
                    
                    # Categorize file types
                    if 'thermal' in key.lower():
                        file_types['thermal'] = file_types.get('thermal', 0) + 1
                    elif 'gas' in key.lower():
                        file_types['gas'] = file_types.get('gas', 0) + 1
                    else:
                        file_types['other'] = file_types.get('other', 0) + 1
                
                print("\nFile type distribution:")
                for file_type, count in file_types.items():
                    print(f"  {file_type}: {count}")
            else:
                print("No files found in bucket")
        except Exception as e:
            print(f"Error checking overall bucket contents: {e}")
            
    except Exception as e:
        print(f"❌ Error in comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    comprehensive_data_analysis()