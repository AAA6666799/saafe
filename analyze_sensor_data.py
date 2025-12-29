#!/usr/bin/env python3
"""
Analyze what sensor data is available in the S3 bucket
"""

import boto3
import csv
from io import StringIO
from datetime import datetime
import pytz

def analyze_sensor_data():
    """Analyze the sensor data structure in S3"""
    print("🔍 Analyzing sensor data in S3 bucket...")
    print("=" * 50)
    
    try:
        # Initialize S3 client
        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'data-collector-of-first-device'
        
        # Get recent files
        response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=10)
        
        if 'Contents' in response:
            # Sort by last modified (most recent first)
            objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
            
            print(f"Found {len(objects)} recent files:")
            print("-" * 50)
            
            # Analyze both thermal and gas data
            thermal_files = []
            gas_files = []
            
            for obj in objects[:10]:  # Check first 10 files
                key = obj['Key']
                if 'thermal' in key:
                    thermal_files.append(obj)
                elif 'gas' in key:
                    gas_files.append(obj)
            
            print(f"Thermal data files: {len(thermal_files)}")
            print(f"Gas data files: {len(gas_files)}")
            
            # Analyze thermal data structure
            if thermal_files:
                print("\n🌡️  Thermal Data Structure:")
                latest_thermal = thermal_files[0]
                print(f"Latest thermal file: {latest_thermal['Key']}")
                
                # Download and parse
                file_response = s3.get_object(Bucket=bucket_name, Key=latest_thermal['Key'])
                content = file_response['Body'].read().decode('utf-8')
                
                csv_reader = csv.reader(StringIO(content))
                rows = list(csv_reader)
                
                if len(rows) > 0:
                    headers = rows[0]
                    print(f"Columns: {len(headers)}")
                    print("Header sample (first 10):", headers[:10])
                    print("Header sample (last 10):", headers[-10:] if len(headers) > 10 else headers)
                    
                    # Show data sample
                    if len(rows) > 1:
                        data_row = rows[1]
                        print("Data sample (first 10):", data_row[:10])
                        
                        # Calculate some statistics
                        pixel_values = []
                        for i, header in enumerate(headers):
                            if header.startswith('pixel_') and i < len(data_row):
                                try:
                                    pixel_values.append(float(data_row[i]))
                                except ValueError:
                                    pass
                        
                        if pixel_values:
                            avg_temp = sum(pixel_values) / len(pixel_values)
                            min_temp = min(pixel_values)
                            max_temp = max(pixel_values)
                            print(f"Temperature stats - Avg: {avg_temp:.2f}°C, Min: {min_temp:.2f}°C, Max: {max_temp:.2f}°C")
            
            # Analyze gas data structure
            if gas_files:
                print("\n💨 Gas Data Structure:")
                latest_gas = gas_files[0]
                print(f"Latest gas file: {latest_gas['Key']}")
                
                # Download and parse
                file_response = s3.get_object(Bucket=bucket_name, Key=latest_gas['Key'])
                content = file_response['Body'].read().decode('utf-8')
                
                csv_reader = csv.reader(StringIO(content))
                rows = list(csv_reader)
                
                if len(rows) > 0:
                    headers = rows[0]
                    print(f"Columns: {len(headers)}")
                    print("Headers:", headers)
                    
                    # Show data sample
                    if len(rows) > 1:
                        data_row = rows[1]
                        print("Data row:", data_row)
                        
                        # Map values to headers
                        for i, (header, value) in enumerate(zip(headers, data_row)):
                            print(f"  {header}: {value}")
            
            # Check for other types of data
            print("\n📋 Other Data Types:")
            other_files = [obj for obj in objects[:10] if 'thermal' not in obj['Key'] and 'gas' not in obj['Key']]
            for obj in other_files:
                print(f"  {obj['Key']}")
                
        else:
            print("❌ No objects found in bucket")
            
    except Exception as e:
        print(f"❌ Error analyzing sensor data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_sensor_data()