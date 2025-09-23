#!/usr/bin/env python3
"""
Detailed S3 bucket analysis to verify live data
"""

import boto3
from datetime import datetime, timedelta
import json
import pandas as pd

def detailed_s3_analysis():
    """Perform detailed analysis of S3 bucket contents."""
    print("🔍 Detailed S3 Bucket Analysis")
    print("=" * 50)
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name='us-east-1')
    
    try:
        # List all objects in the bucket
        print("Listing all objects in bucket 'data-collector-of-first-device'...")
        response = s3_client.list_objects_v2(
            Bucket='data-collector-of-first-device'
        )
        
        if 'Contents' not in response:
            print("❌ No files found in the bucket")
            return
        
        print(f"✅ Found {response['KeyCount']} files in the bucket")
        
        # Get current time
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        six_hours_ago = now - timedelta(hours=6)
        twelve_hours_ago = now - timedelta(hours=12)
        one_day_ago = now - timedelta(days=1)
        
        # Categorize files by time periods
        files_last_hour = []
        files_last_6_hours = []
        files_last_12_hours = []
        files_last_24_hours = []
        older_files = []
        
        thermal_files = 0
        gas_files = 0
        
        print("\nAnalyzing file timestamps...")
        
        # Process all files
        for obj in response['Contents']:
            key = obj['Key']
            last_modified = obj['LastModified']
            
            # Count file types
            if 'thermal_data' in key:
                thermal_files += 1
            elif 'gas_data' in key:
                gas_files += 1
            
            # Categorize by time periods
            if last_modified > one_hour_ago.replace(tzinfo=last_modified.tzinfo):
                files_last_hour.append(obj)
            elif last_modified > six_hours_ago.replace(tzinfo=last_modified.tzinfo):
                files_last_6_hours.append(obj)
            elif last_modified > twelve_hours_ago.replace(tzinfo=last_modified.tzinfo):
                files_last_12_hours.append(obj)
            elif last_modified > one_day_ago.replace(tzinfo=last_modified.tzinfo):
                files_last_24_hours.append(obj)
            else:
                older_files.append(obj)
        
        # Display results
        print(f"\n📊 File Analysis Summary:")
        print(f"   Total files: {response['KeyCount']}")
        print(f"   Thermal data files: {thermal_files}")
        print(f"   Gas data files: {gas_files}")
        print(f"   Files in last hour: {len(files_last_hour)}")
        print(f"   Files in last 6 hours: {len(files_last_6_hours)}")
        print(f"   Files in last 12 hours: {len(files_last_12_hours)}")
        print(f"   Files in last 24 hours: {len(files_last_24_hours)}")
        print(f"   Older files: {len(older_files)}")
        
        # Show recent files in detail
        if files_last_hour:
            print(f"\n🔥 FILES FROM LAST HOUR (LIVE DATA):")
            print("-" * 40)
            sorted_recent = sorted(files_last_hour, key=lambda x: x['LastModified'], reverse=True)
            for i, obj in enumerate(sorted_recent[:20]):  # Show top 20
                print(f"   {i+1:2d}. {obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')} | {obj['Key']}")
        else:
            print(f"\n⚠️  NO FILES FROM LAST HOUR")
            
            # Check for files in last 6 hours
            if files_last_6_hours:
                print(f"\n📅 Most recent files (last 6 hours):")
                print("-" * 40)
                sorted_recent = sorted(files_last_6_hours, key=lambda x: x['LastModified'], reverse=True)
                for i, obj in enumerate(sorted_recent[:10]):  # Show top 10
                    print(f"   {i+1:2d}. {obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')} | {obj['Key']}")
            else:
                print(f"\n📅 No recent files found in the last 6 hours")
        
        # Show file naming patterns
        print(f"\n📋 File Naming Analysis:")
        print("-" * 25)
        
        # Sample files to analyze naming
        sample_files = response['Contents'][:20]  # First 20 files
        thermal_patterns = []
        gas_patterns = []
        
        for obj in sample_files:
            key = obj['Key']
            if 'thermal_data' in key:
                thermal_patterns.append(key)
            elif 'gas_data' in key:
                gas_patterns.append(key)
        
        print(f"   Thermal data pattern examples:")
        for pattern in thermal_patterns[:3]:
            print(f"     - {pattern}")
            
        print(f"   Gas data pattern examples:")
        for pattern in gas_patterns[:3]:
            print(f"     - {pattern}")
        
        # Analyze data frequency if we have recent files
        if files_last_hour or files_last_6_hours:
            all_recent = files_last_hour + files_last_6_hours
            sorted_all = sorted(all_recent, key=lambda x: x['LastModified'], reverse=True)
            
            if len(sorted_all) > 1:
                print(f"\n⏱️  Data Frequency Analysis:")
                print("-" * 25)
                
                # Calculate time intervals between files
                timestamps = [obj['LastModified'] for obj in sorted_all[:20]]  # Last 20 files
                if len(timestamps) > 1:
                    time_diffs = []
                    for i in range(len(timestamps) - 1):
                        diff = (timestamps[i] - timestamps[i+1]).total_seconds()
                        time_diffs.append(diff)
                    
                    if time_diffs:
                        avg_interval = sum(time_diffs) / len(time_diffs)
                        min_interval = min(time_diffs)
                        max_interval = max(time_diffs)
                        
                        print(f"   Average interval: {avg_interval:.1f} seconds")
                        print(f"   Min interval: {min_interval:.1f} seconds")
                        print(f"   Max interval: {max_interval:.1f} seconds")
                        
                        if avg_interval < 60:
                            print("   🚀 High-frequency data collection (< 1 minute intervals)")
                        elif avg_interval < 300:
                            print("   ⚡ Medium-frequency data collection (1-5 minute intervals)")
                        else:
                            print("   🐢 Low-frequency data collection (> 5 minute intervals)")
        
        # Check for any potential issues
        print(f"\n🔍 Potential Issues Check:")
        print("-" * 25)
        
        # Check for files with unusual naming
        unusual_files = []
        for obj in response['Contents']:
            key = obj['Key']
            # Check if file name follows expected pattern
            if 'thermal_data' not in key and 'gas_data' not in key:
                unusual_files.append(key)
        
        if unusual_files:
            print("   ⚠️  Files with unusual naming patterns:")
            for file in unusual_files[:5]:  # Show first 5
                print(f"     - {file}")
        else:
            print("   ✅ All files follow expected naming patterns")
        
        # Check for very large files that might indicate issues
        large_files = []
        for obj in response['Contents']:
            if obj['Size'] > 1000000:  # 1MB
                large_files.append((obj['Key'], obj['Size']))
        
        if large_files:
            print("   ⚠️  Large files (>1MB):")
            for key, size in large_files[:5]:  # Show first 5
                print(f"     - {key} ({size/1024/1024:.1f} MB)")
        
        return {
            'total_files': response['KeyCount'],
            'recent_files': len(files_last_hour),
            'thermal_files': thermal_files,
            'gas_files': gas_files,
            'files_last_hour': files_last_hour,
            'files_last_6_hours': files_last_6_hours
        }
        
    except Exception as e:
        print(f"❌ Error analyzing S3 contents: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_lambda_logs():
    """Check Lambda function logs for recent activity."""
    print(f"\n🔍 Lambda Function Log Analysis:")
    print("-" * 35)
    
    try:
        logs_client = boto3.client('logs', region_name='us-east-1')
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        
        # Check if Lambda function exists
        try:
            response = lambda_client.get_function(FunctionName='saafe-s3-data-processor')
            print("✅ Lambda function 'saafe-s3-data-processor' exists")
        except Exception as e:
            print(f"❌ Lambda function not found: {e}")
            return
        
        # Check CloudWatch logs
        log_group = '/aws/lambda/saafe-s3-data-processor'
        try:
            response = logs_client.describe_log_streams(
                logGroupName=log_group,
                orderBy='LastEventTime',
                descending=True,
                limit=5
            )
            
            if 'logStreams' in response and response['logStreams']:
                print("✅ Log streams found")
                for stream in response['logStreams'][:3]:  # Show top 3
                    timestamp = datetime.fromtimestamp(stream['lastEventTimestamp']/1000)
                    print(f"   - {stream['logStreamName']}")
                    print(f"     Last event: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print("⚠️  No log streams found")
                
        except Exception as e:
            print(f"⚠️  Could not access logs: {e}")
            
    except Exception as e:
        print(f"❌ Error checking Lambda logs: {e}")

def main():
    """Main analysis function."""
    print("🔥 Fire Detection System - Detailed S3 Analysis")
    print("=" * 50)
    
    # Perform detailed S3 analysis
    result = detailed_s3_analysis()
    
    # Check Lambda logs
    check_lambda_logs()
    
    print(f"\n" + "=" * 50)
    if result:
        if result['recent_files'] > 0:
            print("✅ ANALYSIS COMPLETE: Live data detected in S3 bucket")
            print(f"   {result['recent_files']} files uploaded in the last hour")
        else:
            print("⚠️  ANALYSIS COMPLETE: No recent live data detected")
            print("   This may indicate a dashboard filtering issue or actual device connectivity problem")
    else:
        print("❌ ANALYSIS FAILED: Could not retrieve S3 data")
    print("=" * 50)

if __name__ == "__main__":
    main()