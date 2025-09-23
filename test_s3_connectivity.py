#!/usr/bin/env python3
"""
Test S3 connectivity to the data-collector-of-first-device bucket
"""

import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
import pytz

def test_s3_connectivity():
    """Test connectivity to the S3 bucket and check for recent files."""
    print("🔍 Testing S3 connectivity to 'data-collector-of-first-device' bucket...")
    print("=" * 60)
    
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'data-collector-of-first-device'
        
        # Test 1: Check if bucket exists and is accessible
        print("Test 1: Checking bucket accessibility...")
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print("✅ Bucket exists and is accessible")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print("❌ Bucket does not exist")
                return False
            elif error_code == '403':
                print("❌ Access denied to bucket")
                return False
            else:
                print(f"❌ Error accessing bucket: {e}")
                return False
        
        # Test 2: List recent files
        print("\nTest 2: Listing recent files...")
        try:
            # Get current time
            utc_now = datetime.now(pytz.UTC)
            one_hour_ago = utc_now - timedelta(hours=1)
            today_str = utc_now.strftime('%Y%m%d')
            
            # Check for thermal data files from today
            print(f"Checking for thermal data files from {today_str}...")
            thermal_response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=f'thermal-data/thermal_data_{today_str}',
                MaxKeys=10
            )
            
            thermal_files = []
            if 'Contents' in thermal_response:
                for obj in thermal_response['Contents']:
                    file_time = obj['LastModified']
                    if file_time.tzinfo is None:
                        file_time = file_time.replace(tzinfo=pytz.UTC)
                    if file_time > one_hour_ago:
                        thermal_files.append(obj)
            
            print(f"Found {len(thermal_files)} thermal files from the last hour")
            
            # Check for gas data files from today
            print(f"Checking for gas data files from {today_str}...")
            gas_response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=f'gas-data/gas_data_{today_str}',
                MaxKeys=10
            )
            
            gas_files = []
            if 'Contents' in gas_response:
                for obj in gas_response['Contents']:
                    file_time = obj['LastModified']
                    if file_time.tzinfo is None:
                        file_time = file_time.replace(tzinfo=pytz.UTC)
                    if file_time > one_hour_ago:
                        gas_files.append(obj)
            
            print(f"Found {len(gas_files)} gas files from the last hour")
            
            # Show some recent files
            if thermal_files or gas_files:
                print("\nRecent files:")
                all_recent_files = thermal_files + gas_files
                # Sort by last modified time
                all_recent_files.sort(key=lambda x: x['LastModified'], reverse=True)
                
                for i, obj in enumerate(all_recent_files[:5]):  # Show top 5
                    file_type = "thermal" if obj in thermal_files else "gas"
                    print(f"  {i+1}. {obj['Key']} ({obj['LastModified']} UTC) [{file_type}]")
            else:
                print("\nNo recent files found in the last hour")
                
                # Check for any files at all
                print("Checking for any files in the bucket...")
                any_files_response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    MaxKeys=10
                )
                
                if 'Contents' in any_files_response:
                    print(f"Found {any_files_response['KeyCount']} total files in bucket")
                    # Show the most recent files
                    sorted_files = sorted(any_files_response['Contents'], 
                                        key=lambda x: x['LastModified'], reverse=True)
                    print("Most recent files:")
                    for i, obj in enumerate(sorted_files[:5]):  # Show top 5
                        print(f"  {i+1}. {obj['Key']} ({obj['LastModified']} UTC)")
                else:
                    print("No files found in the bucket at all")
            
            return True
            
        except ClientError as e:
            print(f"❌ Error listing files: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to initialize AWS client: {e}")
        return False

def main():
    """Main function."""
    print("🧪 S3 Connectivity Test for Fire Detection System")
    print("=" * 60)
    
    success = test_s3_connectivity()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 S3 connectivity test completed successfully!")
        print("\nNext steps:")
        print("1. If you see recent files, your devices are sending data correctly")
        print("2. If you see old files but no recent ones, check your device connectivity")
        print("3. If you see no files at all, verify your devices are configured correctly")
    else:
        print("❌ S3 connectivity test failed!")
        print("\nTroubleshooting steps:")
        print("1. Verify your AWS credentials are configured correctly")
        print("2. Check that you have permissions to access the bucket")
        print("3. Ensure the bucket name is correct: 'data-collector-of-first-device'")
        print("4. Verify your internet connection")

if __name__ == "__main__":
    main()