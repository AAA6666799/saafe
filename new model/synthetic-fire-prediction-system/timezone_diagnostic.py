import boto3
from datetime import datetime, timedelta
import pytz

# Initialize S3 client
s3_client = boto3.client('s3', region_name='us-east-1')

try:
    # List objects in the bucket
    response = s3_client.list_objects_v2(
        Bucket='data-collector-of-first-device',
        MaxKeys=10
    )
    
    if 'Contents' in response:
        # Get current time in different timezones
        utc_now = datetime.utcnow().replace(tzinfo=pytz.UTC)
        local_now = datetime.now()
        
        print("Current Times:")
        print(f"  UTC: {utc_now}")
        print(f"  Local: {local_now}")
        print()
        
        # Sort files by last modified
        sorted_files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
        
        print("Most Recent Files:")
        print("==================")
        for i, obj in enumerate(sorted_files[:5]):
            key = obj['Key']
            last_modified = obj['LastModified']
            
            print(f"{i+1}. File: {key}")
            print(f"   Last Modified: {last_modified}")
            print(f"   Last Modified (UTC): {last_modified.astimezone(pytz.UTC) if last_modified.tzinfo else 'No timezone info'}")
            
            # Calculate time differences
            if last_modified.tzinfo:
                time_diff_utc = utc_now - last_modified
                print(f"   Age (UTC calc): {time_diff_utc}")
                print(f"   Within last hour (UTC): {time_diff_utc < timedelta(hours=1)}")
            else:
                # If no timezone info, assume it's UTC
                time_diff_assumed = utc_now - last_modified.replace(tzinfo=pytz.UTC)
                print(f"   Age (Assumed UTC): {time_diff_assumed}")
                print(f"   Within last hour (Assumed UTC): {time_diff_assumed < timedelta(hours=1)}")
            print()
    else:
        print("No files found in the bucket")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()