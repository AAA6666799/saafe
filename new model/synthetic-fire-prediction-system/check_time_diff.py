import boto3
from datetime import datetime

# Initialize S3 client
s3_client = boto3.client('s3', region_name='us-east-1')

try:
    # List objects in the bucket
    response = s3_client.list_objects_v2(
        Bucket='data-collector-of-first-device',
        MaxKeys=1
    )
    
    if 'Contents' in response:
        # Get the most recent file
        obj = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)[0]
        
        print(f"Most recent file: {obj['Key']}")
        print(f"Last modified: {obj['LastModified']}")
        
        # Get current time
        now = datetime.now(obj['LastModified'].tzinfo)
        print(f"Current time: {now}")
        
        # Calculate time difference
        time_diff = now - obj['LastModified']
        print(f"Time difference: {time_diff}")
        print(f"Time difference in hours: {time_diff.total_seconds() / 3600}")
        
    else:
        print("No files found in the bucket")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()