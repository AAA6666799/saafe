import boto3
from datetime import datetime, timedelta

# Initialize S3 client
s3_client = boto3.client('s3', region_name='us-east-1')

try:
    # List objects in the bucket
    response = s3_client.list_objects_v2(
        Bucket='data-collector-of-first-device'
    )
    
    if 'Contents' in response:
        # Get today's date
        today = datetime.now().date()
        
        # Filter for files from today
        today_files = []
        for obj in response['Contents']:
            if obj['LastModified'].date() == today:
                today_files.append(obj)
        
        print(f'Files from today ({today}): {len(today_files)}')
        
        # Show today's files
        if today_files:
            sorted_today = sorted(today_files, key=lambda x: x['LastModified'], reverse=True)
            print('Today\'s files:')
            for i, obj in enumerate(sorted_today[:10]):
                print(f'  {i+1}. {obj["LastModified"]} - {obj["Key"]}')
        else:
            print('No files from today')
            
        # Check for any files with today's date in the filename
        print('\nFiles with today\'s date in filename:')
        today_str = today.strftime('%Y%m%d')
        filename_matches = []
        for obj in response['Contents']:
            if today_str in obj['Key']:
                filename_matches.append(obj)
        
        print(f'Files with {today_str} in filename: {len(filename_matches)}')
        if filename_matches:
            sorted_matches = sorted(filename_matches, key=lambda x: x['LastModified'], reverse=True)
            for i, obj in enumerate(sorted_matches[:5]):
                print(f'  {i+1}. {obj["LastModified"]} - {obj["Key"]}')
                
    else:
        print('No files found in the bucket')
        
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()