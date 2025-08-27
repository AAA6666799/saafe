import json
import boto3
import pandas as pd
from io import StringIO

def lambda_handler(event, context):
    """Lambda function to clean a single dataset"""
    
    # Get parameters from event
    input_bucket = event['input_bucket']
    input_key = event['input_key']
    output_bucket = event['output_bucket']
    output_key = event['output_key']
    
    s3_client = boto3.client('s3')
    
    try:
        # Download data from S3
        response = s3_client.get_object(Bucket=input_bucket, Key=input_key)
        csv_content = response['Body'].read().decode('utf-8')
        
        # Read CSV
        df = pd.read_csv(StringIO(csv_content))
        original_rows = len(df)
        
        # Clean data
        df = df.drop_duplicates()
        df = df.dropna()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        if 'is_anomaly' in df.columns:
            df['is_anomaly'] = df['is_anomaly'].map({
                'True': 1, 'False': 0, True: 1, False: 0
            })
        
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            
            mean_val = df['value'].mean()
            std_val = df['value'].std()
            
            if std_val > 0:
                df = df[abs(df['value'] - mean_val) <= 3 * std_val]
        
        cleaned_rows = len(df)
        
        # Upload cleaned data
        cleaned_csv = df.to_csv(index=False)
        s3_client.put_object(
            Bucket=output_bucket,
            Key=output_key,
            Body=cleaned_csv,
            ContentType='text/csv'
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Data cleaning completed successfully',
                'original_rows': original_rows,
                'cleaned_rows': cleaned_rows,
                'reduction_percent': round((1 - cleaned_rows/original_rows) * 100, 2)
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }