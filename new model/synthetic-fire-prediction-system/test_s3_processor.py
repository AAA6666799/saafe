#!/usr/bin/env python3
"""
Test script for the S3 Data Processor Lambda function.
This script creates sample S3 events and tests the Lambda function locally.
"""

import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
import io

def create_sample_thermal_data():
    """Create sample thermal data similar to MLX90640 output."""
    # Create sample data with 32x24 = 768 pixels
    data = np.random.normal(25.0, 5.0, (1, 768))  # 1 row, 768 columns
    
    # Add some hot spots occasionally
    if np.random.random() > 0.7:
        hot_pixels = np.random.choice(768, size=50, replace=False)
        data[0, hot_pixels] = np.random.normal(60.0, 10.0, 50)
    
    # Create DataFrame
    columns = [f"pixel_{i}" for i in range(768)]
    df = pd.DataFrame(data, columns=columns)
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def create_sample_gas_data():
    """Create sample gas data."""
    # Create sample data
    data = {
        'CO': [np.random.normal(400.0, 50.0)],
        'NO2': [np.random.normal(20.0, 5.0)],
        'VOC': [np.random.normal(1.0, 0.2)]
    }
    
    df = pd.DataFrame(data)
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def create_sample_s3_event(bucket, key, file_content):
    """Create a sample S3 event for testing."""
    return {
        "Records": [
            {
                "eventVersion": "2.1",
                "eventSource": "aws:s3",
                "awsRegion": "us-east-1",
                "eventTime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "s3SchemaVersion": "1.0",
                    "configurationId": "test-event",
                    "bucket": {
                        "name": bucket,
                        "arn": f"arn:aws:s3:::{bucket}"
                    },
                    "object": {
                        "key": key,
                        "size": len(file_content),
                        "eTag": "test-etag",
                        "sequencer": "test-sequencer"
                    }
                }
            }
        ]
    }

def upload_test_files():
    """Upload test files to S3 bucket."""
    s3_client = boto3.client('s3')
    bucket_name = 'data-collector-of-first-device'
    
    # Create and upload thermal data file
    thermal_content = create_sample_thermal_data()
    thermal_key = f"thermal_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=thermal_key,
        Body=thermal_content
    )
    
    print(f"Uploaded thermal data: {thermal_key}")
    
    # Create and upload gas data file
    gas_content = create_sample_gas_data()
    gas_key = f"gas_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=gas_key,
        Body=gas_content
    )
    
    print(f"Uploaded gas data: {gas_key}")
    
    return [
        {"bucket": bucket_name, "key": thermal_key},
        {"bucket": bucket_name, "key": gas_key}
    ]

def test_lambda_locally():
    """Test the Lambda function locally."""
    # Import the Lambda function
    import sys
    sys.path.append('/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system/src/aws/lambda')
    
    # Import the function
    try:
        import s3_data_processor
    except ImportError as e:
        print(f"Error importing s3_data_processor: {e}")
        return
    
    # Create sample events
    bucket_name = 'data-collector-of-first-device'
    
    # Test with thermal data
    thermal_content = create_sample_thermal_data()
    thermal_event = create_sample_s3_event(
        bucket_name, 
        f"thermal_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        thermal_content
    )
    
    print("Testing with thermal data...")
    result = s3_data_processor.lambda_handler(thermal_event, None)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test with gas data
    gas_content = create_sample_gas_data()
    gas_event = create_sample_s3_event(
        bucket_name,
        f"gas_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        gas_content
    )
    
    print("\nTesting with gas data...")
    result = s3_data_processor.lambda_handler(gas_event, None)
    print(f"Result: {json.dumps(result, indent=2)}")

def test_lambda_remotely():
    """Test the deployed Lambda function."""
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    # Create sample event
    bucket_name = 'data-collector-of-first-device'
    thermal_content = create_sample_thermal_data()
    thermal_event = create_sample_s3_event(
        bucket_name,
        f"thermal_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        thermal_content
    )
    
    try:
        response = lambda_client.invoke(
            FunctionName='saafe-s3-data-processor',
            InvocationType='RequestResponse',
            Payload=json.dumps(thermal_event)
        )
        
        result = json.loads(response['Payload'].read())
        print(f"Remote Lambda test result: {json.dumps(result, indent=2)}")
        
    except Exception as e:
        print(f"Error testing remote Lambda: {e}")

def main():
    """Main test function."""
    print("🔥 S3 Data Processor Testing")
    print("=" * 40)
    
    # Test locally
    print("\n🧪 Testing Lambda function locally...")
    test_lambda_locally()
    
    # Test remotely (if function is deployed)
    print("\n☁️  Testing deployed Lambda function...")
    test_lambda_remotely()
    
    # Upload test files to S3 (if you want to trigger the actual S3 event)
    print("\n📤 Uploading test files to S3...")
    try:
        uploaded_files = upload_test_files()
        print(f"Uploaded {len(uploaded_files)} files to S3")
        print("These files should trigger the S3 data processor Lambda function")
    except Exception as e:
        print(f"Error uploading test files: {e}")
        print("Make sure you have the necessary AWS credentials configured")
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()