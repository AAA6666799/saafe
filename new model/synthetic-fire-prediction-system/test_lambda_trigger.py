#!/usr/bin/env python3
"""
Test script to manually trigger the Lambda function with a test event
"""

import boto3
import json
from datetime import datetime

def test_lambda_trigger():
    """Test triggering the Lambda function with a sample event."""
    # Initialize Lambda client
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    # Create a test event that simulates an S3 object creation
    test_event = {
        "Records": [
            {
                "eventVersion": "2.1",
                "eventSource": "aws:s3",
                "awsRegion": "us-east-1",
                "eventTime": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "s3SchemaVersion": "1.0",
                    "configurationId": "test-trigger",
                    "bucket": {
                        "name": "data-collector-of-first-device",
                        "arn": "arn:aws:s3:::data-collector-of-first-device"
                    },
                    "object": {
                        "key": "gas_data_20250909_121004_2.csv",
                        "size": 67,
                        "eTag": "test-etag",
                        "sequencer": "test-sequencer"
                    }
                }
            }
        ]
    }
    
    try:
        print("Testing Lambda function trigger...")
        response = lambda_client.invoke(
            FunctionName='saafe-s3-data-processor',
            InvocationType='RequestResponse',
            Payload=json.dumps(test_event)
        )
        
        # Print response
        print(f"Status Code: {response['StatusCode']}")
        print(f"Executed Version: {response.get('ExecutedVersion', 'N/A')}")
        
        # Print response payload
        payload = response['Payload'].read().decode('utf-8')
        print(f"Response Payload: {payload}")
        
        if 'FunctionError' in response:
            print(f"Function Error: {response['FunctionError']}")
            
    except Exception as e:
        print(f"Error invoking Lambda function: {e}")

if __name__ == "__main__":
    test_lambda_trigger()