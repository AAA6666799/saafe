#!/usr/bin/env python3
"""
Test script to verify AWS connections for the Saafe dashboard
"""

import boto3
import sys
from botocore.exceptions import ClientError

def test_aws_connection():
    """Test connection to required AWS services"""
    print("🔍 Testing AWS connections for Saafe Dashboard...")
    print("=" * 50)
    
    # Test S3 connection
    print("1. Testing S3 connection...")
    try:
        s3 = boto3.client('s3', region_name='us-east-1')
        # Test if we can access the bucket
        bucket_name = 'data-collector-of-first-device'
        s3.head_bucket(Bucket=bucket_name)
        print(f"   ✅ S3 connection successful - Bucket '{bucket_name}' accessible")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"   ❌ Bucket '{bucket_name}' not found")
        elif error_code == '403':
            print(f"   ❌ Access denied to bucket '{bucket_name}'")
        else:
            print(f"   ❌ S3 connection failed: {e}")
    except Exception as e:
        print(f"   ❌ S3 connection failed: {e}")
    
    # Test Lambda connection
    print("\n2. Testing Lambda connection...")
    try:
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        response = lambda_client.get_function(FunctionName='saafe-s3-data-processor')
        print("   ✅ Lambda connection successful")
        print(f"   Function: {response['Configuration']['FunctionName']}")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ResourceNotFoundException':
            print("   ❌ Lambda function 'saafe-s3-data-processor' not found")
        else:
            print(f"   ❌ Lambda connection failed: {e}")
    except Exception as e:
        print(f"   ❌ Lambda connection failed: {e}")
    
    # Test SageMaker connection
    print("\n3. Testing SageMaker connection...")
    try:
        sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
        response = sagemaker_client.describe_endpoint(EndpointName='fire-mvp-xgb-endpoint')
        print("   ✅ SageMaker connection successful")
        print(f"   Endpoint: {response['EndpointName']} - Status: {response['EndpointStatus']}")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ValidationException':
            print("   ❌ SageMaker endpoint 'fire-mvp-xgb-endpoint' not found")
        else:
            print(f"   ❌ SageMaker connection failed: {e}")
    except Exception as e:
        print(f"   ❌ SageMaker connection failed: {e}")
    
    # Test CloudWatch connection
    print("\n4. Testing CloudWatch connection...")
    try:
        cloudwatch_client = boto3.client('cloudwatch', region_name='us-east-1')
        # Simple test - list some metrics
        response = cloudwatch_client.list_metrics(Namespace='AWS/Lambda', MaxRecords=1)
        print("   ✅ CloudWatch connection successful")
    except Exception as e:
        print(f"   ❌ CloudWatch connection failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ AWS connection test completed!")

if __name__ == "__main__":
    test_aws_connection()