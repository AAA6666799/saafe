#!/usr/bin/env python3
"""
Simple AWS connectivity and permissions test
"""

import boto3
import json

def test_aws_connectivity():
    """Test AWS connectivity and permissions."""
    print("Testing AWS connectivity and permissions...")
    
    try:
        # Test 1: S3 Access
        print("\n1. Testing S3 access...")
        s3 = boto3.client('s3', region_name='us-east-1')
        response = s3.list_objects_v2(Bucket='data-collector-of-first-device', MaxKeys=1)
        print("   ✅ S3 access successful")
        
        # Test 2: Lambda Access
        print("\n2. Testing Lambda access...")
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        response = lambda_client.get_function(FunctionName='saafe-s3-data-processor')
        print("   ✅ Lambda access successful")
        
        # Test 3: SageMaker Access
        print("\n3. Testing SageMaker access...")
        sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
        response = sagemaker_client.describe_endpoint(EndpointName='fire-mvp-xgb-endpoint')
        print("   ✅ SageMaker access successful")
        
        # Test 4: SNS Access
        print("\n4. Testing SNS access...")
        sns_client = boto3.client('sns', region_name='us-east-1')
        response = sns_client.get_topic_attributes(
            TopicArn='arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
        )
        print("   ✅ SNS access successful")
        
        print("\n🎉 All AWS services are accessible!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_aws_connectivity()