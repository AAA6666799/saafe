#!/usr/bin/env python3
"""
Test script to verify the dashboard can connect to AWS services
"""

import boto3
from botocore.exceptions import ClientError

def test_aws_connections():
    """Test connections to all required AWS services."""
    print("Testing AWS service connections...")
    
    # Test S3
    try:
        s3 = boto3.client('s3', region_name='us-east-1')
        response = s3.list_objects_v2(Bucket='data-collector-of-first-device', MaxKeys=1)
        print("✅ S3 connection successful")
    except ClientError as e:
        print(f"❌ S3 connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ S3 connection failed: {e}")
        return False
    
    # Test Lambda
    try:
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        response = lambda_client.get_function(FunctionName='saafe-s3-data-processor')
        print("✅ Lambda connection successful")
    except ClientError as e:
        print(f"❌ Lambda connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Lambda connection failed: {e}")
        return False
    
    # Test SageMaker
    try:
        sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
        response = sagemaker_client.describe_endpoint(EndpointName='fire-mvp-xgb-endpoint')
        print("✅ SageMaker connection successful")
    except ClientError as e:
        print(f"❌ SageMaker connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ SageMaker connection failed: {e}")
        return False
    
    # Test SNS
    try:
        sns_client = boto3.client('sns', region_name='us-east-1')
        response = sns_client.get_topic_attributes(
            TopicArn='arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
        )
        print("✅ SNS connection successful")
    except ClientError as e:
        print(f"❌ SNS connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ SNS connection failed: {e}")
        return False
    
    print("\n🎉 All AWS service connections successful!")
    print("The dashboard is ready for deployment.")
    return True

if __name__ == "__main__":
    test_aws_connections()