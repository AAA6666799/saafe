#!/usr/bin/env python3
"""
Simple AWS connection test
"""

import boto3

def test_aws_connection():
    """Test basic AWS connectivity."""
    print("Testing AWS connectivity...")
    
    try:
        # Test S3 connection
        s3 = boto3.client('s3', region_name='us-east-1')
        buckets = s3.list_buckets()
        print(f"✅ S3 connection successful. Found {len(buckets.get('Buckets', []))} buckets")
        
        # Test if our specific bucket exists
        try:
            response = s3.list_objects_v2(Bucket='data-collector-of-first-device', MaxKeys=1)
            print("✅ Target S3 bucket accessible")
        except Exception as e:
            print(f"❌ Target S3 bucket not accessible: {e}")
            
    except Exception as e:
        print(f"❌ AWS connection failed: {e}")

if __name__ == "__main__":
    test_aws_connection()