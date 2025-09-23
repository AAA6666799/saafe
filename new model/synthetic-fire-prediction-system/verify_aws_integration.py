#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - AWS Integration Verification
This script verifies that all AWS components are properly configured and working together.
"""

import boto3
import json
from datetime import datetime

def verify_aws_integration():
    """Verify all AWS components for the fire detection system."""
    
    print("FLIR+SCD41 Fire Detection System - AWS Integration Verification")
    print("=" * 65)
    
    # AWS Configuration
    AWS_REGION = 'us-east-1'
    S3_BUCKET = 'fire-detection-training-691595239825'
    
    # Track verification results
    results = {
        's3': False,
        'sagemaker': False,
        'iam': False,
        'overall': False
    }
    
    try:
        # 1. Verify S3 access
        print("\n1. Verifying S3 access...")
        s3 = boto3.client('s3', region_name=AWS_REGION)
        
        # List buckets
        buckets_response = s3.list_buckets()
        bucket_names = [bucket['Name'] for bucket in buckets_response['Buckets']]
        
        if S3_BUCKET in bucket_names:
            print(f"   ✓ S3 bucket '{S3_BUCKET}' found")
            results['s3'] = True
            
            # Check if we can list objects
            try:
                objects_response = s3.list_objects_v2(Bucket=S3_BUCKET, MaxKeys=1)
                print("   ✓ S3 bucket is accessible and readable")
            except Exception as e:
                print(f"   ! S3 bucket access issue: {e}")
        else:
            print(f"   ✗ S3 bucket '{S3_BUCKET}' not found")
            
    except Exception as e:
        print(f"   ✗ S3 verification failed: {e}")
    
    try:
        # 2. Verify SageMaker access
        print("\n2. Verifying SageMaker access...")
        sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
        
        # List training jobs
        training_response = sagemaker.list_training_jobs(MaxResults=1)
        print("   ✓ SageMaker training jobs accessible")
        results['sagemaker'] = True
        
        # List models
        models_response = sagemaker.list_models(MaxResults=1)
        print("   ✓ SageMaker models accessible")
        
    except Exception as e:
        print(f"   ✗ SageMaker verification failed: {e}")
    
    try:
        # 3. Verify IAM permissions
        print("\n3. Verifying IAM permissions...")
        iam = boto3.client('iam', region_name=AWS_REGION)
        
        # Get current user
        user_response = iam.get_user()
        user_name = user_response['User']['UserName']
        print(f"   ✓ IAM user '{user_name}' authenticated")
        results['iam'] = True
        
    except Exception as e:
        print(f"   ✗ IAM verification failed: {e}")
    
    # Overall status
    results['overall'] = all([results['s3'], results['sagemaker'], results['iam']])
    
    print("\n" + "=" * 65)
    print("VERIFICATION SUMMARY")
    print("=" * 65)
    print(f"S3 Access:       {'✓ PASS' if results['s3'] else '✗ FAIL'}")
    print(f"SageMaker Access: {'✓ PASS' if results['sagemaker'] else '✗ FAIL'}")
    print(f"IAM Permissions:  {'✓ PASS' if results['iam'] else '✗ FAIL'}")
    print("-" * 65)
    print(f"Overall Status:   {'✓ ALL SYSTEMS READY' if results['overall'] else '✗ ISSUES DETECTED'}")
    
    if results['overall']:
        print("\n🎉 Your AWS environment is ready for FLIR+SCD41 fire detection system!")
        print("\nNext steps:")
        print("1. Generate training data: python aws_training_pipeline_fixed.py")
        print("2. Train models: python sagemaker_training_example.py")
        print("3. Deploy models: python sagemaker_deployment_example.py")
    else:
        print("\n⚠️  Please resolve the issues above before proceeding.")
    
    return results

def test_data_upload():
    """Test uploading sample data to S3."""
    
    print("\n" + "=" * 65)
    print("DATA UPLOAD TEST")
    print("=" * 65)
    
    AWS_REGION = 'us-east-1'
    S3_BUCKET = 'fire-detection-training-691595239825'
    S3_PREFIX = 'flir_scd41_training'
    
    try:
        # Create sample data
        sample_data = {
            "test_timestamp": datetime.now().isoformat(),
            "test_data": "This is a test upload from the verification script",
            "features": ["t_mean", "t_std", "gas_val"],
            "sample_values": {
                "t_mean": 25.5,
                "t_std": 2.1,
                "gas_val": 420.0
            }
        }
        
        # Save to local file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f, indent=2)
            temp_file = f.name
        
        # Upload to S3
        s3 = boto3.client('s3', region_name=AWS_REGION)
        s3_key = f"{S3_PREFIX}/test/verification_test_{int(datetime.now().timestamp())}.json"
        
        s3.upload_file(temp_file, S3_BUCKET, s3_key)
        print(f"   ✓ Sample data uploaded to s3://{S3_BUCKET}/{s3_key}")
        
        # Clean up local file
        os.unlink(temp_file)
        
        # Verify upload
        s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        print("   ✓ Upload verified successfully")
        
        # Clean up S3 object
        s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)
        print("   ✓ Test object cleaned up from S3")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Data upload test failed: {e}")
        return False

if __name__ == "__main__":
    # Run verification
    results = verify_aws_integration()
    
    # If basic verification passed, test data upload
    if results['overall']:
        test_data_upload()
    
    print("\n" + "=" * 65)
    print("VERIFICATION COMPLETE")
    print("=" * 65)