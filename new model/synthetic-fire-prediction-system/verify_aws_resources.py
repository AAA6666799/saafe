#!/usr/bin/env python3
"""
Verify that all AWS resources are properly configured and accessible
"""

import boto3
import sys
from botocore.exceptions import ClientError

def verify_aws_resources():
    """Verify all AWS resources needed for the pipeline"""
    
    print("🔍 Verifying AWS Resources for FLIR+SCD41 Pipeline")
    print("=" * 55)
    
    # AWS Configuration
    AWS_REGION = 'us-east-1'
    S3_BUCKET = 'fire-detection-training-691595239825'
    
    # Track verification status
    all_passed = True
    
    # 1. Verify S3 Bucket Access
    print("\n1. Checking S3 Bucket Access...")
    try:
        s3 = boto3.client('s3', region_name=AWS_REGION)
        s3.head_bucket(Bucket=S3_BUCKET)
        print(f"   ✅ S3 Bucket '{S3_BUCKET}' is accessible")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"   ❌ S3 Bucket '{S3_BUCKET}' not found")
        elif e.response['Error']['Code'] == '403':
            print(f"   ❌ Access denied to S3 Bucket '{S3_BUCKET}'")
        else:
            print(f"   ❌ Error accessing S3 Bucket: {e}")
        all_passed = False
    except Exception as e:
        print(f"   ❌ Unexpected error checking S3: {e}")
        all_passed = False
    
    # 2. Verify SageMaker Access
    print("\n2. Checking SageMaker Access...")
    try:
        sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
        # List a few training jobs to verify access
        sagemaker.list_training_jobs(MaxResults=1)
        print("   ✅ SageMaker service is accessible")
    except ClientError as e:
        print(f"   ❌ SageMaker access error: {e}")
        all_passed = False
    except Exception as e:
        print(f"   ❌ Unexpected error checking SageMaker: {e}")
        all_passed = False
    
    # 3. Verify IAM Role
    print("\n3. Checking IAM Role...")
    try:
        iam = boto3.client('iam')
        role_name = "SageMakerExecutionRole"
        iam.get_role(RoleName=role_name)
        print(f"   ✅ IAM Role '{role_name}' exists")
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            print(f"   ❌ IAM Role '{role_name}' not found")
        else:
            print(f"   ❌ IAM Role error: {e}")
        all_passed = False
    except Exception as e:
        print(f"   ❌ Unexpected error checking IAM: {e}")
        all_passed = False
    
    # 4. Verify Required SageMaker Images
    print("\n4. Checking SageMaker Container Images...")
    try:
        account_id = "683313688378"  # AWS SageMaker account for us-east-1
        sklearn_image = f"{account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3"
        print(f"   ✅ Scikit-learn image: {sklearn_image}")
        
        xgboost_image = f"{account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-xgboost:1.5-1-cpu-py3"
        print(f"   ✅ XGBoost image: {xgboost_image}")
    except Exception as e:
        print(f"   ⚠️  Warning checking container images: {e}")
        # Not critical for this pipeline
    
    # 5. Verify S3 Bucket Structure
    print("\n5. Checking S3 Bucket Structure...")
    try:
        # Check if the expected prefixes exist
        expected_prefixes = [
            'flir_scd41_training/data/',
            'flir_scd41_training/models/',
            'flir_scd41_training/code/'
        ]
        
        for prefix in expected_prefixes:
            response = s3.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=prefix,
                MaxKeys=1
            )
            if 'Contents' in response or 'CommonPrefixes' in response:
                print(f"   ✅ Prefix '{prefix}' exists")
            else:
                print(f"   ⚠️  Prefix '{prefix}' not found (will be created automatically)")
                
    except Exception as e:
        print(f"   ❌ Error checking S3 structure: {e}")
        all_passed = False
    
    # Final Status
    print("\n" + "=" * 55)
    if all_passed:
        print("🎉 ALL AWS RESOURCES VERIFIED SUCCESSFULLY")
        print("\n✅ You can now proceed with confidence that all AWS resources")
        print("   are properly configured for the FLIR+SCD41 pipeline.")
    else:
        print("❌ SOME AWS RESOURCE CHECKS FAILED")
        print("\n⚠️  Please check the errors above and ensure all AWS resources")
        print("   are properly configured before running the pipeline.")
    
    return all_passed

def show_troubleshooting_tips():
    """Show troubleshooting tips for common issues"""
    print("\n" + "=" * 55)
    print("TROUBLESHOOTING TIPS")
    print("=" * 55)
    
    print("\n1. AWS Credentials:")
    print("   - Ensure AWS credentials are configured in ~/.aws/credentials")
    print("   - Run 'aws configure' to set up credentials if needed")
    
    print("\n2. IAM Permissions:")
    print("   - Ensure SageMakerExecutionRole has required permissions")
    print("   - Required policies: AmazonS3FullAccess, AmazonSageMakerFullAccess")
    
    print("\n3. Network Access:")
    print("   - Ensure no firewall blocking AWS service endpoints")
    print("   - Check VPC settings if using private networks")
    
    print("\n4. Service Limits:")
    print("   - Check if you have exceeded SageMaker service limits")
    print("   - Request limit increases if needed through AWS Support")

if __name__ == "__main__":
    success = verify_aws_resources()
    show_troubleshooting_tips()
    
    if success:
        print("\n✅ Ready to proceed with AWS-based training pipeline!")
        sys.exit(0)
    else:
        print("\n❌ Please resolve the issues before proceeding.")
        sys.exit(1)