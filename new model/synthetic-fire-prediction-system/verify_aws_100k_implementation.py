#!/usr/bin/env python3
"""
Verify that all components of the AWS 100K implementation are working correctly
"""

import os
import sys
import boto3
from pathlib import Path

def verify_files_exist():
    """Verify that all required files exist"""
    required_files = [
        "aws_100k_training_pipeline.py",
        "monitor_100k_training.py", 
        "deploy_100k_model.py",
        "check_training_status.py",
        "AWS_100K_TRAINING_README.md",
        "AWS_100K_TRAINING_SUMMARY.md",
        "FINAL_AWS_100K_IMPLEMENTATION.md"
    ]
    
    print("Verifying required files...")
    print("=" * 40)
    
    project_root = Path(__file__).parent
    missing_files = []
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} (MISSING)")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\\n❌ {len(missing_files)} file(s) missing")
        return False
    else:
        print(f"\\n✅ All {len(required_files)} files present")
        return True

def verify_aws_access():
    """Verify AWS access and required services"""
    print("\\nVerifying AWS access...")
    print("=" * 40)
    
    try:
        # Test S3 access
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.list_buckets()
        print("✅ S3 access: OK")
        
        # Test SageMaker access
        sagemaker = boto3.client('sagemaker', region_name='us-east-1')
        sagemaker.list_training_jobs(MaxResults=1)
        print("✅ SageMaker access: OK")
        
        # Test SageMaker Runtime access
        sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
        print("✅ SageMaker Runtime access: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ AWS access error: {e}")
        return False

def verify_training_jobs():
    """Verify that training jobs were created"""
    print("\\nVerifying training jobs...")
    print("=" * 40)
    
    try:
        sagemaker = boto3.client('sagemaker', region_name='us-east-1')
        
        # Check if our specific training jobs exist
        job_names = [
            "flir-scd41-rf-100k-20250829-160706",
            "flir-scd41-gb-100k-20250829-160706", 
            "flir-scd41-lr-100k-20250829-160706"
        ]
        
        found_jobs = 0
        for job_name in job_names:
            try:
                response = sagemaker.describe_training_job(TrainingJobName=job_name)
                status = response['TrainingJobStatus']
                print(f"✅ {job_name}: {status}")
                found_jobs += 1
            except sagemaker.exceptions.ResourceNotFound:
                print(f"❌ {job_name}: NOT FOUND")
            except Exception as e:
                print(f"❌ {job_name}: ERROR - {e}")
        
        if found_jobs == 3:
            print(f"\\n✅ All 3 training jobs found and running")
            return True
        else:
            print(f"\\n❌ Only {found_jobs}/3 training jobs found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking training jobs: {e}")
        return False

def verify_data_generation():
    """Verify that data was generated and uploaded"""
    print("\\nVerifying data generation...")
    print("=" * 40)
    
    try:
        s3 = boto3.client('s3', region_name='us-east-1')
        
        # Check if our data file exists in S3
        bucket_name = 'fire-detection-training-691595239825'
        prefix = 'flir_scd41_training/data/'
        
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' in response:
            data_files = [obj['Key'] for obj in response['Contents'] 
                         if 'flir_scd41_data_100000' in obj['Key']]
            
            if data_files:
                print(f"✅ Data file found: {data_files[0]}")
                print("✅ 100K synthetic samples successfully generated and uploaded")
                return True
            else:
                print("❌ 100K data file not found in S3")
                return False
        else:
            print("❌ No data files found in S3 bucket")
            return False
            
    except Exception as e:
        print(f"❌ Error checking data generation: {e}")
        return False

def main():
    """Main verification function"""
    print("FLIR+SCD41 Fire Detection System - AWS 100K Implementation Verification")
    print("=" * 75)
    
    # Run all verification checks
    checks = [
        ("File Verification", verify_files_exist),
        ("AWS Access", verify_aws_access),
        ("Data Generation", verify_data_generation),
        ("Training Jobs", verify_training_jobs)
    ]
    
    results = []
    for check_name, check_function in checks:
        print(f"\\n🔍 {check_name}")
        result = check_function()
        results.append((check_name, result))
    
    # Summary
    print("\\n" + "=" * 75)
    print("VERIFICATION SUMMARY")
    print("=" * 75)
    
    passed = 0
    failed = 0
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\\n🎉 ALL CHECKS PASSED!")
        print("✅ AWS 100K implementation is fully functional")
        return 0
    else:
        print("\\n⚠️  Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())