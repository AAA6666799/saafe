#!/usr/bin/env python3
"""
Verification Script for Cloud Deployment
This script verifies that all components of the synthetic fire prediction system are properly deployed on AWS.
"""

import boto3
import json

def verify_lambda_functions():
    """Verify that Lambda functions are deployed."""
    print("🔍 Verifying Lambda functions...")
    
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    try:
        response = lambda_client.list_functions()
        functions = [f['FunctionName'] for f in response['Functions'] if 'saafe' in f['FunctionName']]
        
        expected_functions = ['saafe-monitoring-agent', 'saafe-analysis-agent', 'saafe-response-agent']
        deployed_functions = []
        
        for func in expected_functions:
            if func in functions:
                deployed_functions.append(func)
                print(f"  ✅ {func} - Deployed")
            else:
                print(f"  ❌ {func} - Not found")
        
        return len(deployed_functions) == len(expected_functions)
        
    except Exception as e:
        print(f"  ❌ Error verifying Lambda functions: {e}")
        return False

def verify_cloudwatch_rules():
    """Verify that CloudWatch event rules are deployed."""
    print("\n🔍 Verifying CloudWatch event rules...")
    
    events_client = boto3.client('events', region_name='us-east-1')
    
    try:
        response = events_client.list_rules()
        rules = [r['Name'] for r in response['Rules'] if 'saafe' in r['Name']]
        
        expected_rules = ['saafe-monitoring-schedule']
        deployed_rules = []
        
        for rule in expected_rules:
            if rule in rules:
                deployed_rules.append(rule)
                print(f"  ✅ {rule} - Deployed")
            else:
                print(f"  ❌ {rule} - Not found")
        
        return len(deployed_rules) == len(expected_rules)
        
    except Exception as e:
        print(f"  ❌ Error verifying CloudWatch rules: {e}")
        return False

def verify_cloudwatch_logs():
    """Verify that CloudWatch log groups exist."""
    print("\n🔍 Verifying CloudWatch log groups...")
    
    logs_client = boto3.client('logs', region_name='us-east-1')
    
    try:
        response = logs_client.describe_log_groups(logGroupNamePrefix='/aws/lambda/saafe')
        log_groups = [lg['logGroupName'] for lg in response['logGroups']]
        
        expected_log_groups = ['/aws/lambda/saafe-monitoring-agent', 
                              '/aws/lambda/saafe-analysis-agent', 
                              '/aws/lambda/saafe-response-agent']
        existing_log_groups = []
        
        for log_group in expected_log_groups:
            if log_group in log_groups:
                existing_log_groups.append(log_group)
                print(f"  ✅ {log_group} - Exists")
            else:
                print(f"  ❌ {log_group} - Not found")
        
        return len(existing_log_groups) == len(expected_log_groups)
        
    except Exception as e:
        print(f"  ❌ Error verifying CloudWatch log groups: {e}")
        return False

def verify_s3_buckets():
    """Verify that S3 buckets exist."""
    print("\n🔍 Verifying S3 buckets...")
    
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.list_buckets()
        buckets = [b['Name'] for b in response['Buckets']]
        
        expected_buckets = ['processedd-synthetic-data', 'synthetic-data-4']
        existing_buckets = []
        
        for bucket in expected_buckets:
            if bucket in buckets:
                existing_buckets.append(bucket)
                print(f"  ✅ {bucket} - Exists")
            else:
                print(f"  ❌ {bucket} - Not found")
        
        return len(existing_buckets) >= 1  # At least one bucket should exist
        
    except Exception as e:
        print(f"  ❌ Error verifying S3 buckets: {e}")
        return False

def verify_sagemaker():
    """Verify that SageMaker is accessible."""
    print("\n🔍 Verifying SageMaker access...")
    
    try:
        sagemaker_client = boto3.client('sagemaker', region_name='us-west-2')
        
        # Try to list models (this will be empty but should not error)
        response = sagemaker_client.list_models(MaxResults=1)
        print("  ✅ SageMaker access - Verified")
        return True
        
    except Exception as e:
        print(f"  ❌ Error verifying SageMaker access: {e}")
        return False

def main():
    """Main verification function."""
    print("🚀 Verifying Cloud Deployment Status")
    print("=" * 40)
    
    # Run all verification checks
    checks = [
        ("Lambda Functions", verify_lambda_functions),
        ("CloudWatch Rules", verify_cloudwatch_rules),
        ("CloudWatch Logs", verify_cloudwatch_logs),
        ("S3 Buckets", verify_s3_buckets),
        ("SageMaker Access", verify_sagemaker)
    ]
    
    results = []
    for check_name, check_function in checks:
        try:
            result = check_function()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ❌ {check_name} - Error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 Verification Summary")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Overall Status: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All components are properly deployed!")
        print("\nNext steps:")
        print("1. Deploy trained models to SageMaker using deploy_models_to_sagemaker.py")
        print("2. Configure SNS topics for alerts")
        print("3. Set up additional CloudWatch event triggers")
        return True
    else:
        print("⚠️  Some components are missing or not properly deployed.")
        print("Please check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)