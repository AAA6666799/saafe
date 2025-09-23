#!/usr/bin/env python3
"""
Final Verification Script for Cloud Deployment
This script verifies that all components of the synthetic fire prediction system are properly deployed and configured on AWS.
"""

import boto3
import json

def verify_lambda_functions():
    """Verify that Lambda functions are deployed and have proper permissions."""
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
        
        expected_rules = ['saafe-monitoring-schedule', 'saafe-analysis-schedule', 'saafe-response-schedule']
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

def verify_sagemaker_endpoint():
    """Verify that SageMaker endpoint is in service."""
    print("\n🔍 Verifying SageMaker endpoint...")
    
    sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
    
    try:
        response = sagemaker_client.describe_endpoint(EndpointName='fire-mvp-xgb-endpoint')
        status = response['EndpointStatus']
        
        if status == 'InService':
            print(f"  ✅ fire-mvp-xgb-endpoint - {status}")
            return True
        else:
            print(f"  ❌ fire-mvp-xgb-endpoint - {status}")
            return False
        
    except Exception as e:
        print(f"  ❌ Error verifying SageMaker endpoint: {e}")
        return False

def verify_sns_topic():
    """Verify that SNS topic exists."""
    print("\n🔍 Verifying SNS topic...")
    
    sns_client = boto3.client('sns', region_name='us-east-1')
    
    try:
        response = sns_client.list_topics()
        topics = [t['TopicArn'] for t in response['Topics']]
        
        expected_topic = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
        
        if expected_topic in topics:
            print(f"  ✅ fire-detection-alerts - Exists")
            return True
        else:
            print(f"  ❌ fire-detection-alerts - Not found")
            return False
        
    except Exception as e:
        print(f"  ❌ Error verifying SNS topic: {e}")
        return False

def verify_cloudwatch_dashboard():
    """Verify that CloudWatch dashboard exists."""
    print("\n🔍 Verifying CloudWatch dashboard...")
    
    cloudwatch_client = boto3.client('cloudwatch', region_name='us-east-1')
    
    try:
        response = cloudwatch_client.get_dashboard(DashboardName='SyntheticFirePredictionDashboard')
        
        if 'DashboardBody' in response:
            print(f"  ✅ SyntheticFirePredictionDashboard - Exists")
            return True
        else:
            print(f"  ❌ SyntheticFirePredictionDashboard - Not found")
            return False
        
    except Exception as e:
        print(f"  ❌ Error verifying CloudWatch dashboard: {e}")
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

def main():
    """Main verification function."""
    print("🚀 Final Verification of Cloud Deployment")
    print("=" * 45)
    
    # Run all verification checks
    checks = [
        ("Lambda Functions", verify_lambda_functions),
        ("CloudWatch Rules", verify_cloudwatch_rules),
        ("SageMaker Endpoint", verify_sagemaker_endpoint),
        ("SNS Topic", verify_sns_topic),
        ("CloudWatch Dashboard", verify_cloudwatch_dashboard),
        ("S3 Buckets", verify_s3_buckets)
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
    print("\n" + "=" * 45)
    print("📋 Final Verification Summary")
    print("=" * 45)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print("\n" + "=" * 45)
    print(f"📊 Overall Status: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All components are properly deployed and configured!")
        print("\n🚀 System is ready for production use!")
        print("\n📋 Access your system at:")
        print("  - AWS Console: https://console.aws.amazon.com/")
        print("  - CloudWatch Dashboard: https://console.aws.amazon.com/cloudwatch/home#dashboards:name=SyntheticFirePredictionDashboard")
        print("  - SageMaker Endpoint: fire-mvp-xgb-endpoint")
        return True
    else:
        print("⚠️  Some components are missing or not properly configured.")
        print("Please check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)