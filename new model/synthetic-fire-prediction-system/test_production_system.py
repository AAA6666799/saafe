#!/usr/bin/env python3
"""
Production System Test Script
This script tests the end-to-end functionality of the deployed system.
"""

import boto3
import json
import time
from datetime import datetime

def test_sagemaker_endpoint():
    """Test the SageMaker endpoint with sample data."""
    print("🔍 Testing SageMaker endpoint...")
    
    # Initialize SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
    
    # Sample test data (normal conditions) - CSV format for XGBoost
    # Format: t_mean,t_std,t_max,t_p95,t_hot_area_pct,t_hot_largest_blob_pct,t_grad_mean,t_grad_std,t_diff_mean,t_diff_std,flow_mag_mean,flow_mag_std,tproxy_val,tproxy_delta,tproxy_vel,gas_val,gas_delta,gas_vel
    test_data_normal = "22.5,1.2,25.1,24.8,0.5,0.3,0.1,0.05,0.2,0.1,0.3,0.1,23.0,0.5,0.1,410.0,5.0,1.0"
    
    # Sample test data (potential fire conditions)
    test_data_fire = "45.2,8.7,78.5,72.1,25.3,18.7,3.2,1.8,2.9,1.5,4.2,2.1,52.0,15.0,3.2,850.0,120.0,8.5"
    
    endpoint_name = 'fire-mvp-xgb-endpoint'
    
    try:
        # Test normal conditions
        print("  Testing normal conditions...")
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=test_data_normal
        )
        
        result = response['Body'].read().decode()
        print(f"    Normal conditions prediction: {result}")
        
        # Test fire conditions
        print("  Testing fire conditions...")
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=test_data_fire
        )
        
        result = response['Body'].read().decode()
        print(f"    Fire conditions prediction: {result}")
        
        print("  ✅ SageMaker endpoint test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing SageMaker endpoint: {e}")
        return False

def test_lambda_functions():
    """Test that Lambda functions are properly configured."""
    print("\n🔍 Testing Lambda function configurations...")
    
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    try:
        # Check monitoring agent
        response = lambda_client.get_function(FunctionName='saafe-monitoring-agent')
        print(f"  ✅ saafe-monitoring-agent - Configured")
        
        # Check analysis agent
        response = lambda_client.get_function(FunctionName='saafe-analysis-agent')
        print(f"  ✅ saafe-analysis-agent - Configured")
        
        # Check response agent
        response = lambda_client.get_function(FunctionName='saafe-response-agent')
        print(f"  ✅ saafe-response-agent - Configured")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing Lambda functions: {e}")
        return False

def test_cloudwatch_events():
    """Test that CloudWatch events are properly configured."""
    print("\n🔍 Testing CloudWatch event rules...")
    
    events_client = boto3.client('events', region_name='us-east-1')
    
    try:
        # Check monitoring schedule
        response = events_client.describe_rule(Name='saafe-monitoring-schedule')
        print(f"  ✅ saafe-monitoring-schedule - {response['State']}")
        
        # Check analysis schedule
        response = events_client.describe_rule(Name='saafe-analysis-schedule')
        print(f"  ✅ saafe-analysis-schedule - {response['State']}")
        
        # Check response schedule
        response = events_client.describe_rule(Name='saafe-response-schedule')
        print(f"  ✅ saafe-response-schedule - {response['State']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing CloudWatch events: {e}")
        return False

def test_sns_topic():
    """Test that SNS topic is properly configured."""
    print("\n🔍 Testing SNS topic...")
    
    sns_client = boto3.client('sns', region_name='us-east-1')
    
    try:
        response = sns_client.list_topics()
        topics = [t['TopicArn'] for t in response['Topics']]
        
        expected_topic = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
        
        if expected_topic in topics:
            print(f"  ✅ fire-detection-alerts - Configured")
            return True
        else:
            print(f"  ❌ fire-detection-alerts - Not found")
            return False
        
    except Exception as e:
        print(f"  ❌ Error testing SNS topic: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Production System Test")
    print("=" * 30)
    
    # Run all tests
    tests = [
        ("SageMaker Endpoint", test_sagemaker_endpoint),
        ("Lambda Functions", test_lambda_functions),
        ("CloudWatch Events", test_cloudwatch_events),
        ("SNS Topic", test_sns_topic)
    ]
    
    results = []
    for test_name, test_function in tests:
        try:
            result = test_function()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} - Error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 30)
    print("📋 Test Results")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print("\n" + "=" * 30)
    print(f"📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 System is ready for production use!")
        print("\n📈 Next steps:")
        print("  1. Deploy edge devices with sensors")
        print("  2. Configure data ingestion pipeline")
        print("  3. Monitor system through CloudWatch dashboard")
        print("  4. Set up alert notifications for stakeholders")
        return True
    else:
        print("⚠️  System requires additional configuration before production use.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)