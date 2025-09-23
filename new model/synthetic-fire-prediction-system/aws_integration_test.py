#!/usr/bin/env python3
"""
AWS Integration Test
This script tests the integration of all AWS components in the Synthetic Fire Prediction System.
"""

import boto3
import json
import time
from datetime import datetime

def test_sns_topics():
    """Test that all SNS topics exist and are accessible"""
    print("🔍 Testing SNS Topics...")
    
    sns_client = boto3.client('sns', region_name='us-east-1')
    
    topics = [
        'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts',
        'arn:aws:sns:us-east-1:691595239825:fire-detection-analysis-results',
        'arn:aws:sns:us-east-1:691595239825:fire-detection-emergency-response'
    ]
    
    results = {}
    
    for topic_arn in topics:
        try:
            # Try to get topic attributes
            response = sns_client.get_topic_attributes(TopicArn=topic_arn)
            topic_name = topic_arn.split(':')[-1]
            print(f"   ✅ {topic_name}: EXISTS")
            results[topic_name] = True
        except Exception as e:
            topic_name = topic_arn.split(':')[-1]
            print(f"   ❌ {topic_name}: ERROR - {e}")
            results[topic_name] = False
    
    return results

def test_s3_bucket():
    """Test that the S3 bucket exists and is accessible"""
    print("\n🔍 Testing S3 Bucket...")
    
    s3_client = boto3.client('s3', region_name='us-east-1')
    
    bucket_name = 'fire-detection-realtime-data-691595239825'
    
    try:
        # Try to get bucket information
        response = s3_client.head_bucket(Bucket=bucket_name)
        print(f"   ✅ {bucket_name}: EXISTS")
        return True
    except Exception as e:
        print(f"   ❌ {bucket_name}: ERROR - {e}")
        return False

def test_lambda_functions():
    """Test that all Lambda functions exist and are accessible"""
    print("\n🔍 Testing Lambda Functions...")
    
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    functions = [
        'saafe-monitoring-agent',
        'saafe-analysis-agent',
        'saafe-response-agent'
    ]
    
    results = {}
    
    for function_name in functions:
        try:
            # Try to get function configuration
            response = lambda_client.get_function(FunctionName=function_name)
            state = response['Configuration']['State']
            print(f"   ✅ {function_name}: {state}")
            results[function_name] = state == 'Active'
        except Exception as e:
            print(f"   ❌ {function_name}: ERROR - {e}")
            results[function_name] = False
    
    return results

def test_cloudwatch_rules():
    """Test that CloudWatch event rules exist"""
    print("\n🔍 Testing CloudWatch Rules...")
    
    events_client = boto3.client('events', region_name='us-east-1')
    
    rules = [
        'saafe-monitoring-schedule',
        'fire-detection-data-ingestion'
    ]
    
    results = {}
    
    for rule_name in rules:
        try:
            # Try to describe the rule
            response = events_client.describe_rule(Name=rule_name)
            print(f"   ✅ {rule_name}: EXISTS")
            results[rule_name] = True
        except Exception as e:
            print(f"   ❌ {rule_name}: ERROR - {e}")
            results[rule_name] = False
    
    return results

def test_cloudwatch_alarms():
    """Test that CloudWatch alarms exist"""
    print("\n🔍 Testing CloudWatch Alarms...")
    
    cloudwatch_client = boto3.client('cloudwatch', region_name='us-east-1')
    
    alarms = [
        'saafe-monitoring-agent-errors',
        'saafe-analysis-agent-errors',
        'saafe-response-agent-errors'
    ]
    
    results = {}
    
    for alarm_name in alarms:
        try:
            # Try to describe the alarm
            response = cloudwatch_client.describe_alarms(AlarmNames=[alarm_name])
            if response['MetricAlarms']:
                print(f"   ✅ {alarm_name}: EXISTS")
                results[alarm_name] = True
            else:
                print(f"   ❌ {alarm_name}: NOT FOUND")
                results[alarm_name] = False
        except Exception as e:
            print(f"   ❌ {alarm_name}: ERROR - {e}")
            results[alarm_name] = False
    
    return results

def test_sns_publish():
    """Test publishing to SNS topics"""
    print("\n🔍 Testing SNS Publishing...")
    
    sns_client = boto3.client('sns', region_name='us-east-1')
    
    # Test message
    test_message = {
        'test': True,
        'timestamp': datetime.now().isoformat(),
        'component': 'aws_integration_test'
    }
    
    topics = [
        'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
    ]
    
    results = {}
    
    for topic_arn in topics:
        try:
            # Try to publish a test message
            response = sns_client.publish(
                TopicArn=topic_arn,
                Message=json.dumps(test_message),
                Subject='AWS Integration Test Message'
            )
            topic_name = topic_arn.split(':')[-1]
            print(f"   ✅ {topic_name}: MESSAGE PUBLISHED")
            results[topic_name] = True
        except Exception as e:
            topic_name = topic_arn.split(':')[-1]
            print(f"   ❌ {topic_name}: ERROR - {e}")
            results[topic_name] = False
    
    return results

def main():
    """Main function to run all integration tests"""
    print("🔥 Synthetic Fire Prediction System - AWS Integration Test")
    print("=" * 58)
    
    # Run all tests
    sns_results = test_sns_topics()
    s3_result = test_s3_bucket()
    lambda_results = test_lambda_functions()
    rule_results = test_cloudwatch_rules()
    alarm_results = test_cloudwatch_alarms()
    publish_results = test_sns_publish()
    
    # Summary
    print("\n" + "=" * 58)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 58)
    
    # Calculate overall results
    sns_ready = all(sns_results.values())
    lambda_ready = all(lambda_results.values())
    rules_ready = all(rule_results.values())
    alarms_ready = all(alarm_results.values())
    publish_ready = all(publish_results.values())
    
    print(f"SNS Topics: {'✅ READY' if sns_ready else '❌ ISSUE'}")
    print(f"S3 Bucket: {'✅ READY' if s3_result else '❌ ISSUE'}")
    print(f"Lambda Functions: {'✅ READY' if lambda_ready else '❌ ISSUE'}")
    print(f"CloudWatch Rules: {'✅ READY' if rules_ready else '❌ ISSUE'}")
    print(f"CloudWatch Alarms: {'✅ READY' if alarms_ready else '❌ ISSUE'}")
    print(f"SNS Publishing: {'✅ READY' if publish_ready else '❌ ISSUE'}")
    
    overall_status = sns_ready and s3_result and lambda_ready and rules_ready and alarms_ready and publish_ready
    print("\n" + "=" * 58)
    print(f"OVERALL SYSTEM STATUS: {'✅ OPERATIONAL' if overall_status else '❌ ISSUE'}")
    print("=" * 58)
    
    if overall_status:
        print("\n🎉 The Synthetic Fire Prediction System AWS infrastructure is fully operational!")
        print("   All components are configured and ready for production use.")
        print("\n📋 Next steps:")
        print("   1. Deploy SageMaker models for production inference")
        print("   2. Connect IoT sensors when devices are installed")
        print("   3. Monitor the CloudWatch dashboard for system performance")
    else:
        print("\n⚠️  The system has some issues that need to be addressed.")
        print("   Please check the individual component statuses above.")

if __name__ == "__main__":
    main()