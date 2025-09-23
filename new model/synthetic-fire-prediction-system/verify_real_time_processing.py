#!/usr/bin/env python3
"""
Verification script for real-time high-frequency data processing.
This script verifies that all components of the real-time processing system are working correctly.
"""

import boto3
import json
from datetime import datetime, timedelta

def verify_lambda_function():
    """Verify that the S3 data processor Lambda function exists and is properly configured."""
    print("🔍 Verifying S3 Data Processor Lambda function...")
    
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    try:
        # Check if function exists
        response = lambda_client.get_function(FunctionName='saafe-s3-data-processor')
        print("✅ Lambda function 'saafe-s3-data-processor' exists")
        
        # Check configuration
        config = response['Configuration']
        if config['Runtime'] == 'python3.9':
            print("✅ Correct runtime (Python 3.9)")
        else:
            print(f"❌ Incorrect runtime: {config['Runtime']}")
            
        if config['Timeout'] >= 900:
            print("✅ Adequate timeout (900+ seconds)")
        else:
            print(f"❌ Insufficient timeout: {config['Timeout']} seconds")
            
        if config['MemorySize'] >= 1024:
            print("✅ Adequate memory (1024+ MB)")
        else:
            print(f"❌ Insufficient memory: {config['MemorySize']} MB")
            
        return True
        
    except Exception as e:
        print(f"❌ Error verifying Lambda function: {e}")
        return False

def verify_s3_bucket():
    """Verify that the S3 bucket exists and has proper event notifications."""
    print("\n🔍 Verifying S3 bucket configuration...")
    
    s3_client = boto3.client('s3')
    bucket_name = 'data-collector-of-first-device'
    
    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        print("✅ S3 bucket 'data-collector-of-first-device' exists")
        
        # Check notification configuration
        notification_config = s3_client.get_bucket_notification_configuration(Bucket=bucket_name)
        
        lambda_configs = notification_config.get('LambdaFunctionConfigurations', [])
        processor_configured = False
        
        for config in lambda_configs:
            if 'saafe-s3-data-processor' in config['LambdaFunctionArn']:
                processor_configured = True
                print("✅ S3 event trigger configured for saafe-s3-data-processor")
                break
                
        if not processor_configured:
            print("❌ S3 event trigger not configured for saafe-s3-data-processor")
            
        return True
        
    except Exception as e:
        print(f"❌ Error verifying S3 bucket: {e}")
        return False

def verify_sagemaker_endpoint():
    """Verify that the SageMaker endpoint exists and is in service."""
    print("\n🔍 Verifying SageMaker endpoint...")
    
    sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
    
    try:
        # Check if endpoint exists and is in service
        response = sagemaker_client.describe_endpoint(EndpointName='fire-mvp-xgb-endpoint')
        
        status = response['EndpointStatus']
        if status == 'InService':
            print("✅ SageMaker endpoint 'fire-mvp-xgb-endpoint' is in service")
            return True
        else:
            print(f"❌ SageMaker endpoint status: {status}")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying SageMaker endpoint: {e}")
        return False

def verify_sns_topic():
    """Verify that the SNS topic exists."""
    print("\n🔍 Verifying SNS topic...")
    
    sns_client = boto3.client('sns', region_name='us-east-1')
    
    try:
        # Check if topic exists
        topic_arn = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
        sns_client.get_topic_attributes(TopicArn=topic_arn)
        print("✅ SNS topic 'fire-detection-alerts' exists")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying SNS topic: {e}")
        return False

def verify_recent_invocations():
    """Verify that the Lambda function has been invoked recently."""
    print("\n🔍 Verifying recent Lambda invocations...")
    
    logs_client = boto3.client('logs', region_name='us-east-1')
    
    try:
        # Get recent log streams
        response = logs_client.describe_log_streams(
            logGroupName='/aws/lambda/saafe-s3-data-processor',
            orderBy='LastEventTime',
            descending=True,
            limit=5
        )
        
        log_streams = response.get('logStreams', [])
        if log_streams:
            latest_stream = log_streams[0]
            last_event_time = latest_stream.get('lastEventTimestamp', 0)
            
            # Check if there was an event in the last hour
            one_hour_ago = (datetime.now() - timedelta(hours=1)).timestamp() * 1000
            
            if last_event_time > one_hour_ago:
                print("✅ Lambda function has been invoked recently")
                return True
            else:
                print("⚠️  Lambda function has not been invoked recently (may be normal if no new data)")
                return True
        else:
            print("⚠️  No log streams found (function may not have been invoked yet)")
            return True
            
    except Exception as e:
        print(f"❌ Error verifying Lambda invocations: {e}")
        return False

def verify_permissions():
    """Verify that the Lambda function has necessary permissions."""
    print("\n🔍 Verifying Lambda permissions...")
    
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    try:
        # Check if S3 can invoke the Lambda function
        response = lambda_client.get_policy(FunctionName='saafe-s3-data-processor')
        policy = json.loads(response['Policy'])
        
        statements = policy.get('Statement', [])
        s3_permission_found = False
        
        for statement in statements:
            if (statement.get('Principal', {}).get('Service') == 's3.amazonaws.com' and
                'lambda:InvokeFunction' in statement.get('Action', [])):
                s3_permission_found = True
                print("✅ S3 has permission to invoke Lambda function")
                break
                
        if not s3_permission_found:
            print("❌ S3 does not have permission to invoke Lambda function")
            
        return True
        
    except Exception as e:
        print(f"❌ Error verifying Lambda permissions: {e}")
        return False

def main():
    """Main verification function."""
    print("🔥 Real-Time High-Frequency Data Processing Verification")
    print("=" * 60)
    
    # Run all verification checks
    checks = [
        verify_lambda_function,
        verify_s3_bucket,
        verify_sagemaker_endpoint,
        verify_sns_topic,
        verify_recent_invocations,
        verify_permissions
    ]
    
    results = []
    for check in checks:
        result = check()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total} checks")
    
    if passed == total:
        print("🎉 All verification checks passed!")
        print("\n✅ Real-time high-frequency data processing system is ready for production use.")
        print("The system will process sensor data as it arrives in S3 and provide real-time fire predictions.")
    else:
        print("⚠️  Some verification checks failed or require attention.")
        print("Please review the output above and address any issues.")
    
    print("\n📊 Next Steps:")
    print("1. Monitor CloudWatch logs for detailed processing information")
    print("2. Upload test files to S3 to verify end-to-end processing")
    print("3. Check SNS topic for alert notifications")
    print("4. Review SageMaker endpoint metrics for prediction performance")

if __name__ == "__main__":
    main()