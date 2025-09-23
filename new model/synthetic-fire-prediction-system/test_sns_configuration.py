#!/usr/bin/env python3
"""
Test script to verify SNS configuration
"""

import boto3
from botocore.exceptions import ClientError

def test_sns_topic():
    """Test SNS topic configuration"""
    print("🔍 Testing SNS Topic Configuration")
    print("=" * 40)
    
    try:
        # Initialize SNS client
        sns_client = boto3.client('sns', region_name='us-east-1')
        print("✅ SNS client initialized successfully")
        
        # Test topic ARN from the dashboard
        topic_arn = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
        print(f"📡 Testing topic: {topic_arn}")
        
        # Get topic attributes
        response = sns_client.get_topic_attributes(TopicArn=topic_arn)
        print("✅ Successfully retrieved topic attributes")
        
        # Display topic information
        attributes = response.get('Attributes', {})
        print(f"📋 Topic ARN: {attributes.get('TopicArn', 'N/A')}")
        print(f"📝 Display Name: {attributes.get('DisplayName', 'N/A')}")
        print(f"📧 Owner: {attributes.get('Owner', 'N/A')}")
        print(f"🔒 Policy: {attributes.get('Policy', 'N/A')[:100]}..." if attributes.get('Policy') else "🔒 Policy: N/A")
        
        # List subscriptions
        subs_response = sns_client.list_subscriptions_by_topic(TopicArn=topic_arn)
        subscriptions = subs_response.get('Subscriptions', [])
        print(f"🔔 Subscriptions: {len(subscriptions)}")
        
        if subscriptions:
            print("📋 Subscription Details:")
            for i, sub in enumerate(subscriptions, 1):
                print(f"  {i}. Protocol: {sub.get('Protocol', 'N/A')}")
                print(f"     Endpoint: {sub.get('Endpoint', 'N/A')}")
                print(f"     Status: {sub.get('SubscriptionArn', 'N/A')}")
        else:
            print("⚠️  No subscriptions found - you'll need to add subscriptions to receive alerts")
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"❌ AWS Client Error ({error_code}): {error_message}")
        
        if error_code == 'NotFound':
            print("💡 The SNS topic does not exist. You may need to create it.")
        elif error_code == 'Forbidden':
            print("💡 Access denied. Check your AWS credentials and permissions.")
            
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("🚀 SNS Configuration Test")
    print("=" * 40)
    test_sns_topic()
    print("=" * 40)
    print("✅ Test completed")

if __name__ == "__main__":
    main()