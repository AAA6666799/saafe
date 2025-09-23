#!/usr/bin/env python3
"""
Script to verify SNS functionality and send a test message
"""

import boto3
import sys
from botocore.exceptions import ClientError

def verify_sns_and_send_test():
    """Verify SNS functionality and send test message"""
    print("🔍 SNS Functionality Verification")
    print("=" * 35)
    
    # SNS topic information
    topic_arn = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
    
    try:
        # Initialize SNS client
        sns_client = boto3.client('sns', region_name='us-east-1')
        print("✅ SNS client initialized")
        
        # Get topic attributes
        response = sns_client.get_topic_attributes(TopicArn=topic_arn)
        print("✅ Successfully retrieved topic attributes")
        
        # Display topic information
        attributes = response.get('Attributes', {})
        print(f"📋 Topic ARN: {attributes.get('TopicArn', 'N/A')}")
        print(f"📧 Owner: {attributes.get('Owner', 'N/A')}")
        
        # List subscriptions
        subs_response = sns_client.list_subscriptions_by_topic(TopicArn=topic_arn)
        subscriptions = subs_response.get('Subscriptions', [])
        print(f"🔔 Subscriptions: {len(subscriptions)}")
        
        # Send test message if there are subscriptions
        if subscriptions:
            confirmed_subs = [sub for sub in subscriptions if sub.get('SubscriptionArn') != "PendingConfirmation"]
            print(f"✅ Confirmed subscriptions: {len(confirmed_subs)}")
            
            if confirmed_subs:
                print("\n📤 Sending test message...")
                test_message = {
                    "subject": "Fire Detection System - Test Alert",
                    "message": "This is a test message from your fire detection system. The SNS alerting system is working correctly."
                }
                
                response = sns_client.publish(
                    TopicArn=topic_arn,
                    Subject=test_message["subject"],
                    Message=test_message["message"]
                )
                
                print("✅ Test message sent successfully!")
                print(f"   Message ID: {response.get('MessageId')}")
            else:
                print("⚠️  No confirmed subscriptions found. Please confirm your subscription first.")
                print("   Check your email/SMS for confirmation message.")
        else:
            print("⚠️  No subscriptions found. Please set up subscriptions first.")
            print("   Run: python3 auto_setup_sns_subscription.py your-email@example.com")
            
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"❌ AWS Client Error ({error_code}): {error_message}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Fire Detection SNS Verification")
    print("=" * 35)
    
    success = verify_sns_and_send_test()
    
    if success:
        print("\n" + "=" * 35)
        print("✅ SNS Verification Completed")
        if len(sys.argv) > 1 and sys.argv[1] == "--send-test":
            print("💡 Test message sent to all confirmed subscribers")
    else:
        print("\n" + "=" * 35)
        print("❌ SNS Verification Failed")

if __name__ == "__main__":
    main()