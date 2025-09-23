#!/usr/bin/env python3
"""
Script to set up SNS subscriptions for fire detection alerts
"""

import boto3
from botocore.exceptions import ClientError

def setup_sns_subscription():
    """Set up SNS subscription for fire detection alerts"""
    print("🔧 SNS Subscription Setup")
    print("=" * 30)
    
    # SNS topic information
    topic_arn = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
    
    try:
        # Initialize SNS client
        sns_client = boto3.client('sns', region_name='us-east-1')
        print("✅ SNS client initialized")
        
        # Get current subscriptions
        subs_response = sns_client.list_subscriptions_by_topic(TopicArn=topic_arn)
        current_subscriptions = subs_response.get('Subscriptions', [])
        
        print(f"📊 Current subscriptions: {len(current_subscriptions)}")
        
        if current_subscriptions:
            print("📋 Current subscriptions:")
            for i, sub in enumerate(current_subscriptions, 1):
                print(f"  {i}. {sub.get('Protocol', 'N/A')}: {sub.get('Endpoint', 'N/A')}")
        
        # Ask user what type of subscription they want to add
        print("\n📝 Subscription Options:")
        print("1. Email")
        print("2. SMS")
        print("3. Exit")
        
        choice = input("\nSelect subscription type (1-3): ").strip()
        
        if choice == '1':
            email = input("Enter email address: ").strip()
            if email:
                response = sns_client.subscribe(
                    TopicArn=topic_arn,
                    Protocol='email',
                    Endpoint=email
                )
                subscription_arn = response.get('SubscriptionArn')
                print(f"✅ Email subscription requested for {email}")
                print("💡 Please check your email to confirm the subscription")
                print(f"   Subscription ARN: {subscription_arn}")
            else:
                print("❌ Invalid email address")
                
        elif choice == '2':
            phone = input("Enter phone number (with country code, e.g., +1234567890): ").strip()
            if phone:
                response = sns_client.subscribe(
                    TopicArn=topic_arn,
                    Protocol='sms',
                    Endpoint=phone
                )
                subscription_arn = response.get('SubscriptionArn')
                print(f"✅ SMS subscription requested for {phone}")
                print("💡 Please check your phone to confirm the subscription")
                print(f"   Subscription ARN: {subscription_arn}")
            else:
                print("❌ Invalid phone number")
                
        elif choice == '3':
            print("👋 Exiting subscription setup")
            return
            
        else:
            print("❌ Invalid choice")
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"❌ AWS Client Error ({error_code}): {error_message}")
        
        if error_code == 'NotFound':
            print("💡 The SNS topic does not exist. You may need to create it first.")
        elif error_code == 'Forbidden':
            print("💡 Access denied. Check your AWS credentials and permissions.")
        elif error_code == 'InvalidParameter':
            print("💡 Invalid parameter. Check the email or phone number format.")
            
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

def display_sns_info():
    """Display current SNS configuration"""
    print("📊 SNS Configuration Information")
    print("=" * 35)
    
    topic_arn = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
    print(f"📡 Topic ARN: {topic_arn}")
    print(f"📍 Region: us-east-1")
    print(f"🆔 Account ID: 691595239825")
    print()
    print("📋 To receive alerts, you need to:")
    print("  1. Subscribe to the topic (using this script or AWS Console)")
    print("  2. Confirm the subscription (via email or SMS)")
    print("  3. Ensure the fire detection system is properly configured")
    print()
    print("🔗 AWS SNS Console URL:")
    print("  https://console.aws.amazon.com/sns/v3/home?region=us-east-1#/topics")
    print()

def main():
    """Main function"""
    print("🚀 Fire Detection SNS Alert Setup")
    print("=" * 35)
    
    display_sns_info()
    setup_sns_subscription()
    
    print("\n" + "=" * 35)
    print("✅ SNS Setup Completed")

if __name__ == "__main__":
    main()