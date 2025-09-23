#!/usr/bin/env python3
"""
Script to automatically set up SNS subscription for fire detection alerts
"""

import boto3
import sys
from botocore.exceptions import ClientError

def setup_email_subscription(email_address):
    """Set up email subscription for fire detection alerts"""
    print("🔧 Auto SNS Subscription Setup")
    print("=" * 35)
    
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
        
        # Check if email is already subscribed
        email_already_subscribed = False
        for sub in current_subscriptions:
            if sub.get('Protocol') == 'email' and sub.get('Endpoint') == email_address:
                email_already_subscribed = True
                print(f"⚠️  Email {email_address} is already subscribed")
                break
        
        if not email_already_subscribed:
            # Subscribe email
            print(f"📧 Subscribing email: {email_address}")
            response = sns_client.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint=email_address
            )
            subscription_arn = response.get('SubscriptionArn')
            print(f"✅ Subscription requested for {email_address}")
            print("💡 Please check your email to confirm the subscription")
            print(f"   Subscription ARN: {subscription_arn}")
            
        # Display final status
        subs_response = sns_client.list_subscriptions_by_topic(TopicArn=topic_arn)
        final_subscriptions = subs_response.get('Subscriptions', [])
        print(f"\n📊 Final subscription count: {len(final_subscriptions)}")
        
        if final_subscriptions:
            print("📋 Current subscriptions:")
            for i, sub in enumerate(final_subscriptions, 1):
                status = "✅ Confirmed" if sub.get('SubscriptionArn') != "PendingConfirmation" else "⏳ Pending"
                print(f"  {i}. {sub.get('Protocol', 'N/A')}: {sub.get('Endpoint', 'N/A')} ({status})")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"❌ AWS Client Error ({error_code}): {error_message}")
        
        if error_code == 'NotFound':
            print("💡 The SNS topic does not exist. You may need to create it first.")
        elif error_code == 'Forbidden':
            print("💡 Access denied. Check your AWS credentials and permissions.")
        elif error_code == 'InvalidParameter':
            print("💡 Invalid parameter. Check the email address format.")
            
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Fire Detection SNS Alert Auto Setup")
    print("=" * 40)
    
    # Check if email address was provided as argument
    if len(sys.argv) < 2:
        print("❌ Usage: python3 auto_setup_sns_subscription.py <email_address>")
        print("💡 Example: python3 auto_setup_sns_subscription.py your-email@example.com")
        sys.exit(1)
    
    email_address = sys.argv[1]
    print(f"📧 Setting up subscription for: {email_address}")
    
    success = setup_email_subscription(email_address)
    
    if success:
        print("\n" + "=" * 40)
        print("✅ SNS Setup Completed Successfully")
        print("💡 Next steps:")
        print("  1. Check your email for confirmation")
        print("  2. Click the confirmation link")
        print("  3. Refresh the dashboard to see updated status")
    else:
        print("\n" + "=" * 40)
        print("❌ SNS Setup Failed")
        sys.exit(1)

if __name__ == "__main__":
    main()