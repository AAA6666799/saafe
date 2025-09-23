#!/usr/bin/env python3
"""
Configure SNS subscriptions for fire detection alerts
"""

import boto3
import json

def configure_sns_subscriptions():
    """Configure SNS subscriptions for alerting."""
    print("Configuring SNS subscriptions for fire detection alerts...")
    
    # Initialize SNS client
    sns_client = boto3.client('sns', region_name='us-east-1')
    
    # SNS topic ARN
    topic_arn = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
    
    # For demonstration, we'll show how to add subscriptions
    # In a real environment, you would add actual endpoints
    
    try:
        # List existing subscriptions
        print("\n1. Checking existing subscriptions...")
        response = sns_client.list_subscriptions_by_topic(TopicArn=topic_arn)
        subscriptions = response.get('Subscriptions', [])
        
        print(f"   Found {len(subscriptions)} existing subscriptions:")
        for sub in subscriptions:
            print(f"   - {sub['Protocol']}: {sub['Endpoint']}")
        
        # Example of how to add a subscription (commented out for safety)
        # Uncomment and modify as needed for your environment
        """
        print("\n2. Adding email subscription...")
        response = sns_client.subscribe(
            TopicArn=topic_arn,
            Protocol='email',
            Endpoint='fire-team@example.com'
        )
        print(f"   Subscription ARN: {response.get('SubscriptionArn')}")
        
        print("\n3. Adding SMS subscription...")
        response = sns_client.subscribe(
            TopicArn=topic_arn,
            Protocol='sms',
            Endpoint='+1234567890'
        )
        print(f"   Subscription ARN: {response.get('SubscriptionArn')}")
        """
        
        print("\n✅ SNS subscription configuration check completed")
        print("\nTo add actual subscriptions, uncomment the relevant sections")
        print("and modify the endpoints to your actual email addresses or phone numbers.")
        
    except Exception as e:
        print(f"❌ Error configuring SNS subscriptions: {e}")

def show_sns_topic_info():
    """Show information about the SNS topic."""
    print("\nSNS Topic Information:")
    print("=" * 30)
    
    # Initialize SNS client
    sns_client = boto3.client('sns', region_name='us-east-1')
    
    # SNS topic ARN
    topic_arn = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
    
    try:
        # Get topic attributes
        response = sns_client.get_topic_attributes(TopicArn=topic_arn)
        attributes = response.get('Attributes', {})
        
        print(f"Topic ARN: {attributes.get('TopicArn', 'N/A')}")
        print(f"Display Name: {attributes.get('DisplayName', 'N/A')}")
        print(f"Owner: {attributes.get('Owner', 'N/A')}")
        print(f"Subscriptions Confirmed: {attributes.get('SubscriptionsConfirmed', '0')}")
        print(f"Subscriptions Pending: {attributes.get('SubscriptionsPending', '0')}")
        print(f"Subscriptions Deleted: {attributes.get('SubscriptionsDeleted', '0')}")
        
    except Exception as e:
        print(f"❌ Error getting topic information: {e}")

if __name__ == "__main__":
    configure_sns_subscriptions()
    show_sns_topic_info()
