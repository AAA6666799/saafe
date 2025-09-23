# 📢 SNS Alerting System Configuration

## Overview
The fire detection system uses AWS SNS (Simple Notification Service) for alerting. The system is properly configured with a topic, but currently has no subscribers, which means no one will receive alerts.

## Current Configuration

### Topic Information
- **Topic ARN**: `arn:aws:sns:us-east-1:691595239825:fire-detection-alerts`
- **Region**: `us-east-1`
- **Account ID**: `691595239825`
- **Status**: ✅ ACTIVE

### Subscription Status
- **Current Subscriptions**: 0
- **Alert Delivery**: ❌ NOT CONFIGURED

## Setting Up Subscriptions

### Option 1: Interactive Setup Script
Run the provided setup script:
```bash
cd "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system"
python3 setup_sns_subscriptions.py
```

This script will guide you through:
1. Choosing subscription type (Email or SMS)
2. Entering your contact information
3. Sending subscription request
4. Providing confirmation instructions

### Option 2: AWS Console
1. Navigate to [AWS SNS Console](https://console.aws.amazon.com/sns/v3/home?region=us-east-1#/topics)
2. Select the topic: `arn:aws:sns:us-east-1:691595239825:fire-detection-alerts`
3. Click "Create subscription"
4. Choose protocol (Email or SMS)
5. Enter endpoint (email address or phone number)
6. Click "Create subscription"

### Option 3: Programmatic Setup
```python
import boto3

# Initialize SNS client
sns_client = boto3.client('sns', region_name='us-east-1')

# Subscribe email
response = sns_client.subscribe(
    TopicArn='arn:aws:sns:us-east-1:691595239825:fire-detection-alerts',
    Protocol='email',
    Endpoint='your-email@example.com'
)

# Subscribe SMS
response = sns_client.subscribe(
    TopicArn='arn:aws:sns:us-east-1:691595239825:fire-detection-alerts',
    Protocol='sms',
    Endpoint='+1234567890'  # Include country code
)
```

## Subscription Confirmation
After creating a subscription:
1. **Email**: Check your inbox for a confirmation email from AWS Notifications
2. **SMS**: Check your phone for a confirmation SMS from AWS
3. Click the confirmation link or follow the instructions
4. The subscription will show as "Confirmed" in the AWS Console

## Testing Alerts
Once subscriptions are confirmed:
1. The dashboard will show the updated subscription count
2. You can trigger a test alert through the system
3. Alerts will be sent for fire detection events

## Best Practices
1. **Multiple Subscriptions**: Set up multiple subscribers for redundancy
2. **Confirmation**: Always confirm subscriptions promptly
3. **Monitoring**: Regularly check subscription status in the dashboard
4. **Security**: Use appropriate email/SMS addresses for alerts

## Troubleshooting
### Common Issues:
1. **No alerts received**: 
   - Check subscription status (must be "Confirmed")
   - Verify email/SMS address is correct
   - Check spam/junk folders

2. **Subscription not confirmed**:
   - Check email/SMS for confirmation message
   - Resend confirmation if needed
   - Ensure the endpoint is accessible

3. **AWS permissions**:
   - Verify AWS credentials have SNS permissions
   - Check IAM policies for SNS access

## Dashboard Integration
The Streamlit dashboard shows:
- ✅ SNS service status
- 📋 Topic ARN
- 🔔 Subscription count
- ⚠️ Warning when no subscriptions are configured

When subscriptions are added, the dashboard will automatically reflect the updated count on refresh.