# 📢 SNS Alerting System - Operations Guide

## Overview
This guide provides comprehensive instructions for configuring, managing, and troubleshooting the SNS alerting system for the fire detection system.

## System Architecture
The fire detection system uses AWS SNS for alerting:
- **Topic ARN**: `arn:aws:sns:us-east-1:691595239825:fire-detection-alerts`
- **Region**: `us-east-1`
- **Integration**: Dashboard displays subscription status and count
- **Trigger**: Fire detection events from the analysis system

## Configuration Steps

### 1. Setting Up Subscriptions

#### Using the Auto Setup Script
```bash
cd "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system"
python3 auto_setup_sns_subscription.py your-email@example.com
```

#### Manual Setup via AWS Console
1. Navigate to [AWS SNS Console](https://console.aws.amazon.com/sns/v3/home?region=us-east-1#/topics)
2. Select the topic: `arn:aws:sns:us-east-1:691595239825:fire-detection-alerts`
3. Click "Create subscription"
4. Choose protocol (Email or SMS)
5. Enter endpoint (email address or phone number)
6. Click "Create subscription"

#### Programmatic Setup
```python
import boto3

sns_client = boto3.client('sns', region_name='us-east-1')
response = sns_client.subscribe(
    TopicArn='arn:aws:sns:us-east-1:691595239825:fire-detection-alerts',
    Protocol='email',  # or 'sms'
    Endpoint='your-email@example.com'  # or phone number
)
```

### 2. Confirming Subscriptions
After creating a subscription:
1. Check email/SMS for confirmation message from AWS
2. Click the confirmation link
3. Subscription status changes from "Pending" to "Confirmed"
4. Refresh the dashboard to see updated subscription count

### 3. Verifying Functionality
```bash
cd "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system"
python3 verify_sns_functionality.py
```

To send a test message:
```bash
python3 verify_sns_functionality.py --send-test
```

## Dashboard Integration
The Streamlit dashboard shows:
- ✅ SNS service status
- 📋 Topic ARN
- 🔔 Subscription count
- ⚠️ Warning when no subscriptions are configured

## Alert Message Format
Fire detection alerts follow this format:
```
🚨 FIRE DETECTION ALERT
Timestamp: 2025-09-09 14:30:25 UTC
Risk Level: EMERGENCY (0.85)
Location: Device ID first-device
Action Required: Immediate investigation needed
```

## Best Practices

### Subscription Management
1. **Multiple Subscribers**: Set up multiple subscribers for redundancy
2. **Confirmation**: Always confirm subscriptions promptly
3. **Monitoring**: Regularly check subscription status in the dashboard
4. **Security**: Use appropriate email/SMS addresses for alerts

### Email Configuration
1. **Whitelist**: Add AWS notifications to email whitelist
2. **Spam Filter**: Check spam/junk folders for alerts
3. **Availability**: Ensure email addresses are monitored during critical hours

### SMS Configuration
1. **Coverage**: Verify phone coverage in deployment areas
2. **International**: Include country codes for international numbers
3. **Format**: Use E.164 format (+1234567890)

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: No Alerts Received
**Symptoms**: Dashboard shows subscriptions but no alerts received
**Solutions**:
1. Check subscription status (must be "Confirmed")
2. Verify email/SMS address is correct
3. Check spam/junk folders
4. Whitelist AWS notification addresses

#### Issue: Subscription Not Confirmed
**Symptoms**: Dashboard shows "Pending" subscriptions
**Solutions**:
1. Check email/SMS for confirmation message
2. Resend confirmation if needed
3. Ensure the endpoint is accessible

#### Issue: AWS Permissions Error
**Symptoms**: Setup scripts fail with permission errors
**Solutions**:
1. Verify AWS credentials have SNS permissions
2. Check IAM policies for SNS access
3. Ensure proper AWS CLI configuration

#### Issue: Invalid Parameter Error
**Symptoms**: Setup fails with invalid parameter message
**Solutions**:
1. Check email address format (must be valid)
2. Check phone number format (E.164 format required)
3. Verify no special characters in addresses

## Maintenance Procedures

### Regular Checks
1. **Weekly**: Verify subscription status in dashboard
2. **Monthly**: Test alert functionality with test message
3. **Quarterly**: Review and update subscriber list

### Monitoring
1. **Dashboard**: Check SNS status indicator
2. **AWS Console**: Monitor subscription health
3. **Logs**: Review SNS delivery logs for failures

### Updates
1. **Adding Subscribers**: Use setup scripts or AWS Console
2. **Removing Subscribers**: Unsubscribe via AWS Console
3. **Changing Protocols**: Create new subscription with different protocol

## Security Considerations

### Access Control
1. **IAM Policies**: Restrict SNS access to authorized users
2. **Credentials**: Use secure credential management
3. **Encryption**: Enable encryption for sensitive data

### Privacy
1. **Data Handling**: Follow privacy regulations for alert data
2. **Consent**: Ensure subscribers have consented to alerts
3. **Retention**: Implement appropriate data retention policies

## Integration Testing

### Test Procedures
1. **Subscription Setup**: Verify new subscriptions can be created
2. **Message Delivery**: Confirm test messages are delivered
3. **Dashboard Update**: Ensure dashboard reflects subscription changes
4. **Error Handling**: Test error conditions and recovery

### Test Scenarios
1. **Normal Operation**: Successful alert delivery
2. **No Subscriptions**: Dashboard warning display
3. **Failed Delivery**: Error handling and logging
4. **Multiple Subscribers**: Delivery to all confirmed subscribers

## Recovery Procedures

### Failed Subscriptions
1. Remove failed subscriptions
2. Recreate with correct parameters
3. Confirm new subscriptions
4. Verify dashboard update

### Service Disruption
1. Check AWS service health dashboard
2. Verify IAM permissions
3. Test with simple publish operation
4. Contact AWS support if needed

## Contact Information

### AWS Support
- **Console**: https://console.aws.amazon.com/support
- **Service Health**: https://status.aws.amazon.com

### System Administrator
- **Email**: [System Administrator Email]
- **Phone**: [System Administrator Phone]

## Revision History
- **Version 1.0**: Initial release
- **Date**: September 9, 2025
- **Author**: Fire Detection System Team