# Email Configuration Guide for Fire Dashboard

## Overview

The fire dashboard backend now includes email alert functionality that automatically sends emails when fire events are detected or predicted. This guide explains how the email system works and how to configure it.

## Email Functionality

### When Emails Are Sent

The system automatically sends email alerts when:
- **Risk Score ≥ 40%**: Fire Predicted alerts
- **Risk Score ≥ 80%**: Fire Detected alerts (urgent)

### Email Cooldown

To prevent email spam, the system has a **5-minute cooldown** between emails. This means:
- Only one email will be sent every 5 minutes
- Subsequent alerts within the cooldown period will be logged but not emailed
- This prevents flooding recipients with duplicate alerts

### Email Content

Emails include:
- **Alert Level**: Fire Detected (red) or Fire Predicted (orange)
- **Risk Score**: Percentage indicating fire probability
- **Confidence Level**: AI model confidence in the prediction
- **Timestamp**: When the alert was triggered
- **Action Required**: Guidance on what to do next

## Configuration

### Default Configuration

The system comes pre-configured with:
```javascript
Email: ch.ajay1707@gmail.com
Password: oznfunikrcfutxxn (App-specific password)
Recipients: ch.ajay1707@gmail.com
```

### Using Environment Variables (Recommended for Production)

For security, you should configure email settings using Vercel environment variables:

1. **Go to Vercel Dashboard**:
   - Navigate to your project: https://vercel.com/saiajay1s-projects/fire-dashboard
   - Click on "Settings" → "Environment Variables"

2. **Add These Variables**:
   ```
   SENDER_EMAIL=your-email@gmail.com
   SENDER_PASSWORD=your-app-specific-password
   RECIPIENT_EMAIL=recipient@example.com
   ```

3. **Redeploy**:
   - After adding environment variables, redeploy your project
   - The new settings will take effect

### Getting a Gmail App-Specific Password

If you want to use your own Gmail account:

1. **Enable 2-Factor Authentication**:
   - Go to Google Account settings
   - Security → 2-Step Verification
   - Enable it if not already enabled

2. **Generate App Password**:
   - Go to: https://myaccount.google.com/apppasswords
   - Select "Mail" and "Other (Custom name)"
   - Name it "Fire Dashboard"
   - Copy the 16-character password
   - Use this as `SENDER_PASSWORD`

3. **Update Environment Variables**:
   - Add the app password to Vercel environment variables
   - Redeploy the application

## Managing Email Recipients

### View Recipients

```bash
GET https://fire-dashboard-xi.vercel.app/api/email-recipients
```

### Add New Recipient

```bash
POST https://fire-dashboard-xi.vercel.app/api/email-recipients
Content-Type: application/json

{
  "email": "newrecipient@example.com",
  "name": "New Recipient",
  "alertLevels": ["all"],
  "enabled": true
}
```

### Update Recipient

```bash
PUT https://fire-dashboard-xi.vercel.app/api/email-recipients/recipient@example.com
Content-Type: application/json

{
  "enabled": false
}
```

### Delete Recipient

```bash
DELETE https://fire-dashboard-xi.vercel.app/api/email-recipients/recipient@example.com
```

## Testing Email Functionality

### Using Fire Data Sender

1. **Open the Data Sender**:
   - Go to: https://fire-data-sender-standalone.vercel.app

2. **Send a Fire Alert**:
   - Click "Fire Predicted" (Risk Score: 60%)
   - Or click "Fire" (Risk Score: 90%)

3. **Check Email**:
   - Wait a few seconds for the email to arrive
   - Check spam folder if not in inbox
   - Verify the email contains correct alert details

### Manual API Test

```bash
curl -X POST https://fire-dashboard-xi.vercel.app/api/send-alert \
  -H "Content-Type: application/json" \
  -d '{
    "riskScore": 85,
    "level": 3,
    "message": "Fire Detected - Test Alert",
    "confidence": 0.95
  }'
```

## Troubleshooting

### Emails Not Being Sent

1. **Check Console Logs**:
   - Go to Vercel Dashboard → Deployments → Click on latest deployment
   - View Function Logs to see email sending status

2. **Verify Environment Variables**:
   - Ensure `SENDER_EMAIL` and `SENDER_PASSWORD` are set correctly
   - Redeploy after changing environment variables

3. **Check Gmail Settings**:
   - Ensure 2FA is enabled
   - Verify app-specific password is correct
   - Check if Gmail is blocking the app

4. **Verify Risk Score**:
   - Emails only sent for risk scores ≥ 40%
   - Check if cooldown period is active (5 minutes)

### Emails Going to Spam

1. **Add Sender to Contacts**:
   - Add the sender email to your contacts
   - This helps Gmail recognize it as legitimate

2. **Mark as Not Spam**:
   - If emails go to spam, mark them as "Not Spam"
   - Gmail will learn to deliver future emails to inbox

3. **Check SPF/DKIM**:
   - Gmail's authentication may flag automated emails
   - Consider using a dedicated email service for production

## Email Service Alternatives

For production use, consider these alternatives to Gmail:

### SendGrid
- More reliable for automated emails
- Better deliverability
- Free tier: 100 emails/day
- Setup: https://sendgrid.com

### AWS SES
- Integrated with AWS services
- Very cost-effective
- Requires domain verification
- Setup: https://aws.amazon.com/ses

### Mailgun
- Developer-friendly API
- Good deliverability
- Free tier: 5,000 emails/month
- Setup: https://www.mailgun.com

## Current Deployment Status

- **Dashboard URL**: https://fire-dashboard-xi.vercel.app
- **API URL**: https://fire-dashboard-xi.vercel.app/api
- **Data Sender**: https://fire-data-sender-standalone.vercel.app
- **Email Status**: ✅ Configured and Active

## Email Flow Diagram

```
Fire Alert Triggered (Risk ≥ 40%)
         ↓
Check Cooldown (5 min)
         ↓
    [Active?] → Yes → Skip Email
         ↓ No
Get Enabled Recipients
         ↓
Format Email (HTML)
         ↓
Send via Gmail SMTP
         ↓
Update Last Email Timestamp
         ↓
Log Result
```

## Security Best Practices

1. **Never Commit Credentials**:
   - Always use environment variables
   - Never hardcode passwords in code

2. **Use App-Specific Passwords**:
   - Don't use your main Gmail password
   - Generate app-specific passwords

3. **Rotate Passwords Regularly**:
   - Change app passwords every 3-6 months
   - Update environment variables after rotation

4. **Monitor Email Logs**:
   - Check Vercel logs regularly
   - Watch for failed email attempts

5. **Limit Recipients**:
   - Only add necessary recipients
   - Remove inactive recipients

## Support

If you encounter issues:
1. Check this guide first
2. Review Vercel function logs
3. Test with the Fire Data Sender
4. Verify Gmail app password is valid
5. Check environment variables are set correctly

---

**Last Updated**: October 21, 2025
**Version**: 1.0.0