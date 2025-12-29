# SendGrid Email Configuration Guide

This guide will help you set up SendGrid for email notifications in the SAAFE Fire Detection Dashboard.

## Why SendGrid?

SendGrid is a professional email delivery service that offers:
- **No cooldown periods** - Send emails without Gmail's rate limiting
- **Higher reliability** - 99.9% uptime SLA
- **Better deliverability** - Dedicated IP addresses and domain authentication
- **Scalability** - Send thousands of emails per day
- **Free tier** - 100 emails/day free forever
- **Analytics** - Track email opens, clicks, and bounces

## Prerequisites

1. A SendGrid account (free or paid)
2. A verified sender email address
3. Access to your Vercel project settings

## Step 1: Create a SendGrid Account

1. Go to [SendGrid](https://sendgrid.com/)
2. Click "Start for Free" or "Sign Up"
3. Complete the registration process
4. Verify your email address

## Step 2: Verify Your Sender Email

SendGrid requires you to verify the email address you'll send from:

1. Log in to your SendGrid dashboard
2. Go to **Settings** → **Sender Authentication**
3. Click **Verify a Single Sender**
4. Fill in the form with your details:
   - **From Name**: SAAFE AI Alert System (or your preferred name)
   - **From Email Address**: Your email (e.g., alerts@yourdomain.com)
   - **Reply To**: Same as From Email or a support email
   - **Company Address**: Your company details
5. Click **Create**
6. Check your email and click the verification link

**Note**: For production use, consider setting up Domain Authentication for better deliverability.

## Step 3: Create an API Key

1. In your SendGrid dashboard, go to **Settings** → **API Keys**
2. Click **Create API Key**
3. Choose a name (e.g., "SAAFE Fire Dashboard")
4. Select **Full Access** or **Restricted Access** with:
   - Mail Send: **Full Access**
5. Click **Create & View**
6. **IMPORTANT**: Copy the API key immediately - you won't be able to see it again!
   - It will look like: `SG.xxxxxxxxxxxxxxxxxx.yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy`

## Step 4: Configure Environment Variables

### For Vercel Deployment:

1. Go to your Vercel project dashboard
2. Navigate to **Settings** → **Environment Variables**
3. Add the following variables:

| Variable Name | Value | Description |
|--------------|-------|-------------|
| `SENDGRID_API_KEY` | Your SendGrid API key | Required for sending emails |
| `SENDER_EMAIL` | Your verified email | The email address to send from |
| `SENDER_NAME` | SAAFE AI Alert System | The name that appears in emails |
| `RECIPIENT_EMAIL` | admin@example.com | Default recipient email |

4. Click **Save** for each variable
5. Redeploy your application for changes to take effect

### For Local Development:

Create a `.env` file in the `fire-dashboard 21-10-25/api/` directory:

```env
SENDGRID_API_KEY=SG.your_api_key_here
SENDER_EMAIL=your-verified-email@example.com
SENDER_NAME=SAAFE AI Alert System
RECIPIENT_EMAIL=admin@example.com
```

**IMPORTANT**: Never commit the `.env` file to version control!

## Step 5: Install Dependencies

Navigate to your API directory and install the SendGrid package:

```bash
cd "fire-dashboard 21-10-25/api"
npm install
```

This will install `@sendgrid/mail` as specified in `package.json`.

## Step 6: Test the Configuration

### Method 1: Using the Dashboard

1. Deploy your application to Vercel
2. Open the dashboard in your browser
3. Navigate to the Email Recipients section
4. Click "Test Email" button
5. Check your inbox for the test email

### Method 2: Using the API Directly

Send a POST request to test the email:

```bash
curl -X POST https://your-app.vercel.app/api/test-alert-email \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "riskScore": 85
  }'
```

### Method 3: Check Health Endpoint

Verify SendGrid is configured:

```bash
curl https://your-app.vercel.app/api/health
```

Expected response:
```json
{
  "status": "ok",
  "timestamp": "2025-10-22T21:00:00.000Z",
  "sendgridAvailable": true,
  "sendgridConfigured": true
}
```

## Email Features

### Automatic Alerts

The system automatically sends emails when:
- **Risk Score ≥ 40**: Fire predicted (warning email)
- **Risk Score ≥ 80**: Fire detected (urgent email)

### Cooldown Period

To prevent email spam, there's a 5-minute cooldown between alerts. This can be adjusted in `index.js`:

```javascript
const EMAIL_COOLDOWN = 5 * 60 * 1000; // 5 minutes in milliseconds
```

### Multiple Recipients

You can add multiple email recipients through the dashboard or API:

```bash
curl -X POST https://your-app.vercel.app/api/email-recipients \
  -H "Content-Type: application/json" \
  -d '{
    "email": "recipient@example.com",
    "name": "John Doe",
    "alertLevels": ["all"],
    "enabled": true
  }'
```

## Troubleshooting

### Issue: "SendGrid not configured" error

**Solution**: Ensure `SENDGRID_API_KEY` environment variable is set in Vercel and redeploy.

### Issue: Emails not being received

**Possible causes**:
1. **Sender not verified**: Verify your sender email in SendGrid
2. **API key invalid**: Generate a new API key
3. **Spam folder**: Check recipient's spam/junk folder
4. **Risk score too low**: Emails only sent for risk scores ≥ 40
5. **Cooldown active**: Wait 5 minutes between alerts

### Issue: "403 Forbidden" error

**Solution**: Your API key may not have Mail Send permissions. Create a new API key with Full Access or Mail Send permissions.

### Issue: Emails going to spam

**Solutions**:
1. Set up Domain Authentication in SendGrid
2. Add SPF and DKIM records to your domain
3. Use a professional email address (not Gmail/Yahoo)
4. Warm up your sending reputation gradually

## SendGrid Dashboard Features

Monitor your email delivery:

1. **Activity Feed**: See all sent emails and their status
2. **Statistics**: Track delivery rates, opens, and clicks
3. **Suppressions**: Manage bounced and unsubscribed emails
4. **Templates**: Create reusable email templates (optional)

## Rate Limits

### Free Plan
- 100 emails/day
- Perfect for small deployments and testing

### Paid Plans
- Essentials: 50,000 emails/month starting at $19.95
- Pro: 100,000+ emails/month with advanced features

## Security Best Practices

1. **Never expose your API key** in client-side code
2. **Use environment variables** for all sensitive data
3. **Rotate API keys** periodically
4. **Use restricted API keys** with minimal permissions
5. **Enable two-factor authentication** on your SendGrid account
6. **Monitor your SendGrid activity** for suspicious behavior

## Migration from Gmail

The migration is complete! Key differences:

| Feature | Gmail (Old) | SendGrid (New) |
|---------|-------------|----------------|
| Rate Limit | ~500/day with cooldowns | 100/day (free) or unlimited (paid) |
| Reliability | Consumer-grade | Enterprise-grade |
| Deliverability | Good | Excellent |
| Analytics | None | Comprehensive |
| Setup | App password | API key |
| Cost | Free | Free tier available |

## Support

- **SendGrid Documentation**: https://docs.sendgrid.com/
- **SendGrid Support**: https://support.sendgrid.com/
- **API Reference**: https://docs.sendgrid.com/api-reference/mail-send/mail-send

## Next Steps

1. ✅ Set up SendGrid account
2. ✅ Verify sender email
3. ✅ Create API key
4. ✅ Configure environment variables
5. ✅ Test email delivery
6. 🎯 Monitor email analytics
7. 🎯 Consider domain authentication for production

---

**Last Updated**: October 22, 2025
**Version**: 1.0.0