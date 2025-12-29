# Gmail Functionality Restored ✅

## Status: Successfully Reverted to Gmail

### What's Been Done:

✅ **Reverted package.json** - Changed from @sendgrid/mail back to nodemailer
✅ **Restored Gmail configuration** - Updated index.js with Gmail SMTP settings
✅ **Installed nodemailer** - Dependencies updated
✅ **Deployed to Vercel** - Live at https://fire-dashboard-2qru2xwbt-saiajay1s-projects.vercel.app
✅ **Environment variables updated** - SENDER_PASSWORD configured

### ⚠️ Action Required: Update Gmail App Password

The current Gmail app password (`oznfunikrcfutxxn`) is no longer valid. You need to generate a new one:

## How to Generate a New Gmail App Password:

### Step 1: Enable 2-Step Verification (if not already enabled)
1. Go to https://myaccount.google.com/security
2. Click on "2-Step Verification"
3. Follow the prompts to enable it

### Step 2: Generate App Password
1. Go to https://myaccount.google.com/apppasswords
2. Sign in with your Gmail account (ch.ajay1707@gmail.com)
3. Select app: **Mail**
4. Select device: **Other (Custom name)**
5. Enter name: **SAAFE Fire Dashboard**
6. Click **Generate**
7. **Copy the 16-character password** (it will look like: `abcd efgh ijkl mnop`)

### Step 3: Update Vercel Environment Variable

```bash
cd "fire-dashboard 21-10-25"

# Remove old password
vercel env rm SENDER_PASSWORD production

# Add new password (replace with your new app password)
echo "your-new-app-password-here" | vercel env add SENDER_PASSWORD production

# Redeploy
vercel --prod
```

### Step 4: Test

```bash
curl -X POST https://fire-dashboard-2qru2xwbt-saiajay1s-projects.vercel.app/api/test-alert-email \
  -H "Content-Type: application/json" \
  -d '{"email": "ch.ajay1707@gmail.com", "riskScore": 85}'
```

Expected response:
```json
{
  "status": "success",
  "message": "Test email sent successfully to ch.ajay1707@gmail.com"
}
```

## Current Configuration:

- **Email Service**: Gmail (via nodemailer)
- **Sender Email**: ch.ajay1707@gmail.com
- **Recipient Email**: ch.ajay1707@gmail.com
- **Deployment**: https://fire-dashboard-2qru2xwbt-saiajay1s-projects.vercel.app

## Files Modified:

1. [`api/package.json`](api/package.json) - Reverted to nodemailer
2. [`api/index.js`](api/index.js) - Restored Gmail SMTP configuration

## Notes:

- Gmail has cooldown periods (you mentioned this was an issue)
- The app password needs to be regenerated periodically for security
- Make sure 2-Step Verification is enabled on your Gmail account
- App passwords only work with 2-Step Verification enabled

## Alternative:

If Gmail cooldowns continue to be an issue, consider:
1. Using a different Gmail account for sending
2. Upgrading to G Suite/Google Workspace (higher limits)
3. Using SendGrid (which we tried, but had Vercel issues)
4. Using another email service like AWS SES or Mailgun

---

**Date**: October 23, 2025  
**Status**: ✅ Reverted to Gmail (App password needs update)