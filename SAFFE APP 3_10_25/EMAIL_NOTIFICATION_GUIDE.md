# Email Notification System Guide

## Overview

The SAAFE Fire Detection System now includes a comprehensive email notification system that automatically sends alerts to multiple recipients when fire risks are detected. The system supports three alert levels with customizable recipient lists and intelligent spam prevention.

## Features

### 1. **Multi-Recipient Support**
- Add unlimited email recipients
- Each recipient can have a custom name
- Enable/disable recipients without removing them
- Manage recipients via REST API

### 2. **Alert Level Filtering**
- **Urgent (Risk Score ≥ 80)**: Critical fire detection - immediate evacuation required
- **Warning (Risk Score ≥ 40)**: Elevated fire risk - investigation required
- **Caution (Risk Score ≥ 20)**: Increased fire risk - monitoring recommended
- Recipients can subscribe to specific alert levels or all alerts

### 3. **Professional Email Templates**
- Beautiful HTML email templates with responsive design
- Color-coded by severity (Red for Urgent, Orange for Warning, Yellow for Caution)
- Includes detailed alert information (risk score, confidence, location, device ID)
- Actionable recommendations based on alert level

### 4. **Intelligent Spam Prevention**
- 5-minute cooldown between emails to prevent flooding
- Only sends new emails if risk score increases by 20+ points during cooldown
- Tracks last email sent timestamp and risk score

### 5. **Automatic Integration**
- Emails automatically triggered when alert state is updated via [`/api/alert-state`](backend/server.js:1207)
- Only sends when `isActive: true` and `riskScore >= 40`
- Returns email notification results in API response

## Configuration

### Environment Variables

Set these in your environment or `.env` file:

```bash
SENDER_EMAIL=your-gmail@gmail.com
SENDER_PASSWORD=your-app-specific-password
RECIPIENT_EMAIL=default-recipient@example.com  # Optional: default recipient
```

### Gmail App Password Setup

1. Go to your Google Account settings
2. Navigate to Security → 2-Step Verification
3. Scroll to "App passwords"
4. Generate a new app password for "Mail"
5. Use this password as `SENDER_PASSWORD`

## API Endpoints

### 1. Get Email Recipients
```http
GET /api/email-recipients
```

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "email": "admin@example.com",
      "name": "Primary Admin",
      "alertLevels": ["all"],
      "enabled": true
    }
  ],
  "count": 1
}
```

### 2. Add Email Recipient
```http
POST /api/email-recipients
Content-Type: application/json

{
  "email": "user@example.com",
  "name": "John Doe",
  "alertLevels": ["urgent", "warning"],  // or ["all"]
  "enabled": true
}
```

**Alert Levels:**
- `"all"` - Receives all alerts (recommended for admins)
- `"urgent"` - Only critical fire detection (risk ≥ 80)
- `"warning"` - Elevated fire risk (risk ≥ 40)
- `"caution"` - Increased fire risk (risk ≥ 20)

**Response:**
```json
{
  "status": "success",
  "message": "Email recipient added successfully",
  "data": {
    "email": "user@example.com",
    "name": "John Doe",
    "alertLevels": ["urgent", "warning"],
    "enabled": true
  },
  "totalRecipients": 2
}
```

### 3. Update Email Recipient
```http
PUT /api/email-recipients/:email
Content-Type: application/json

{
  "name": "Jane Doe",
  "alertLevels": ["all"],
  "enabled": false
}
```

### 4. Delete Email Recipient
```http
DELETE /api/email-recipients/:email
```

### 5. Send Test Alert Email
```http
POST /api/test-alert-email
Content-Type: application/json

{
  "email": "test@example.com",
  "riskScore": 85  // Optional: defaults to 85 (urgent level)
}
```

This sends a test email with `[TEST]` prefix to verify email configuration.

### 6. Update Alert State (Triggers Emails)
```http
POST /api/alert-state
Content-Type: application/json

{
  "isActive": true,
  "level": 4,
  "message": "Fire detected in kitchen",
  "riskScore": 85,
  "confidence": 0.95,
  "location": "Kitchen",
  "deviceId": "SAAFE-KITCHEN-001"
}
```

**Response includes email notification results:**
```json
{
  "status": "success",
  "message": "Alert state updated successfully",
  "data": {
    "isActive": true,
    "level": 4,
    "message": "Fire detected in kitchen",
    "riskScore": 85,
    "confidence": 0.95,
    "timestamp": "2025-01-04T22:30:00.000Z"
  },
  "emailNotifications": {
    "sent": 3,
    "failed": 0,
    "skipped": 0
  }
}
```

## Email Template Examples

### Urgent Alert (Risk Score ≥ 80)
- **Subject:** 🚨 URGENT: Fire Detected - Immediate Action Required
- **Color:** Red (#dc2626)
- **Priority:** High
- **Actions:** Evacuate immediately, call emergency services

### Warning Alert (Risk Score ≥ 40)
- **Subject:** ⚠️ WARNING: Fire Risk Detected - Action Required
- **Color:** Orange (#f59e0b)
- **Priority:** High
- **Actions:** Investigate area, check heat sources, monitor closely

### Caution Alert (Risk Score ≥ 20)
- **Subject:** ⚡ CAUTION: Elevated Fire Risk Detected
- **Color:** Yellow (#eab308)
- **Priority:** Normal
- **Actions:** Review sensor readings, check for unusual activity

## Usage Examples

### Example 1: Add Multiple Recipients

```bash
# Add primary admin (receives all alerts)
curl -X POST http://localhost:8080/api/email-recipients \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@company.com",
    "name": "System Administrator",
    "alertLevels": ["all"],
    "enabled": true
  }'

# Add security team (only urgent alerts)
curl -X POST http://localhost:8080/api/email-recipients \
  -H "Content-Type: application/json" \
  -d '{
    "email": "security@company.com",
    "name": "Security Team",
    "alertLevels": ["urgent"],
    "enabled": true
  }'

# Add facility manager (urgent and warning)
curl -X POST http://localhost:8080/api/email-recipients \
  -H "Content-Type: application/json" \
  -d '{
    "email": "facilities@company.com",
    "name": "Facility Manager",
    "alertLevels": ["urgent", "warning"],
    "enabled": true
  }'
```

### Example 2: Test Email Configuration

```bash
# Send test email to verify setup
curl -X POST http://localhost:8080/api/test-alert-email \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "riskScore": 85
  }'
```

### Example 3: Trigger Alert with Email Notification

```bash
# Simulate fire detection (will send emails to eligible recipients)
curl -X POST http://localhost:8080/api/alert-state \
  -H "Content-Type: application/json" \
  -d '{
    "isActive": true,
    "level": 4,
    "message": "Fire detected in kitchen area",
    "riskScore": 85,
    "confidence": 0.95,
    "location": "Kitchen",
    "deviceId": "SAAFE-KITCHEN-001"
  }'
```

### Example 4: Temporarily Disable a Recipient

```bash
# Disable without removing
curl -X PUT http://localhost:8080/api/email-recipients/user@example.com \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": false
  }'
```

## Spam Prevention

The system includes intelligent spam prevention:

1. **5-Minute Cooldown:** After sending emails, the system waits 5 minutes before sending again
2. **Significant Change Detection:** During cooldown, emails are only sent if risk score increases by 20+ points
3. **Per-Alert Tracking:** Each alert level is tracked separately

Example:
- Email sent at 10:00 AM (Risk Score: 45)
- New alert at 10:02 AM (Risk Score: 50) → **Skipped** (within cooldown, increase < 20)
- New alert at 10:03 AM (Risk Score: 70) → **Sent** (increase ≥ 20 points)
- New alert at 10:06 AM (Risk Score: 75) → **Sent** (cooldown expired)

## Monitoring

Check server logs for email activity:

```
✓ Alert email sent to System Administrator (admin@company.com)
✓ Alert email sent to Security Team (security@company.com)
Email notification results: { sent: 2, failed: 0, skipped: 1 }
```

## Troubleshooting

### Emails Not Sending

1. **Check Gmail App Password:**
   - Ensure you're using an app-specific password, not your regular Gmail password
   - Verify 2-Step Verification is enabled on your Google account

2. **Check Environment Variables:**
   ```bash
   echo $SENDER_EMAIL
   echo $SENDER_PASSWORD
   ```

3. **Test Email Service:**
   ```bash
   curl -X POST http://localhost:8080/api/send-test-email \
     -H "Content-Type: application/json" \
     -d '{"test_email": "your-email@example.com"}'
   ```

4. **Check Server Logs:**
   Look for error messages in the terminal running [`server.js`](backend/server.js:1)

### Recipients Not Receiving Emails

1. **Check Recipient Status:**
   ```bash
   curl http://localhost:8080/api/email-recipients
   ```
   Ensure `enabled: true`

2. **Verify Alert Levels:**
   - Check if recipient's `alertLevels` match the current risk score
   - Use `["all"]` to receive all alerts

3. **Check Spam/Junk Folder:**
   - Emails might be filtered by recipient's email provider
   - Add sender email to safe senders list

### Cooldown Issues

If emails aren't sending due to cooldown:

1. **Wait 5 Minutes:** The cooldown resets after 5 minutes
2. **Increase Risk Score:** Increase by 20+ points to bypass cooldown
3. **Restart Server:** Clears cooldown state (for testing only)

## Security Best Practices

1. **Use Environment Variables:** Never commit email credentials to version control
2. **App-Specific Passwords:** Use Gmail app passwords instead of account passwords
3. **Limit Recipients:** Only add necessary personnel to recipient list
4. **Regular Audits:** Periodically review and update recipient list
5. **Test Regularly:** Send test emails to verify configuration

## Integration with Frontend

The frontend can integrate with these endpoints to:

1. Display current email recipients in settings
2. Allow admins to add/remove recipients
3. Show email notification status in alert responses
4. Provide test email functionality

Example React component usage:

```javascript
// Fetch recipients
const recipients = await fetch('/api/email-recipients').then(r => r.json());

// Add recipient
await fetch('/api/email-recipients', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: 'new@example.com',
    name: 'New User',
    alertLevels: ['all'],
    enabled: true
  })
});

// Send test email
await fetch('/api/test-alert-email', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ email: 'test@example.com', riskScore: 85 })
});
```

## Summary

The email notification system provides:

✅ **Automatic alerts** when fire risks are detected  
✅ **Multiple recipients** with customizable alert levels  
✅ **Professional templates** with detailed information  
✅ **Spam prevention** to avoid email flooding  
✅ **Easy management** via REST API  
✅ **Gmail integration** with app-specific passwords  

For questions or issues, check the server logs or refer to the troubleshooting section above.