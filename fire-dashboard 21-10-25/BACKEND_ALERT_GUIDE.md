# Backend Fire Alert Sending Guide

This guide explains how to send dummy fire alert data to the backend after deployment.

## 📋 Overview

Once your dashboard is deployed, you can trigger fire alerts and email notifications from the backend using API calls. This is useful for:
- Testing the email notification system
- Simulating fire scenarios
- Demonstrating the system to stakeholders
- Automated monitoring and testing

## 🚀 Quick Start

### Method 1: Using the Shell Script (Recommended)

We've provided a convenient shell script that makes sending alerts easy:

```bash
# Make the script executable (first time only)
chmod +x send-fire-alerts.sh

# Send a fire alert to local backend
./send-fire-alerts.sh http://localhost:8080 fire

# Send a fire alert to deployed backend
./send-fire-alerts.sh https://your-backend-url.com fire

# Send all alert types in sequence
./send-fire-alerts.sh http://localhost:8080 all
```

### Method 2: Using cURL Directly

You can also send alerts directly using cURL:

```bash
# Fire Detected Alert (Risk Score: 95)
curl -X POST http://localhost:8080/api/alert-state \
  -H "Content-Type: application/json" \
  -d '{
    "isActive": true,
    "level": 9,
    "message": "FIRE DETECTED - immediate action required",
    "riskScore": 95,
    "confidence": 0.95,
    "location": "Kitchen",
    "deviceId": "SAAFE-KITCHEN-001"
  }'
```

### Method 3: Using Postman or API Client

1. Create a new POST request
2. URL: `http://your-backend-url/api/alert-state`
3. Headers: `Content-Type: application/json`
4. Body: Use one of the JSON payloads below

## 📊 Alert Types and Payloads

### 1. Non-Fire (Normal Conditions)
**Risk Score: 10** | **No Emails Sent**

```json
{
  "isActive": false,
  "level": 1,
  "message": "System operating normally",
  "riskScore": 10,
  "confidence": 0.9,
  "location": "Kitchen",
  "deviceId": "SAAFE-KITCHEN-001"
}
```

### 2. Caution Alert
**Risk Score: 25** | **Emails Sent to "Caution" Recipients**

```json
{
  "isActive": true,
  "level": 2,
  "message": "Elevated fire risk - monitoring recommended",
  "riskScore": 25,
  "confidence": 0.7,
  "location": "Kitchen",
  "deviceId": "SAAFE-KITCHEN-001"
}
```

### 3. Fire Predicted (Warning)
**Risk Score: 55** | **Emails Sent to "Warning" and "All" Recipients**

```json
{
  "isActive": true,
  "level": 5,
  "message": "Fire predicted - elevated risk detected",
  "riskScore": 55,
  "confidence": 0.75,
  "location": "Kitchen",
  "deviceId": "SAAFE-KITCHEN-001"
}
```

### 4. Fire Detected (Urgent)
**Risk Score: 95** | **Emails Sent to "Urgent" and "All" Recipients**

```json
{
  "isActive": true,
  "level": 9,
  "message": "FIRE DETECTED - immediate action required",
  "riskScore": 95,
  "confidence": 0.95,
  "location": "Kitchen",
  "deviceId": "SAAFE-KITCHEN-001"
}
```

## 🎯 Script Usage Examples

### Local Testing
```bash
# Test with local backend
./send-fire-alerts.sh http://localhost:8080 fire
./send-fire-alerts.sh http://localhost:8080 predicted
./send-fire-alerts.sh http://localhost:8080 caution
./send-fire-alerts.sh http://localhost:8080 non-fire
```

### Production/Deployed Backend
```bash
# Replace with your actual backend URL
./send-fire-alerts.sh https://api.yourcompany.com fire
./send-fire-alerts.sh https://your-backend.herokuapp.com predicted
```

### Testing All Alert Types
```bash
# Sends all alert types in sequence with 2-second delays
./send-fire-alerts.sh http://localhost:8080 all
```

## 📧 Email Notification Behavior

### Email Sending Rules:
- **Risk Score < 20**: No emails sent
- **Risk Score 20-39**: Emails to "Caution" and "All" recipients
- **Risk Score 40-79**: Emails to "Warning" and "All" recipients  
- **Risk Score 80+**: Emails to "Urgent" and "All" recipients

### Spam Prevention:
- 5-minute cooldown between emails
- Cooldown bypassed if risk score increases by 20+ points
- Only active alerts trigger emails (`isActive: true`)

## 🔧 Advanced Usage

### Custom Alert with Different Location
```bash
curl -X POST http://localhost:8080/api/alert-state \
  -H "Content-Type: application/json" \
  -d '{
    "isActive": true,
    "level": 7,
    "message": "Fire detected in storage room",
    "riskScore": 85,
    "confidence": 0.92,
    "location": "Storage Room B",
    "deviceId": "SAAFE-STORAGE-002"
  }'
```

### Automated Testing Script
```bash
#!/bin/bash
# Test all alert levels automatically

BACKEND="http://localhost:8080"

echo "Testing alert system..."

# Normal
./send-fire-alerts.sh $BACKEND non-fire
sleep 5

# Caution
./send-fire-alerts.sh $BACKEND caution
sleep 5

# Warning
./send-fire-alerts.sh $BACKEND predicted
sleep 5

# Urgent
./send-fire-alerts.sh $BACKEND fire
sleep 5

# Back to normal
./send-fire-alerts.sh $BACKEND non-fire

echo "Testing complete!"
```

## 🌐 API Endpoints Reference

### Alert State Endpoint
- **URL**: `/api/alert-state`
- **Method**: `POST`
- **Content-Type**: `application/json`

### Required Fields:
- `isActive` (boolean): Whether alert is active
- `level` (number): Alert level (1-10)
- `message` (string): Alert message
- `riskScore` (number): Risk score (0-100)
- `confidence` (number): Confidence level (0-1)

### Optional Fields:
- `location` (string): Location of alert
- `deviceId` (string): Device identifier

### Response Format:
```json
{
  "status": "success",
  "message": "Alert state updated successfully",
  "data": {
    "isActive": true,
    "level": 9,
    "message": "FIRE DETECTED - immediate action required",
    "riskScore": 95,
    "confidence": 0.95,
    "timestamp": "2025-10-21T15:20:00.000Z"
  },
  "emailNotifications": {
    "sent": 3,
    "failed": 0,
    "skipped": 0
  }
}
```

## 🔐 Security Considerations

### For Production Deployments:
1. **Add Authentication**: Protect the API endpoint with authentication
2. **Rate Limiting**: Implement rate limiting to prevent abuse
3. **IP Whitelisting**: Only allow trusted IPs to send alerts
4. **API Keys**: Require API keys for alert submissions
5. **Logging**: Log all alert submissions for audit trails

### Example with API Key:
```bash
curl -X POST https://api.yourcompany.com/api/alert-state \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{"isActive": true, "level": 9, ...}'
```

## 📝 Troubleshooting

### Emails Not Sending?
1. Check Gmail app password is valid
2. Verify environment variables are set
3. Check recipient email addresses are configured
4. Review backend logs for errors
5. Ensure risk score meets threshold (≥40 for emails)

### Script Not Working?
1. Make sure script is executable: `chmod +x send-fire-alerts.sh`
2. Check backend URL is correct
3. Verify backend is running and accessible
4. Check for firewall or network issues

### Testing Email Configuration:
```bash
# Send test email
curl -X POST http://localhost:8080/api/test-alert-email \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "riskScore": 85
  }'
```

## 📚 Additional Resources

- Backend API Documentation: See `EMAIL_NOTIFICATION_GUIDE.md`
- Email Configuration: Dashboard → Email Alerts section
- Frontend Integration: See `src/components/FireDataSender.tsx`

## 🎓 Best Practices

1. **Test in Development First**: Always test alerts in development before production
2. **Use Descriptive Messages**: Include clear, actionable alert messages
3. **Set Appropriate Risk Scores**: Use realistic risk scores for testing
4. **Monitor Email Delivery**: Check email logs to ensure delivery
5. **Document Custom Alerts**: Keep track of custom alert scenarios
6. **Regular Testing**: Schedule regular tests of the alert system

---

For questions or issues, refer to the main project documentation or contact the development team.