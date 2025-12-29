# 🚀 Fire Dashboard - Quick Start Guide

## 📦 What You Have

This fire dashboard includes:
- ✅ Full fire detection monitoring system
- ✅ Email alert configuration interface
- ✅ Backend API for sending fire alerts
- ✅ Automated email notifications
- ✅ Shell script for testing alerts

## 🎯 Quick Setup (3 Steps)

### 1. Install Dependencies
```bash
cd fire-dashboard\ 21-10-25
npm install --legacy-peer-deps
```

### 2. Start the Dashboard
```bash
npm run dev
```
Dashboard will be available at: **http://localhost:5174**

### 3. Configure Email (Optional)
- Click "Email Alerts" in the sidebar
- Add email recipients
- Configure alert levels
- Send test emails

## 🔥 Sending Fire Alerts from Backend

### Quick Test
```bash
# Send a fire alert
./send-fire-alerts.sh http://localhost:8080 fire

# Send a warning alert
./send-fire-alerts.sh http://localhost:8080 predicted

# Send all alert types
./send-fire-alerts.sh http://localhost:8080 all
```

### Using cURL
```bash
curl -X POST http://localhost:8080/api/alert-state \
  -H "Content-Type: application/json" \
  -d '{
    "isActive": true,
    "level": 9,
    "message": "FIRE DETECTED",
    "riskScore": 95,
    "confidence": 0.95
  }'
```

## 📧 Email Configuration

### Fix Email Sending
The backend needs a valid Gmail app password:

1. Go to: https://myaccount.google.com/security
2. Enable 2-Step Verification
3. Create App Password for "Mail"
4. Update backend `.env` file with new password
5. Restart backend server

## 🌐 After Deployment

Once deployed, replace `http://localhost:8080` with your backend URL:

```bash
# Production example
./send-fire-alerts.sh https://your-backend.com fire
```

## 📚 Documentation

- **Full Alert Guide**: See `BACKEND_ALERT_GUIDE.md`
- **Email Setup**: See `EMAIL_NOTIFICATION_GUIDE.md` in SAAFE APP
- **Script Usage**: Run `./send-fire-alerts.sh` without arguments for help

## 🎨 Dashboard Features

### Views Available:
1. **Helios** - Global map view
2. **Athena** - Strategic dashboard
3. **Grid** - Asset manager
4. **Fire Detection** - Detection system
5. **Email Alerts** - Email configuration ⭐ NEW

### Email Alert Levels:
- **All** - Receives all alerts
- **Urgent** - Risk score ≥ 80
- **Warning** - Risk score ≥ 40
- **Caution** - Risk score ≥ 20

## 🔧 Troubleshooting

### Dashboard won't start?
```bash
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps
npm run dev
```

### Backend not responding?
Check if backend is running on port 8080:
```bash
curl http://localhost:8080/api/email-recipients
```

### Emails not sending?
1. Check Gmail app password is valid
2. Verify recipients are configured
3. Check backend logs for errors
4. Test with: `./send-fire-alerts.sh http://localhost:8080 fire`

## 💡 Tips

- **Test locally first** before deploying
- **Use the script** for easy alert testing
- **Monitor backend logs** to see email activity
- **Configure multiple recipients** for redundancy
- **Set appropriate alert levels** per recipient

## 🎓 Next Steps

1. ✅ Configure email recipients in dashboard
2. ✅ Test alerts using the shell script
3. ✅ Verify email delivery
4. ✅ Deploy to production
5. ✅ Update script with production URL

---

Need help? Check `BACKEND_ALERT_GUIDE.md` for detailed instructions.