# 🔥 SAAFE Fire Data Sender - Standalone Application

A standalone web application for sending test fire alert data to the SAAFE backend API. This application can be hosted separately from the main dashboard.

## 📋 Overview

This is a simple HTML/CSS/JavaScript application that allows you to send fire alert test data to your backend API. The dashboard will automatically update within 5 seconds to reflect the changes.

## 🚀 Features

- **Standalone Application**: Single HTML file, no build process required
- **Configurable API URL**: Point to any backend API endpoint
- **3 Alert Types**: Non-Fire, Fire Predicted, and Fire
- **Real-time Feedback**: Shows success/error messages
- **Beautiful UI**: Modern, responsive design
- **No Dependencies**: Pure HTML/CSS/JavaScript

## 📦 Deployment Options

### Option 1: Local Testing

Simply open `index.html` in your web browser:

```bash
# Navigate to the directory
cd fire-data-sender-standalone

# Open in browser (macOS)
open index.html

# Open in browser (Linux)
xdg-open index.html

# Open in browser (Windows)
start index.html
```

### Option 2: Simple HTTP Server

Using Python:
```bash
# Python 3
python3 -m http.server 8000

# Then open: http://localhost:8000
```

Using Node.js:
```bash
# Install http-server globally
npm install -g http-server

# Run server
http-server -p 8000

# Then open: http://localhost:8000
```

### Option 3: Deploy to Netlify

1. Create a new site on [Netlify](https://netlify.com)
2. Drag and drop the `fire-data-sender-standalone` folder
3. Your site will be live at `https://your-site-name.netlify.app`

### Option 4: Deploy to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd fire-data-sender-standalone
vercel

# Follow the prompts
```

### Option 5: Deploy to GitHub Pages

1. Create a new GitHub repository
2. Push the `fire-data-sender-standalone` folder
3. Go to Settings → Pages
4. Select main branch and root folder
5. Your site will be live at `https://username.github.io/repo-name`

### Option 6: Deploy to AWS S3

```bash
# Create S3 bucket
aws s3 mb s3://saafe-data-sender

# Enable static website hosting
aws s3 website s3://saafe-data-sender --index-document index.html

# Upload files
aws s3 cp index.html s3://saafe-data-sender/ --acl public-read

# Your site will be live at:
# http://saafe-data-sender.s3-website-us-east-1.amazonaws.com
```

## 🎮 How to Use

1. **Open the Application**
   - Access via your chosen deployment method

2. **Configure API URL**
   - Enter your backend API URL (e.g., `http://localhost:8080` or `https://your-api.com`)
   - The default is `http://localhost:8080`

3. **Send Test Data**
   - Click **"✅ Non-Fire"** to send normal/safe data (Risk Score: 10)
   - Click **"⚠️ Fire Predicted"** to send warning data (Risk Score: 55)
   - Click **"🔥 Fire"** to send critical fire data (Risk Score: 95)

4. **View Results**
   - Success/error messages appear below the buttons
   - The dashboard updates within 5 seconds

## 📊 Alert Types

### Non-Fire (Green)
- **Risk Score**: 10
- **Level**: 1
- **Message**: "System operating normally"
- **Confidence**: 90%
- **Dashboard Effect**: All components show green/safe status

### Fire Predicted (Yellow)
- **Risk Score**: 55
- **Level**: 5
- **Message**: "Fire predicted - elevated risk detected"
- **Confidence**: 75%
- **Dashboard Effect**: All components show yellow/warning status

### Fire (Red)
- **Risk Score**: 95
- **Level**: 9
- **Message**: "FIRE DETECTED - immediate action required"
- **Confidence**: 95%
- **Dashboard Effect**: All components show red/fire status + sends emails

## 🔧 API Configuration

The application sends POST requests to:
```
POST {API_URL}/api/alert-state
```

With payload:
```json
{
  "isActive": boolean,
  "level": number (1-10),
  "message": string,
  "riskScore": number (0-100),
  "confidence": number (0-1),
  "timestamp": ISO string
}
```

## 🌐 CORS Configuration

If your backend is on a different domain, ensure CORS is enabled:

```javascript
// In your backend (Node.js/Express)
app.use(cors({
  origin: '*', // or specify your data sender URL
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));
```

## 📱 Architecture

```
┌─────────────────────┐
│  Data Sender App    │
│  (Standalone HTML)  │
└──────────┬──────────┘
           │ POST /api/alert-state
           ▼
┌─────────────────────┐
│   Backend API       │
│  (Node.js/Express)  │
└──────────┬──────────┘
           │ Stores alert state
           ▼
┌─────────────────────┐
│  Fire Dashboard     │
│  (React App)        │
│  Polls every 5s     │
└─────────────────────┘
```

## 🎯 Use Cases

1. **Testing**: Test dashboard without real sensors
2. **Demos**: Show fire detection system to stakeholders
3. **Training**: Train staff on system responses
4. **Development**: Develop dashboard features without hardware
5. **QA**: Quality assurance testing of alert workflows

## 🔒 Security Notes

- **Production**: Use HTTPS for both data sender and backend
- **Authentication**: Add API keys or JWT tokens if needed
- **Rate Limiting**: Implement rate limiting on backend
- **Input Validation**: Backend should validate all inputs

## 📝 Customization

### Change Colors
Edit the CSS in `index.html`:
```css
.alert-button.fire {
    background: #your-color;
}
```

### Add More Alert Types
Add new buttons and cases in the `sendAlert()` function.

### Change API Endpoint
Modify the default API URL in the input field or JavaScript.

## 🐛 Troubleshooting

### "Failed to send alert"
- Check if backend is running
- Verify API URL is correct
- Check browser console for CORS errors
- Ensure backend accepts POST requests

### Dashboard Not Updating
- Wait 5 seconds for polling cycle
- Check if dashboard is polling the same backend
- Verify backend is storing alert state correctly

### CORS Errors
- Enable CORS on backend
- Use same protocol (HTTP/HTTPS)
- Check browser console for specific error

## 📞 Support

For issues or questions:
- Check backend logs
- Verify API endpoints
- Test with curl first
- Check browser console

## 🎉 Success!

Once deployed, you can:
- ✅ Send test data from anywhere
- ✅ Dashboard updates automatically
- ✅ No code changes needed
- ✅ Works with any backend API
- ✅ Easy to share with team

Enjoy testing your SAAFE fire detection system! 🔥