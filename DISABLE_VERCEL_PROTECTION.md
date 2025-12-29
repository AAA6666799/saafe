# How to Disable Vercel Deployment Protection

Both your dashboard and data sender are deployed but require authentication. Follow these steps to make them publicly accessible.

## Step 1: Access Vercel Dashboard

Go to: https://vercel.com/dashboard

## Step 2: Disable Protection for Fire Dashboard

1. Go to: https://vercel.com/saiajay1s-projects/fire-dashboard
2. Click on **"Settings"** tab
3. Scroll down to **"Deployment Protection"**
4. Change from **"All Deployments"** to **"Only Preview Deployments"** or **"Disabled"**
5. Click **"Save"**

## Step 3: Disable Protection for Data Sender

1. Go to: https://vercel.com/saiajay1s-projects/fire-data-sender-standalone
2. Click on **"Settings"** tab
3. Scroll down to **"Deployment Protection"**
4. Change from **"All Deployments"** to **"Only Preview Deployments"** or **"Disabled"**
5. Click **"Save"**

## Step 4: Test the Deployments

### Test Dashboard
Open: https://fire-dashboard-xi.vercel.app
- Should load without authentication
- All components should be visible

### Test Data Sender
Open: https://fire-data-sender-standalone.vercel.app
- Should load without authentication
- You should see the three alert buttons

### Test Complete Flow
1. Open Data Sender: https://fire-data-sender-standalone.vercel.app
2. In the "Backend API URL" field, enter: `https://fire-dashboard-xi.vercel.app`
3. Click any alert button (Non-Fire, Fire Predicted, or Fire)
4. Open Dashboard: https://fire-dashboard-xi.vercel.app
5. All components should update within 5 seconds

## Alternative: Use Vercel CLI to Disable Protection

```bash
# For fire-dashboard
cd "fire-dashboard 21-10-25"
vercel env add VERCEL_DEPLOYMENT_PROTECTION_BYPASS
# Enter a secret value when prompted

# For fire-data-sender
cd fire-data-sender-standalone
vercel env add VERCEL_DEPLOYMENT_PROTECTION_BYPASS
# Enter the same secret value
```

## Why This Happens

Vercel enables deployment protection by default for new projects to prevent unauthorized access during development. For production use, you can:

1. **Disable it completely** - Anyone can access (good for public dashboards)
2. **Use bypass tokens** - Share a secret URL with authorized users
3. **Keep it enabled** - Only you can access (good for internal tools)

## Recommended Setting

For a fire detection dashboard that needs to be accessible:
- **Dashboard**: Disable protection (public access needed)
- **Data Sender**: Keep protection OR disable (depends on who should send alerts)

## After Disabling Protection

Your complete system will work:
```
Data Sender (https://fire-data-sender-standalone.vercel.app)
    ↓ POST /api/send-alert
Backend API (https://fire-dashboard-xi.vercel.app/api)
    ↓ GET /api/alert-state (every 5s)
Dashboard (https://fire-dashboard-xi.vercel.app)
```

No CORS issues, no authentication issues - everything on Vercel! 🎉