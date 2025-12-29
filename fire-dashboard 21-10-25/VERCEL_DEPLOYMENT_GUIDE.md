# Fire Dashboard - Vercel Deployment Guide

## Overview
This guide explains how to deploy the Fire Dashboard (frontend + backend API) to Vercel.

## Architecture
```
Fire Dashboard on Vercel
├── Frontend (React + Vite) → Static files in /dist
└── Backend API (Serverless) → /api/index.js
```

## Prerequisites
1. Vercel account (free tier works)
2. Vercel CLI installed: `npm i -g vercel`
3. Git repository (optional but recommended)

## Deployment Steps

### Option 1: Deploy via Vercel CLI (Recommended)

1. **Install Vercel CLI** (if not already installed):
```bash
npm i -g vercel
```

2. **Navigate to project directory**:
```bash
cd "fire-dashboard 21-10-25"
```

3. **Login to Vercel**:
```bash
vercel login
```

4. **Deploy**:
```bash
vercel --prod
```

5. **Follow the prompts**:
   - Set up and deploy? `Y`
   - Which scope? Select your account
   - Link to existing project? `N`
   - What's your project's name? `fire-dashboard` (or your choice)
   - In which directory is your code located? `./`
   - Want to override the settings? `N`

6. **Get your deployment URL**:
   - Vercel will provide a URL like: `https://fire-dashboard-xxx.vercel.app`
   - The backend API will be at: `https://fire-dashboard-xxx.vercel.app/api/alert-state`

### Option 2: Deploy via Vercel Dashboard

1. **Push code to GitHub** (if not already):
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

2. **Import to Vercel**:
   - Go to https://vercel.com/new
   - Import your GitHub repository
   - Vercel will auto-detect the Vite framework
   - Click "Deploy"

## Configuration Files

### vercel.json
```json
{
  "version": 2,
  "name": "fire-dashboard",
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "/api/index.js"
    },
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ],
  "functions": {
    "api/index.js": {
      "memory": 1024,
      "maxDuration": 10
    }
  }
}
```

### API Endpoints
Once deployed, your API will be available at:
- `GET /api/health` - Health check
- `GET /api/alert-state` - Get current alert state
- `POST /api/send-alert` - Send alert (from FireDataSender)
- `GET /api/alert-history` - Get alert history

## Environment Variables

### For Dashboard (Frontend)
Update `src/config/config.ts`:
```typescript
export const API_BASE_URL = 'https://your-deployment.vercel.app';
```

Or use environment variable:
```bash
# In Vercel Dashboard → Settings → Environment Variables
VITE_API_BASE_URL=https://your-deployment.vercel.app
```

Then in `config.ts`:
```typescript
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080';
```

### For Mapbox (if needed)
```bash
VITE_MAPBOX_TOKEN=your_mapbox_token_here
```

## Post-Deployment Steps

1. **Test the deployment**:
```bash
# Test backend API
curl https://your-deployment.vercel.app/api/health

# Test alert state
curl https://your-deployment.vercel.app/api/alert-state
```

2. **Update FireDataSender**:
   - Open http://localhost:8001
   - In "Backend API URL" field, enter: `https://your-deployment.vercel.app`
   - Click any alert button to test

3. **Verify dashboard updates**:
   - Open https://your-deployment.vercel.app
   - All components should update within 5 seconds after sending an alert

## Updating the Deployment

### Via CLI:
```bash
cd "fire-dashboard 21-10-25"
vercel --prod
```

### Via Git (if connected to GitHub):
```bash
git add .
git commit -m "Update dashboard"
git push
# Vercel will auto-deploy
```

## Troubleshooting

### Issue: API returns 404
- Check that `/api/index.js` exists
- Verify `vercel.json` rewrites configuration
- Check Vercel deployment logs

### Issue: CORS errors
- The API has CORS enabled for all origins (`*`)
- If issues persist, check browser console for specific errors

### Issue: Dashboard not updating
- Verify API_BASE_URL in `config.ts` points to deployed URL
- Check browser network tab for API calls
- Verify backend API is responding: `curl https://your-url.vercel.app/api/alert-state`

### Issue: Build fails
- Check Node.js version (should be 18.x or higher)
- Verify all dependencies are in `package.json`
- Check Vercel build logs for specific errors

## Important Notes

1. **In-Memory Storage**: The backend uses in-memory storage, which resets on each cold start (after ~5 minutes of inactivity)

2. **Cold Starts**: First request after inactivity may take 1-2 seconds

3. **Rate Limits**: Vercel free tier has limits:
   - 100 GB bandwidth/month
   - 100 hours serverless function execution/month
   - 6000 minutes build time/month

4. **Custom Domain**: You can add a custom domain in Vercel Dashboard → Settings → Domains

## Next Steps

After deployment:
1. Deploy the FireDataSender (optional - can keep it local)
2. Test the complete flow: DataSender → Backend API → Dashboard
3. Share the dashboard URL with your team
4. Monitor usage in Vercel Dashboard → Analytics

## Support

For issues:
- Check Vercel deployment logs
- Review browser console for errors
- Verify API endpoints are responding
- Check CORS configuration if cross-origin issues occur