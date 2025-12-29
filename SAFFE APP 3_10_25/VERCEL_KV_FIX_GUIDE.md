# Vercel KV Fix for Fire Alert Persistence

## Problem Summary

Fire alerts sent from the Data Sender were not appearing on the Dashboard in the deployed Vercel application due to **serverless function statelessness**. The in-memory storage was being reset on each cold start, causing data loss between requests.

## Root Cause

- **Vercel serverless functions are stateless** - each invocation may get a fresh instance
- In-memory variables (`alertState`, `alertHistory`) were reset on cold starts
- Data Sender and Dashboard could hit different serverless instances
- Result: Alert state didn't persist between requests

## Solution Implemented

Replaced in-memory storage with **Vercel KV (Redis)** for persistent data storage across all serverless function invocations.

### Changes Made

1. **Installed Vercel KV package**:
   ```bash
   npm install @vercel/kv
   ```

2. **Updated `api/index.js`**:
   - Imported `@vercel/kv`
   - Created helper functions for KV operations:
     - `getAlertState()` - Retrieves alert state from Redis
     - `setAlertState(state)` - Stores alert state in Redis
     - `getAlertHistory()` - Retrieves alert history from Redis
     - `setAlertHistory(history)` - Stores alert history in Redis
   - Updated all endpoints to use KV storage instead of in-memory variables
   - Made all storage operations asynchronous

3. **KV Keys Used**:
   - `saafe:alert-state` - Current alert state
   - `saafe:alert-history` - Alert history (max 50 events)

## Deployment Steps

### Step 1: Set Up Vercel KV Database

1. **Go to Vercel Dashboard**:
   - Navigate to https://vercel.com/dashboard
   - Select your project: `saafe-fire-dashboard`

2. **Create KV Database**:
   - Go to the "Storage" tab
   - Click "Create Database"
   - Select "KV (Redis)"
   - Name it: `saafe-kv-store`
   - Choose region: `us-east-1` (or closest to your users)
   - Click "Create"

3. **Connect to Project**:
   - After creation, click "Connect to Project"
   - Select your `saafe-fire-dashboard` project
   - Click "Connect"
   - This automatically adds the required environment variables:
     - `KV_REST_API_URL`
     - `KV_REST_API_TOKEN`
     - `KV_REST_API_READ_ONLY_TOKEN`
     - `KV_URL`

### Step 2: Deploy Updated Code

1. **Commit Changes**:
   ```bash
   cd "SAFFE APP 3_10_25"
   git add .
   git commit -m "Fix: Implement Vercel KV for persistent alert storage"
   git push origin main
   ```

2. **Vercel Auto-Deploy**:
   - Vercel will automatically detect the push and deploy
   - Monitor deployment at: https://vercel.com/dashboard

3. **Manual Deploy (Alternative)**:
   ```bash
   cd "SAFFE APP 3_10_25"
   vercel --prod
   ```

### Step 3: Verify the Fix

1. **Open Data Sender**:
   - Navigate to: `https://your-app.vercel.app/data-sender`

2. **Send Fire Alert**:
   - Click the "🔥 Fire" button
   - Wait for success confirmation

3. **Check Dashboard**:
   - Navigate to: `https://your-app.vercel.app/`
   - The fire alert should now appear within 5 seconds
   - Alert status display should show "FIRE DETECTED"
   - Map markers should turn red
   - Alert history should show the fire event

4. **Test Different Scenarios**:
   - Send "⚠️ Fire Predicted" alert → Should show yellow/amber
   - Send "✅ Non-Fire" alert → Should show green
   - Refresh the page → Alerts should persist

## How It Works Now

### Data Flow with Vercel KV

```
┌─────────────────┐
│  Data Sender    │
│  (POST alert)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Serverless Function    │
│  Instance A             │
│  ├─ Receives POST       │
│  ├─ Stores in KV Redis  │
│  └─ Returns success     │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Vercel KV (Redis)      │
│  ├─ saafe:alert-state   │
│  └─ saafe:alert-history │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Serverless Function    │
│  Instance B (different) │
│  ├─ Receives GET        │
│  ├─ Reads from KV Redis │
│  └─ Returns alert data  │
└─────────────────────────┘
         │
         ▼
┌─────────────────┐
│   Dashboard     │
│  (Shows alert)  │
└─────────────────┘
```

### Key Benefits

1. **Persistence**: Data survives cold starts and instance changes
2. **Consistency**: All serverless instances read from same Redis store
3. **Performance**: Redis is fast (sub-millisecond reads)
4. **Scalability**: Handles multiple concurrent requests
5. **Reliability**: Vercel KV has built-in redundancy

## Testing Checklist

- [ ] Fire alert appears on dashboard within 5 seconds
- [ ] Fire predicted alert shows yellow/amber status
- [ ] Non-fire alert shows green status
- [ ] Alert history displays correctly
- [ ] Map markers update with correct colors
- [ ] Alerts persist after page refresh
- [ ] Multiple alerts can be sent in sequence
- [ ] Dashboard polling (every 5s) works correctly

## Troubleshooting

### Issue: Alerts still not appearing

**Check KV Connection**:
```bash
# In Vercel Dashboard
1. Go to Storage tab
2. Verify KV database is connected to project
3. Check environment variables are set
```

**Check Logs**:
```bash
# In Vercel Dashboard
1. Go to Deployments
2. Click on latest deployment
3. View Function Logs
4. Look for KV connection errors
```

### Issue: "KV_REST_API_URL is not defined"

**Solution**:
- KV database not properly connected to project
- Go to Storage → Select KV database → Connect to Project
- Redeploy after connection

### Issue: Slow response times

**Solution**:
- Check KV database region matches deployment region
- Consider upgrading Vercel plan for better performance
- Monitor KV usage in Vercel Dashboard

## Cost Considerations

**Vercel KV Pricing** (as of 2025):
- **Hobby Plan**: 256 MB storage, 3,000 commands/day (FREE)
- **Pro Plan**: 1 GB storage, 100,000 commands/day
- **Enterprise**: Custom limits

**Current Usage Estimate**:
- Alert state updates: ~10-50 per day
- Dashboard polling: ~17,280 reads per day (1 device, 5s interval)
- Alert history: ~50 events stored
- **Total**: Well within Hobby plan limits

## Monitoring

**Track KV Usage**:
1. Go to Vercel Dashboard → Storage
2. Select your KV database
3. View metrics:
   - Commands per day
   - Storage used
   - Response times

**Set Up Alerts**:
- Configure usage alerts in Vercel Dashboard
- Get notified when approaching limits

## Additional Resources

- [Vercel KV Documentation](https://vercel.com/docs/storage/vercel-kv)
- [Redis Commands Reference](https://redis.io/commands/)
- [Vercel Serverless Functions](https://vercel.com/docs/functions)

## Support

If issues persist:
1. Check Vercel Status: https://www.vercel-status.com/
2. Review Vercel KV Docs: https://vercel.com/docs/storage/vercel-kv
3. Contact Vercel Support: https://vercel.com/support

---

**Last Updated**: 2025-10-06
**Status**: ✅ Ready for Deployment