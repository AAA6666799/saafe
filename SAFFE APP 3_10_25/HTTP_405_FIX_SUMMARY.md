# HTTP 405 Error Fix Summary

## Problem Identified
**Error**: HTTP 405 (Method Not Allowed) when clicking event buttons in Fire Data Sender component

**Root Cause**: The [`FireDataSender.tsx`](src/components/FireDataSender.tsx:117-121) component was making POST requests to `/api/predict/saafe`, but this endpoint was not implemented in the [`api/index.js`](api/index.js:1) serverless function.

## Investigation Details

### Frontend Analysis
- **Component**: [`FireDataSender.tsx`](src/components/FireDataSender.tsx:1)
- **Line 117-121**: POST request to `/api/predict/saafe`
- **Line 162-166**: POST request to `/api/alert-state` (this one worked)
- **Trigger**: Clicking "Non Fire Predicted", "Fire Predicted", or "Fire" buttons

### Backend Analysis
- **File**: [`api/index.js`](api/index.js:1)
- **Issue**: Missing handler for `/predict/saafe` endpoint
- **Existing Endpoints**:
  - ✅ GET `/health`
  - ✅ GET `/fire-detection-data`
  - ✅ GET `/devices`
  - ✅ GET `/alert-state`
  - ✅ POST `/alert-state`
  - ✅ GET `/alert-history`
  - ✅ GET `/email-recipients`
  - ❌ POST `/predict/saafe` (MISSING - causing 405 error)

## Solution Implemented

### Changes Made to [`api/index.js`](api/index.js:229-252)

Added new endpoint handler at lines 229-252:

```javascript
// Predict endpoint - forward to Saafe model
if (path === '/predict/saafe' && method === 'POST') {
  try {
    const response = await fetch(MODEL_URLS.saafe, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(req.body)
    });

    const data = await response.json();
    
    return res.status(response.status).json({
      status: "success",
      message: "Prediction completed",
      data: data,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Prediction error:', error);
    return res.status(500).json({
      status: "error",
      message: "Failed to get prediction from model",
      error: error.message
    });
  }
}
```

### Key Features of the Fix

1. **Endpoint Handler**: Added POST handler for `/predict/saafe`
2. **Model Forwarding**: Forwards requests to the Saafe Lambda model URL
3. **Native Fetch**: Uses Node.js 18+ native fetch API (no dependencies needed)
4. **Error Handling**: Proper try-catch with error responses
5. **Response Format**: Consistent JSON response structure

### Model URL Configuration
The endpoint forwards to: `https://bjggbotpq6aglni3wd3qe5luf40wszod.lambda-url.us-east-1.on.aws/`

## Testing Instructions

### 1. Deploy the Fix
```bash
cd "SAFFE APP 3_10_25"
./deploy-405-fix.sh
```

Or manually:
```bash
cd "SAFFE APP 3_10_25"
vercel --prod
```

### 2. Test the Fire Data Sender

1. Open your deployed Vercel dashboard
2. Navigate to the "Fire Data Sender" tab
3. Click each button and verify:
   - ✅ **Non Fire Predicted** button works
   - ✅ **Fire Predicted** button works
   - ✅ **Fire** button works
4. Check browser console - should see:
   - "Alert sent successfully: {data}"
   - "Alert state updated on backend: {data}"
5. Verify no 405 errors appear

### 3. Expected Behavior

**When clicking "Non Fire Predicted":**
- Sends normal sensor data
- Risk score: 10
- Level: 1
- Message: "System operating normally"

**When clicking "Fire Predicted":**
- Sends elevated risk data
- Risk score: 55
- Level: 5
- Message: "Fire predicted - elevated risk detected"

**When clicking "Fire":**
- Sends critical fire data
- Risk score: 95
- Level: 9
- Message: "FIRE DETECTED - immediate action required"

## Technical Details

### Request Flow
1. User clicks button in [`FireDataSender.tsx`](src/components/FireDataSender.tsx:308-345)
2. Component sends POST to `/api/predict/saafe` with sensor data
3. Vercel routes to [`api/index.js`](api/index.js:229-252)
4. Handler forwards to Saafe Lambda model
5. Model returns prediction
6. Handler returns response to frontend
7. Component updates alert state via POST to `/api/alert-state`

### Data Flow
```
FireDataSender Component
    ↓ POST /api/predict/saafe
Vercel Serverless Function (api/index.js)
    ↓ Forward to Lambda
Saafe Fire Detection Model
    ↓ Return prediction
Vercel Serverless Function
    ↓ Return to frontend
FireDataSender Component
    ↓ POST /api/alert-state
Update Backend Alert State
```

## Files Modified

1. **[`api/index.js`](api/index.js:229-252)** - Added `/predict/saafe` endpoint handler
2. **[`deploy-405-fix.sh`](deploy-405-fix.sh:1)** - Created deployment script

## Verification Checklist

- [x] Identified missing endpoint causing 405 error
- [x] Added POST handler for `/predict/saafe`
- [x] Implemented model forwarding logic
- [x] Used native fetch API (no new dependencies)
- [x] Added proper error handling
- [x] Created deployment script
- [x] Documented the fix

## Next Steps

1. **Deploy**: Run `./deploy-405-fix.sh` to deploy to Vercel
2. **Test**: Verify all three buttons work without 405 errors
3. **Monitor**: Check Vercel logs for any issues
4. **Validate**: Confirm alert state updates correctly

## Additional Notes

- The fix uses Node.js 18+ native `fetch()` API
- No additional npm packages required
- Maintains consistency with existing endpoint patterns
- Proper CORS headers already configured in main handler
- Error responses follow existing format

## Related Files

- [`src/components/FireDataSender.tsx`](src/components/FireDataSender.tsx:1) - Frontend component
- [`api/index.js`](api/index.js:1) - Backend API handler
- [`vercel.json`](vercel.json:1) - Vercel configuration
- [`deploy-405-fix.sh`](deploy-405-fix.sh:1) - Deployment script

---

**Status**: ✅ Fix implemented and ready for deployment

**Date**: 2025-10-06

**Impact**: Resolves HTTP 405 errors in Fire Data Sender, enabling full testing functionality