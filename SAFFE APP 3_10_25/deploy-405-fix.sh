#!/bin/bash

# Deploy 405 Error Fix to Vercel
# This script deploys the fix for the HTTP 405 error in FireDataSender

echo "🚀 Deploying 405 Error Fix to Vercel..."
echo "================================================"

# Navigate to project directory
cd "SAFFE APP 3_10_25" || exit 1

# Check if we're in the right directory
if [ ! -f "vercel.json" ]; then
    echo "❌ Error: vercel.json not found. Are you in the correct directory?"
    exit 1
fi

echo "✅ Found vercel.json"

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo "✅ Vercel CLI is available"

# Show what we're deploying
echo ""
echo "📦 Changes being deployed:"
echo "  - Fixed /api/predict/saafe endpoint (405 error)"
echo "  - Added POST handler for fire prediction requests"
echo "  - Using native fetch API for model forwarding"
echo ""

# Deploy to Vercel
echo "🚀 Starting deployment..."
vercel --prod

# Check deployment status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
    echo ""
    echo "🎉 The 405 error fix has been deployed!"
    echo ""
    echo "📋 What was fixed:"
    echo "  1. Added missing /api/predict/saafe endpoint"
    echo "  2. Endpoint now forwards POST requests to Saafe model"
    echo "  3. FireDataSender buttons should now work correctly"
    echo ""
    echo "🧪 Test the fix:"
    echo "  1. Open your Vercel dashboard"
    echo "  2. Click on 'Fire Data Sender' tab"
    echo "  3. Click any event button (Non Fire, Fire Predicted, Fire)"
    echo "  4. Verify no 405 errors appear"
    echo ""
else
    echo ""
    echo "❌ Deployment failed!"
    echo "Please check the error messages above and try again."
    exit 1
fi