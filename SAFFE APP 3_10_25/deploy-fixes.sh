#!/bin/bash

# SAAFE Fire Dashboard - Deploy Data Flow Fixes to Vercel
# This script builds and deploys the fixed application to Vercel

echo "🔧 SAAFE Fire Dashboard - Deploying Data Flow Fixes"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the SAAFE APP 3_10_25 directory."
    exit 1
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "⚠️  Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo "📦 Step 1: Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "🏗️  Step 2: Building the application..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "✅ Build successful!"
echo ""
echo "🚀 Step 3: Deploying to Vercel..."
echo ""
echo "Choose deployment type:"
echo "  1) Production deployment (recommended)"
echo "  2) Preview deployment (for testing)"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "Deploying to production..."
        vercel --prod
        ;;
    2)
        echo ""
        echo "Deploying preview..."
        vercel
        ;;
    *)
        echo "Invalid choice. Deploying to production by default..."
        vercel --prod
        ;;
esac

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Visit your deployment URL"
    echo "2. Test the Fire Data Sender component"
    echo "3. Verify the Dashboard receives updates"
    echo "4. Check the browser console for any errors"
    echo ""
    echo "🔍 Testing Checklist:"
    echo "  ✓ Click 'Non-Fire' button - should show green status"
    echo "  ✓ Click 'Fire Predicted' button - should show yellow status"
    echo "  ✓ Click 'Fire' button - should show red status"
    echo "  ✓ Dashboard should update within 5 seconds"
    echo "  ✓ Alert history should show new events"
    echo ""
else
    echo ""
    echo "❌ Deployment failed. Please check the error messages above."
    exit 1
fi