#!/bin/bash

# Deploy Vercel KV Fix for Fire Alert Persistence
# This script deploys the updated code with Vercel KV integration

set -e  # Exit on error

echo "🚀 Deploying Vercel KV Fix for Fire Alert Persistence"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the SAFFE APP 3_10_25 directory."
    exit 1
fi

# Check if @vercel/kv is installed
if ! grep -q "@vercel/kv" package.json; then
    echo "❌ Error: @vercel/kv not found in package.json"
    echo "Please run: npm install @vercel/kv"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit with Vercel KV fix"
else
    echo "📦 Git repository already initialized"
fi

# Stage changes
echo "📝 Staging changes..."
git add .

# Check if there are changes to commit
if git diff-index --quiet HEAD --; then
    echo "ℹ️  No changes to commit"
else
    echo "💾 Committing changes..."
    git commit -m "Fix: Implement Vercel KV for persistent alert storage

- Replace in-memory storage with Vercel KV (Redis)
- Add helper functions for KV operations
- Update all endpoints to use persistent storage
- Fix fire alerts not appearing on dashboard
- Ensure data persists across serverless function invocations"
fi

echo ""
echo "🔧 Deployment Options:"
echo "1. Deploy to Vercel (requires Vercel CLI)"
echo "2. Push to Git (for automatic Vercel deployment)"
echo "3. Skip deployment (just commit changes)"
echo ""
read -p "Select option (1-3): " option

case $option in
    1)
        echo ""
        echo "🚀 Deploying to Vercel..."
        
        # Check if vercel CLI is installed
        if ! command -v vercel &> /dev/null; then
            echo "❌ Vercel CLI not found. Installing..."
            npm install -g vercel
        fi
        
        echo ""
        echo "⚠️  IMPORTANT: Before deploying, ensure you have:"
        echo "   1. Created a Vercel KV database in your Vercel Dashboard"
        echo "   2. Connected it to your project"
        echo "   3. Environment variables are set (KV_REST_API_URL, KV_REST_API_TOKEN)"
        echo ""
        read -p "Have you completed the above steps? (y/n): " confirm
        
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            vercel --prod
            echo ""
            echo "✅ Deployment complete!"
        else
            echo "❌ Deployment cancelled. Please complete the setup steps first."
            echo "📖 See VERCEL_KV_FIX_GUIDE.md for detailed instructions."
            exit 1
        fi
        ;;
    2)
        echo ""
        echo "📤 Pushing to Git..."
        
        # Check if remote is set
        if ! git remote | grep -q "origin"; then
            echo "❌ No git remote found. Please add a remote first:"
            echo "   git remote add origin <your-repo-url>"
            exit 1
        fi
        
        git push origin main
        echo ""
        echo "✅ Pushed to Git! Vercel will auto-deploy if connected."
        ;;
    3)
        echo ""
        echo "✅ Changes committed locally. Deploy manually when ready."
        ;;
    *)
        echo "❌ Invalid option selected."
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "🎉 Deployment Process Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Verify Vercel KV database is connected to your project"
echo "2. Check deployment status in Vercel Dashboard"
echo "3. Test fire alerts:"
echo "   - Go to /data-sender"
echo "   - Click 'Fire' button"
echo "   - Check dashboard for alert"
echo ""
echo "📖 For detailed instructions, see: VERCEL_KV_FIX_GUIDE.md"
echo "=================================================="