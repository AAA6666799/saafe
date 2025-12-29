#!/bin/bash

# Deploy SendGrid Email Update to Vercel
# This script helps deploy the updated email configuration

echo "🚀 Deploying SendGrid Email Configuration Update"
echo "================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the fire-dashboard directory."
    exit 1
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "⚠️  Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo "📋 Pre-deployment Checklist:"
echo ""
echo "Before deploying, make sure you have:"
echo "  ✓ Created a SendGrid account"
echo "  ✓ Verified your sender email address"
echo "  ✓ Generated a SendGrid API key"
echo ""
read -p "Have you completed these steps? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Please complete the setup steps first:"
    echo "1. Go to https://sendgrid.com/ and create an account"
    echo "2. Verify your sender email"
    echo "3. Create an API key with Mail Send permissions"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo ""
echo "📝 You'll need to set these environment variables in Vercel:"
echo ""
echo "  SENDGRID_API_KEY     - Your SendGrid API key"
echo "  SENDER_EMAIL         - Your verified sender email"
echo "  SENDER_NAME          - Display name (e.g., 'SAAFE AI Alert System')"
echo "  RECIPIENT_EMAIL      - Default recipient email"
echo ""
read -p "Press Enter to continue with deployment..."

echo ""
echo "🔧 Installing dependencies..."
cd api && npm install && cd ..

echo ""
echo "🚀 Deploying to Vercel..."
vercel --prod

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "📋 Next Steps:"
echo "1. Go to your Vercel project dashboard"
echo "2. Navigate to Settings → Environment Variables"
echo "3. Add the following variables:"
echo "   - SENDGRID_API_KEY"
echo "   - SENDER_EMAIL"
echo "   - SENDER_NAME"
echo "   - RECIPIENT_EMAIL"
echo "4. Redeploy your application"
echo ""
echo "📖 For detailed instructions, see: SENDGRID_CONFIGURATION_GUIDE.md"
echo ""
echo "🧪 To test the email configuration:"
echo "   curl -X POST https://your-app.vercel.app/api/test-alert-email \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"email\": \"your-email@example.com\", \"riskScore\": 85}'"
echo ""