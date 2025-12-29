#!/bin/bash
# CodeArtifact Workflow for Saafe MVP

set -e

echo "🔧 Saafe MVP - CodeArtifact Workflow"
echo "===================================="

# Configuration
DOMAIN="saafeai"
DOMAIN_OWNER="691595239825"
REPOSITORY="saafe"
REGION="eu-west-1"

echo "📋 Configuration:"
echo "   Domain: $DOMAIN"
echo "   Repository: $REPOSITORY"
echo "   Region: $REGION"
echo "   Domain Owner: $DOMAIN_OWNER"
echo ""

# Step 1: Login to CodeArtifact
echo "🔐 Step 1: Logging into CodeArtifact..."
aws codeartifact login --tool pip --repository $REPOSITORY --domain $DOMAIN --domain-owner $DOMAIN_OWNER --region $REGION

if [ $? -eq 0 ]; then
    echo "✅ Successfully logged into CodeArtifact"
else
    echo "❌ Failed to login to CodeArtifact"
    exit 1
fi

# Step 2: Test package installation
echo ""
echo "🧪 Step 2: Testing package installation..."
pip install --dry-run boto3 > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Package installation test successful"
else
    echo "❌ Package installation test failed"
    exit 1
fi

# Step 3: Install project dependencies
echo ""
echo "📦 Step 3: Installing project dependencies..."
if [ -f "requirements-codeartifact.txt" ]; then
    pip install -r requirements-codeartifact.txt
    echo "✅ Dependencies installed from CodeArtifact"
else
    echo "⚠️  requirements-codeartifact.txt not found, using regular requirements.txt"
    pip install -r requirements.txt
fi

# Step 4: Build Docker image (optional)
echo ""
read -p "🐳 Step 4: Build Docker image with CodeArtifact? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Building Docker image..."
    docker build -f Dockerfile-codeartifact -t saafe-mvp:codeartifact \
        --build-arg AWS_ACCOUNT_ID=$DOMAIN_OWNER \
        --build-arg AWS_REGION=$REGION \
        --build-arg CODEARTIFACT_DOMAIN=$DOMAIN \
        --build-arg CODEARTIFACT_REPO=$REPOSITORY \
        .
    
    if [ $? -eq 0 ]; then
        echo "✅ Docker image built successfully"
        echo "   Image: saafe-mvp:codeartifact"
        echo "   Run with: docker run -p 8501:8501 saafe-mvp:codeartifact"
    else
        echo "❌ Docker build failed"
    fi
fi

# Step 5: Test application
echo ""
read -p "🚀 Step 5: Test application locally? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting Saafe MVP application..."
    echo "   URL: http://localhost:8501"
    echo "   Press Ctrl+C to stop"
    streamlit run app.py
fi

echo ""
echo "✅ CodeArtifact workflow completed!"
echo ""
echo "🎯 Next steps:"
echo "   1. Your pip is now configured to use CodeArtifact"
echo "   2. Dependencies are installed from your private repository"
echo "   3. Docker image is ready for deployment"
echo "   4. Deploy to AWS using the updated deployment guide"
echo ""
echo "💡 Tips:"
echo "   • CodeArtifact tokens expire after 12 hours"
echo "   • Re-run this script when tokens expire"
echo "   • Use 'docker-compose -f docker-compose-codeartifact.yml up' for local testing"