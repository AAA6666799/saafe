#!/bin/bash

# Final Fire Detection Dashboard Deployment Script for Elastic Beanstalk
# This script deploys the Streamlit dashboard to AWS Elastic Beanstalk with the simplest approach

set -e  # Exit on any error

echo "🚀 Starting Fire Detection Dashboard Deployment to Elastic Beanstalk (Final Version)..."

# Check if AWS CLI and EB CLI are installed
if ! command -v aws &> /dev/null
then
    echo "❌ AWS CLI is not installed. Please install AWS CLI and configure credentials."
    exit 1
fi

if ! command -v eb &> /dev/null
then
    echo "❌ EB CLI is not installed. Please install EB CLI."
    echo "Installation instructions: https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html"
    exit 1
fi

# Variables
APP_NAME="fire-detection-dashboard"
ENV_NAME="fire-detection-dashboard-env"
REGION="us-east-1"

echo "📋 Deployment Configuration:"
echo "   Application: $APP_NAME"
echo "   Environment: $ENV_NAME"
echo "   Region: $REGION"

# Clean hidden files
echo "🧹 Cleaning hidden files..."
find . -name "._*" -type f -delete

# Create a clean deployment package with only essential files
echo "📦 Creating clean deployment package..."
rm -f deploy-package.zip
zip deploy-package.zip \
  fire_detection_streamlit_dashboard.py \
  application.py \
  Procfile \
  requirements-dashboard.txt \
  .ebextensions/python.config

# Initialize Elastic Beanstalk application if it doesn't exist
if ! eb list | grep -q $APP_NAME; then
    echo "🏗️ Creating new Elastic Beanstalk application..."
    eb init -p "Python 3.9" $APP_NAME --region $REGION
else
    echo "✅ Application already exists"
fi

# Deploy using the clean package
echo "📤 Deploying application with clean package..."
eb deploy $ENV_NAME --region $REGION

echo "✅ Deployment completed successfully!"

echo "🌐 To access your dashboard:"
echo "   1. Wait 5-10 minutes for the environment to be created"
echo "   2. Get the URL:"
echo "      eb status"
echo "   3. Open the URL in your browser"

echo "🔄 To update the application after changes:"
echo "   eb deploy"

echo "🧹 To terminate the environment when no longer needed:"
echo "   eb terminate $ENV_NAME"

echo "🎉 Deployment script finished!"