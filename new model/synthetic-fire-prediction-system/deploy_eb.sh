#!/bin/bash

# Fire Detection Dashboard Deployment Script for Elastic Beanstalk
# This script deploys the Streamlit dashboard to AWS Elastic Beanstalk

set -e  # Exit on any error

echo "🚀 Starting Fire Detection Dashboard Deployment to Elastic Beanstalk..."

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

# Initialize Elastic Beanstalk application if it doesn't exist
if ! eb list | grep -q $APP_NAME; then
    echo "🏗️ Creating new Elastic Beanstalk application..."
    eb init -p "Python 3.9" $APP_NAME --region $REGION
else
    echo "✅ Application already exists"
fi

# Create environment and deploy
echo "📤 Deploying application..."
eb create $ENV_NAME --instance-type t3.small --region $REGION --vpc

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