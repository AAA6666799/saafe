#!/bin/bash

# Simple deployment script for AWS Elastic Beanstalk
# This script creates a clean deployment package and deploys to EB

set -e  # Exit on any error

echo "🚀 Starting clean deployment to AWS Elastic Beanstalk..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null
then
    echo "❌ AWS CLI is not installed. Please install AWS CLI and configure credentials."
    exit 1
fi

# Variables
APP_NAME="saafe-dashboard"
ENV_NAME="saafe-dashboard-env"
REGION="us-west-2"
VERSION_LABEL="v$(date +%Y%m%d-%H%M%S)"

echo "📋 Deployment Configuration:"
echo "   Application: $APP_NAME"
echo "   Environment: $ENV_NAME"
echo "   Region: $REGION"
echo "   Version: $VERSION_LABEL"

# Create clean deployment package (if not already created)
if [ ! -f "saafe-dashboard-clean.zip" ]; then
    echo "📦 Creating clean deployment package..."
    python3 create_clean_deployment.py
else
    echo "📦 Using existing clean deployment package..."
fi

# Check if application exists, create if it doesn't
echo "🔍 Checking if application exists..."
if aws elasticbeanstalk describe-applications --application-names $APP_NAME --region $REGION 2>/dev/null | grep -q "$APP_NAME"; then
    echo "✅ Application already exists"
else
    echo "🏗️ Creating application..."
    aws elasticbeanstalk create-application --application-name $APP_NAME --region $REGION
fi

# Upload application version
echo "📤 Uploading application version..."
aws elasticbeanstalk create-application-version \
    --application-name $APP_NAME \
    --version-label $VERSION_LABEL \
    --source-bundle S3Bucket="elasticbeanstalk-$REGION-$(aws sts get-caller-identity --query Account --output text)",S3Key="saafe-dashboard/$VERSION_LABEL.zip" \
    --region $REGION 2>/dev/null || echo "Will upload directly"

# Upload the zip file to S3 (EB will automatically create the application version)
echo "☁️ Uploading to S3..."
aws s3 cp saafe-dashboard-clean.zip s3://elasticbeanstalk-$REGION-$(aws sts get-caller-identity --query Account --output text)/saafe-dashboard/$VERSION_LABEL.zip --region $REGION

# Create application version from S3
echo "🔖 Creating application version..."
aws elasticbeanstalk create-application-version \
    --application-name $APP_NAME \
    --version-label $VERSION_LABEL \
    --source-bundle S3Bucket="elasticbeanstalk-$REGION-$(aws sts get-caller-identity --query Account --output text)",S3Key="saafe-dashboard/$VERSION_LABEL.zip" \
    --region $REGION

# Check if environment exists
echo "🔍 Checking if environment exists..."
if aws elasticbeanstalk describe-environments --application-name $APP_NAME --environment-names $ENV_NAME --region $REGION 2>/dev/null | grep -q "$ENV_NAME"; then
    echo "🔄 Updating existing environment..."
    aws elasticbeanstalk update-environment \
        --environment-name $ENV_NAME \
        --version-label $VERSION_LABEL \
        --region $REGION
else
    echo "🌿 Creating new environment..."
    aws elasticbeanstalk create-environment \
        --application-name $APP_NAME \
        --environment-name $ENV_NAME \
        --version-label $VERSION_LABEL \
        --solution-stack-name "64bit Amazon Linux 2023 v4.7.1 running Python 3.13" \
        --region $REGION \
        --option-settings file://eb-config-simple.json
fi

echo "✅ Deployment initiated successfully!"
echo "🌐 Monitor deployment status with:"
echo "   aws elasticbeanstalk describe-environments --application-name $APP_NAME --environment-names $ENV_NAME --region $REGION"
echo "   aws elasticbeanstalk describe-events --application-name $APP_NAME --environment-name $ENV_NAME --region $REGION"

# Wait a moment and then show environment status
sleep 5
echo "📊 Current environment status:"
aws elasticbeanstalk describe-environments --application-name $APP_NAME --environment-names $ENV_NAME --region $REGION --query 'Environments[0].[Status, Health]' --output table 2>/dev/null || echo "Environment status check failed (may still be initializing)"