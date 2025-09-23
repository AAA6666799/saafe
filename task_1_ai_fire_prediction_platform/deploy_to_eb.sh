#!/bin/bash

# Deployment script for Saafe Fire Detection Dashboard to AWS Elastic Beanstalk

set -e  # Exit on any error

echo "🚀 Starting Saafe Fire Detection Dashboard Deployment to AWS Elastic Beanstalk..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null
then
    echo "❌ AWS CLI is not installed. Please install AWS CLI and configure credentials."
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null
then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

# Variables
APP_NAME="saafe-fire-dashboard"
ENV_NAME="saafe-fire-dashboard-env"
REGION="us-east-1"
VERSION_LABEL="v$(date +%Y%m%d-%H%M%S)"

echo "📋 Deployment Configuration:"
echo "   Application: $APP_NAME"
echo "   Environment: $ENV_NAME"
echo "   Region: $REGION"
echo "   Version: $VERSION_LABEL"

# Change to the dashboard directory
cd "/Volumes/Ajay/saafe copy 3/task_1_synthetic_fire_system"

# Create application zip if it doesn't exist
if [ ! -f "saafe-fire-dashboard.zip" ]; then
    echo "📦 Creating deployment package..."
    
    # Create a clean zip file without macOS metadata files
    zip -r saafe-fire-dashboard.zip \
        dashboard.py \
        dashboard_requirements.txt \
        run_dashboard.py \
        synthetic_fire_system \
        -x "*/.*" "*.pyc" "__pycache__/*" "*/__pycache__/*" "._*" "*.DS_Store"
else
    echo "📦 Using existing deployment package..."
fi

# Check if application exists, create if it doesn't
echo "🔍 Checking if application exists..."
if aws elasticbeanstalk describe-applications --application-names $APP_NAME --region $REGION 2>/dev/null | grep -q "$APP_NAME"; then
    echo "✅ Application already exists"
else
    echo "🏗️ Creating application..."
    aws elasticbeanstalk create-application --application-name $APP_NAME --region $REGION
fi

# Upload the zip file to S3 (EB will automatically create the application version)
echo "☁️ Uploading to S3..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET_NAME="elasticbeanstalk-$REGION-$ACCOUNT_ID"
aws s3 cp saafe-fire-dashboard.zip s3://$BUCKET_NAME/saafe-fire-dashboard/$VERSION_LABEL.zip --region $REGION

# Create application version from S3
echo "🔖 Creating application version..."
aws elasticbeanstalk create-application-version \
    --application-name $APP_NAME \
    --version-label $VERSION_LABEL \
    --source-bundle S3Bucket=$BUCKET_NAME,S3Key="saafe-fire-dashboard/$VERSION_LABEL.zip" \
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
        --option-settings file://eb-dashboard-config.json
fi

echo "✅ Deployment initiated successfully!"
echo "🌐 Monitor deployment status with:"
echo "   aws elasticbeanstalk describe-environments --application-name $APP_NAME --environment-names $ENV_NAME --region $REGION"
echo "   aws elasticbeanstalk describe-events --application-name $APP_NAME --environment-name $ENV_NAME --region $REGION"

# Wait a moment and then show environment status
sleep 5
echo "📊 Current environment status:"
aws elasticbeanstalk describe-environments --application-name $APP_NAME --environment-names $ENV_NAME --region $REGION --query 'Environments[0].[Status, Health, CNAME]' --output table 2>/dev/null || echo "Environment status check failed (may still be initializing)"