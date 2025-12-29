#!/bin/bash

# SAAFE Dashboard Elastic Beanstalk Deployment Script
# Deploys the dashboard without Docker using Node.js platform

set -e

echo "🚀 Starting SAAFE Dashboard Elastic Beanstalk Deployment"

# Configuration
APP_NAME="saafe-dashboard"
ENV_NAME="saafe-dashboard-prod"
REGION="us-east-1"
PLATFORM="64bit Amazon Linux 2023 v6.6.5 running Node.js 20"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if EB CLI is installed
if ! command -v eb &> /dev/null; then
    print_warning "EB CLI is not installed. Installing..."
    pip3 install awsebcli --upgrade --user
    export PATH=$PATH:~/.local/bin
fi

# Navigate to the project directory
cd saafe-lovable

print_status "Initializing Elastic Beanstalk application..."

# Initialize EB application if not already initialized
if [ ! -d ".elasticbeanstalk" ]; then
    echo "y" | eb init $APP_NAME --region $REGION --platform "$PLATFORM"
else
    print_status "EB application already initialized"
fi

print_status "Creating Elastic Beanstalk environment..."

# Create environment if it doesn't exist
if ! eb list | grep -q $ENV_NAME; then
    eb create $ENV_NAME \
        --instance-type t3.micro \
        --min-instances 1 \
        --max-instances 2
else
    print_status "Environment $ENV_NAME already exists"
fi

print_status "Setting up IAM roles for S3 access..."

# Create IAM role for EC2 instances (if not exists)
ROLE_NAME="saafe-dashboard-ec2-role"
INSTANCE_PROFILE_NAME="saafe-dashboard-instance-profile"

# Check if role exists
if ! aws iam get-role --role-name $ROLE_NAME &> /dev/null; then
    print_status "Creating IAM role for EC2 instances..."

    # Create trust policy
    cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    # Create the role
    aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document file://trust-policy.json

    # Attach S3 read-only policy
    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

    # Create instance profile
    aws iam create-instance-profile \
        --instance-profile-name $INSTANCE_PROFILE_NAME

    # Add role to instance profile
    aws iam add-role-to-instance-profile \
        --role-name $ROLE_NAME \
        --instance-profile-name $INSTANCE_PROFILE_NAME

    # Clean up
    rm trust-policy.json

    print_status "IAM role created successfully"
else
    print_status "IAM role already exists"
fi

print_status "Configuring environment variables..."

# Set environment variables for the EB environment
eb setenv NODE_ENV=production AWS_REGION=$REGION

print_status "Deploying application..."

# Deploy the application
eb deploy $ENV_NAME

print_status "Deployment completed successfully!"

# Get the application URL
URL=$(eb status $ENV_NAME | grep "CNAME" | awk '{print $2}')

print_status "Application deployed successfully!"
echo -e "${GREEN}Public URL: http://$URL${NC}"

print_status "To check status: eb status $ENV_NAME"
print_status "To view logs: eb logs $ENV_NAME"
print_status "To terminate: eb terminate $ENV_NAME"

echo "🎉 Deployment complete!"