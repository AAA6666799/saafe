#!/bin/bash

# SAAFE Dashboard AWS Deployment Script
# This script deploys the SAAFE dashboard to AWS S3 + CloudFront for global access

set -e

echo "🚀 SAAFE Dashboard AWS Deployment"
echo "================================="

# Configuration
BUCKET_NAME="saafe-dashboard-$(date +%s)"
REGION="us-east-1"  # Required for CloudFront
DISTRIBUTION_NAME="SAAFE Dashboard"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Please install it first:${NC}"
    echo "curl 'https://awscli.amazonaws.com/AWSCLIV2.pkg' -o 'AWSCLIV2.pkg'"
    echo "sudo installer -pkg AWSCLIV2.pkg -target /"
    exit 1
fi

# Check if AWS is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not configured. Please run:${NC}"
    echo "aws configure"
    exit 1
fi

echo -e "${GREEN}✅ AWS CLI configured${NC}"

# Build the dashboard
echo -e "${YELLOW}📦 Building dashboard...${NC}"
cd saafe-lovable
npm install
npm run build
cd ..

# Create S3 bucket
echo -e "${YELLOW}🪣 Creating S3 bucket: $BUCKET_NAME${NC}"
aws s3 mb s3://$BUCKET_NAME --region $REGION

# Configure bucket for static website hosting
echo -e "${YELLOW}🌐 Configuring static website hosting...${NC}"
aws s3 website s3://$BUCKET_NAME --index-document index.html --error-document index.html

# Upload files
echo -e "${YELLOW}📤 Uploading files to S3...${NC}"
aws s3 sync saafe-lovable/dist/ s3://$BUCKET_NAME --delete

# Set bucket policy for public read access
echo -e "${YELLOW}🔓 Setting bucket policy for public access...${NC}"
cat > bucket-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::$BUCKET_NAME/*"
        }
    ]
}
EOF

aws s3api put-bucket-policy --bucket $BUCKET_NAME --policy file://bucket-policy.json
rm bucket-policy.json

# Create CloudFront distribution
echo -e "${YELLOW}🌍 Creating CloudFront distribution for global access...${NC}"
cat > cloudfront-config.json << EOF
{
    "CallerReference": "saafe-dashboard-$(date +%s)",
    "Comment": "$DISTRIBUTION_NAME",
    "DefaultRootObject": "index.html",
    "Origins": {
        "Quantity": 1,
        "Items": [
            {
                "Id": "S3-$BUCKET_NAME",
                "DomainName": "$BUCKET_NAME.s3-website-$REGION.amazonaws.com",
                "CustomOriginConfig": {
                    "HTTPPort": 80,
                    "HTTPSPort": 443,
                    "OriginProtocolPolicy": "http-only"
                }
            }
        ]
    },
    "DefaultCacheBehavior": {
        "TargetOriginId": "S3-$BUCKET_NAME",
        "ViewerProtocolPolicy": "redirect-to-https",
        "MinTTL": 0,
        "ForwardedValues": {
            "QueryString": false,
            "Cookies": {
                "Forward": "none"
            }
        }
    },
    "CustomErrorResponses": {
        "Quantity": 1,
        "Items": [
            {
                "ErrorCode": 404,
                "ResponsePagePath": "/index.html",
                "ResponseCode": "200",
                "ErrorCachingMinTTL": 300
            }
        ]
    },
    "Enabled": true,
    "PriceClass": "PriceClass_All"
}
EOF

DISTRIBUTION_ID=$(aws cloudfront create-distribution --distribution-config file://cloudfront-config.json --query 'Distribution.Id' --output text)
rm cloudfront-config.json

# Get CloudFront domain
CLOUDFRONT_DOMAIN=$(aws cloudfront get-distribution --id $DISTRIBUTION_ID --query 'Distribution.DomainName' --output text)

echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo "================================="
echo -e "${GREEN}✅ S3 Bucket:${NC} $BUCKET_NAME"
echo -e "${GREEN}✅ S3 Website URL:${NC} http://$BUCKET_NAME.s3-website-$REGION.amazonaws.com"
echo -e "${GREEN}✅ CloudFront Distribution ID:${NC} $DISTRIBUTION_ID"
echo -e "${GREEN}✅ Global CDN URL:${NC} https://$CLOUDFRONT_DOMAIN"
echo ""
echo -e "${YELLOW}⏳ Note: CloudFront deployment takes 15-20 minutes to propagate globally${NC}"
echo -e "${YELLOW}💡 Use the CloudFront URL for production - it's faster and has SSL${NC}"
echo ""
echo "🌍 Your SAAFE dashboard is now accessible worldwide!"