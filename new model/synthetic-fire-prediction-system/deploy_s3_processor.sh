#!/bin/bash
# Deployment script for S3 Data Processor Lambda function

set -e

# Configuration
PROJECT_NAME="saafe-fire-detection"
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="691595239825"
LAMBDA_ROLE_NAME="SaafeLambdaExecutionRole"
FUNCTION_NAME="saafe-s3-data-processor"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting S3 Data Processor Lambda Deployment${NC}"
echo "=============================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "\n${YELLOW}📋 Checking prerequisites...${NC}"

if ! command_exists aws; then
    echo -e "${RED}❌ AWS CLI not found. Please install AWS CLI first.${NC}"
    exit 1
fi

# Verify AWS credentials
echo -e "${YELLOW}🔐 Verifying AWS credentials...${NC}"
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}❌ AWS credentials not configured. Run 'aws configure' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Get the role ARN
LAMBDA_ROLE_ARN=$(aws iam get-role --role-name ${LAMBDA_ROLE_NAME} --query 'Role.Arn' --output text)
echo "Lambda Role ARN: ${LAMBDA_ROLE_ARN}"

# Step 1: Create deployment package
echo -e "\n${YELLOW}📦 Creating deployment package...${NC}"

# Create temporary directory
rm -rf /tmp/lambda_package
mkdir -p /tmp/lambda_package

# Copy function code
cp "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system/src/aws/lambda/s3_data_processor.py" /tmp/lambda_package/

# Download pre-compiled dependencies for Lambda
echo "Downloading pre-compiled dependencies..."
cd /tmp/lambda_package

# Create a requirements.txt file
cat > requirements.txt << EOF
pandas==1.3.5
numpy==1.21.6
EOF

# Install dependencies using a more compatible approach
pip install -r requirements.txt -t . --platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.9 -i https://pypi.org/simple/

# Remove unnecessary files
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete

# Create zip file
zip -r /tmp/s3_data_processor.zip .

echo -e "${GREEN}✅ Deployment package created${NC}"

# Step 2: Deploy Lambda function
echo -e "\n${YELLOW}☁️  Deploying S3 Data Processor Lambda function...${NC}"

if aws lambda get-function --function-name ${FUNCTION_NAME} > /dev/null 2>&1; then
    echo "Updating existing function..."
    aws lambda update-function-code \
        --function-name ${FUNCTION_NAME} \
        --zip-file fileb:///tmp/s3_data_processor.zip \
        --region ${AWS_REGION}
else
    echo "Creating new function..."
    aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --runtime python3.9 \
        --role ${LAMBDA_ROLE_ARN} \
        --handler s3_data_processor.lambda_handler \
        --zip-file fileb:///tmp/s3_data_processor.zip \
        --region ${AWS_REGION} \
        --timeout 900 \
        --memory-size 1024
fi

echo -e "${GREEN}✅ S3 Data Processor deployed${NC}"

# Step 3: Set up S3 event trigger
echo -e "\n${YELLOW}⏰ Setting up S3 event trigger...${NC}"

# Add S3 trigger as target
# First, get existing notification configuration
EXISTING_CONFIG=$(aws s3api get-bucket-notification-configuration --bucket data-collector-of-first-device)
echo "Existing configuration: $EXISTING_CONFIG"

# Update the configuration to include our new function
aws s3api put-bucket-notification-configuration \
    --bucket data-collector-of-first-device \
    --notification-configuration '{
        "LambdaFunctionConfigurations": [
            {
                "Id": "thermal-to-lambda",
                "LambdaFunctionArn": "arn:aws:lambda:us-east-1:691595239825:function:frame-feature-builder",
                "Events": ["s3:ObjectCreated:*"]
            },
            {
                "Id": "s3-data-processor",
                "LambdaFunctionArn": "arn:aws:lambda:us-east-1:691595239825:function:saafe-s3-data-processor",
                "Events": ["s3:ObjectCreated:*"]
            }
        ]
    }'

# Give S3 permission to invoke Lambda
aws lambda add-permission \
    --function-name ${FUNCTION_NAME} \
    --statement-id s3-data-processor-trigger \
    --action lambda:InvokeFunction \
    --principal s3.amazonaws.com \
    --source-arn arn:aws:s3:::data-collector-of-first-device \
    --region ${AWS_REGION}

echo -e "${GREEN}✅ S3 event trigger configured${NC}"

# Step 4: Display deployment information
echo -e "\n${GREEN}🎉 S3 Data Processor Deployment Completed Successfully!${NC}"
echo "=============================================="
echo -e "${BLUE}Deployed Function:${NC}"
echo "  - ${FUNCTION_NAME}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  - Timeout: 900 seconds"
echo "  - Memory: 1024 MB"
echo "  - Trigger: S3 object creation events"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Monitor CloudWatch logs for any issues"
echo "  2. Test with sample S3 events"
echo "  3. Verify alerts are being sent correctly"
echo "=============================================="

echo -e "\n${GREEN}✅ Deployment script completed!${NC}"