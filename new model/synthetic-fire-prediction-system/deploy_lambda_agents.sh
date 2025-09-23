#!/bin/bash
# Deployment script for AWS Lambda agents

set -e

# Configuration
PROJECT_NAME="saafe-fire-detection"
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="691595239825"
LAMBDA_ROLE_NAME="SaafeLambdaExecutionRole"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Saafe Lambda Agents Deployment${NC}"
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

# Step 1: Create Lambda execution role if it doesn't exist
echo -e "\n${YELLOW}🔑 Setting up Lambda execution role...${NC}"

if ! aws iam get-role --role-name ${LAMBDA_ROLE_NAME} > /dev/null 2>&1; then
    echo "Creating Lambda execution role..."
    
    # Create trust policy
    cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    # Create the role
    aws iam create-role \
        --role-name ${LAMBDA_ROLE_NAME} \
        --assume-role-policy-document file://trust-policy.json

    # Attach basic execution policy
    aws iam attach-role-policy \
        --role-name ${LAMBDA_ROLE_NAME} \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

    # Attach SageMaker full access policy
    aws iam attach-role-policy \
        --role-name ${LAMBDA_ROLE_NAME} \
        --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

    # Attach SNS full access policy
    aws iam attach-role-policy \
        --role-name ${LAMBDA_ROLE_NAME} \
        --policy-arn arn:aws:iam::aws:policy/AmazonSNSFullAccess

    # Wait for role to be available
    sleep 10

    rm trust-policy.json
    echo -e "${GREEN}✅ Lambda execution role created${NC}"
else
    echo -e "${GREEN}✅ Lambda execution role already exists${NC}"
fi

# Get the role ARN
LAMBDA_ROLE_ARN=$(aws iam get-role --role-name ${LAMBDA_ROLE_NAME} --query 'Role.Arn' --output text)
echo "Lambda Role ARN: ${LAMBDA_ROLE_ARN}"

# Skip creating deployment packages since they already exist

# Step 2: Deploy monitoring agent Lambda function
echo -e "\n${YELLOW}☁️  Deploying monitoring agent Lambda function...${NC}"

FUNCTION_NAME="saafe-monitoring-agent"

if aws lambda get-function --function-name ${FUNCTION_NAME} > /dev/null 2>&1; then
    echo "Updating existing function..."
    aws lambda update-function-code \
        --function-name ${FUNCTION_NAME} \
        --zip-file fileb://monitoring_agent.zip \
        --region ${AWS_REGION}
else
    echo "Creating new function..."
    aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --runtime python3.9 \
        --role ${LAMBDA_ROLE_ARN} \
        --handler monitoring_agent.lambda_handler \
        --zip-file fileb://monitoring_agent.zip \
        --region ${AWS_REGION} \
        --timeout 30 \
        --memory-size 256
fi

echo -e "${GREEN}✅ Monitoring agent deployed${NC}"

# Step 3: Deploy response agent Lambda function
echo -e "\n${YELLOW}☁️  Deploying response agent Lambda function...${NC}"

FUNCTION_NAME="saafe-response-agent"

if aws lambda get-function --function-name ${FUNCTION_NAME} > /dev/null 2>&1; then
    echo "Updating existing function..."
    aws lambda update-function-code \
        --function-name ${FUNCTION_NAME} \
        --zip-file fileb://response_agent.zip \
        --region ${AWS_REGION}
else
    echo "Creating new function..."
    aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --runtime python3.9 \
        --role ${LAMBDA_ROLE_ARN} \
        --handler response_agent.lambda_handler \
        --zip-file fileb://response_agent.zip \
        --region ${AWS_REGION} \
        --timeout 30 \
        --memory-size 256
fi

echo -e "${GREEN}✅ Response agent deployed${NC}"

# Step 4: Deploy analysis agent Lambda function
echo -e "\n${YELLOW}☁️  Deploying analysis agent Lambda function...${NC}"

FUNCTION_NAME="saafe-analysis-agent"

if aws lambda get-function --function-name ${FUNCTION_NAME} > /dev/null 2>&1; then
    echo "Updating existing function..."
    aws lambda update-function-code \
        --function-name ${FUNCTION_NAME} \
        --zip-file fileb://analysis_agent.zip \
        --region ${AWS_REGION}
else
    echo "Creating new function..."
    aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --runtime python3.9 \
        --role ${LAMBDA_ROLE_ARN} \
        --handler analysis_agent.lambda_handler \
        --zip-file fileb://analysis_agent.zip \
        --region ${AWS_REGION} \
        --timeout 60 \
        --memory-size 512
fi

echo -e "${GREEN}✅ Analysis agent deployed${NC}"

# Step 5: Set up CloudWatch event triggers (example)
echo -e "\n${YELLOW}⏰ Setting up CloudWatch event triggers...${NC}"

# Create monitoring schedule (every 5 minutes)
RULE_NAME="saafe-monitoring-schedule"
if ! aws events describe-rule --name ${RULE_NAME} > /dev/null 2>&1; then
    aws events put-rule \
        --name ${RULE_NAME} \
        --schedule-expression "rate(5 minutes)" \
        --region ${AWS_REGION}
fi

# Add monitoring agent as target
aws events put-targets \
    --rule ${RULE_NAME} \
    --targets "Id"="1","Arn"="arn:aws:lambda:${AWS_REGION}:${AWS_ACCOUNT_ID}:function:saafe-monitoring-agent" \
    --region ${AWS_REGION}

# Give CloudWatch permission to invoke Lambda
aws lambda add-permission \
    --function-name saafe-monitoring-agent \
    --statement-id saafe-monitoring-schedule \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:${AWS_REGION}:${AWS_ACCOUNT_ID}:rule/${RULE_NAME}" \
    --region ${AWS_REGION}

echo -e "${GREEN}✅ CloudWatch triggers configured${NC}"

# Step 6: Display deployment information
echo -e "\n${GREEN}🎉 Lambda Agents Deployment Completed Successfully!${NC}"
echo "=============================================="
echo -e "${BLUE}Deployed Functions:${NC}"
echo "  - saafe-monitoring-agent"
echo "  - saafe-response-agent"
echo "  - saafe-analysis-agent"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Configure SNS topics for alerts"
echo "  2. Set up additional event triggers as needed"
echo "  3. Test the functions with sample events"
echo "  4. Monitor CloudWatch logs for any issues"
echo "=============================================="

echo -e "\n${GREEN}✅ Deployment script completed!${NC}"