#!/bin/bash

# SAAFE Dashboard Docker + AWS App Runner Deployment
# This approach uses Docker for containerization and AWS App Runner for hosting

set -e

echo "🚀 SAAFE Dashboard Docker + AWS App Runner Deployment"
echo "===================================================="

# Configuration
APP_NAME="saafe-dashboard"
REGION="us-east-1"
ECR_REPO_NAME="saafe-dashboard"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites ready${NC}"

# Build the application
echo -e "${YELLOW}📦 Building React frontend...${NC}"
cd saafe-lovable
npm install
npm run build
cd ..

echo -e "${GREEN}✅ Frontend built${NC}"

# Create ECR repository
echo -e "${YELLOW}🏗️ Creating ECR repository...${NC}"
aws ecr create-repository --repository-name $ECR_REPO_NAME --region $REGION || echo "Repository may already exist"

# Get ECR login token
echo -e "${YELLOW}🔐 Logging into ECR...${NC}"
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI

# Build Docker image
echo -e "${YELLOW}🐳 Building Docker image...${NC}"
docker build -t $ECR_REPO_NAME .

# Tag image for ECR
docker tag $ECR_REPO_NAME:latest $ECR_URI:latest

# Push to ECR
echo -e "${YELLOW}📤 Pushing image to ECR...${NC}"
docker push $ECR_URI:latest

# Create App Runner service configuration
echo -e "${YELLOW}⚙️ Creating App Runner service...${NC}"
cat > apprunner-config.json << EOF
{
  "ServiceName": "$APP_NAME",
  "SourceConfiguration": {
    "ImageRepository": {
      "ImageIdentifier": "$ECR_URI:latest",
      "ImageConfiguration": {
        "Port": "8000",
        "RuntimeEnvironmentVariables": {
          "NODE_ENV": "production",
          "AWS_REGION": "$REGION"
        }
      },
      "ImageRepositoryType": "ECR"
    },
    "AutoDeploymentsEnabled": false
  },
  "InstanceConfiguration": {
    "Cpu": "1 vCPU",
    "Memory": "2 GB"
  },
  "HealthCheckConfiguration": {
    "Protocol": "HTTP",
    "Path": "/api/fire-detection-data",
    "Interval": 10,
    "Timeout": 5,
    "HealthyThreshold": 1,
    "UnhealthyThreshold": 5
  }
}
EOF

# Create the App Runner service
echo -e "${YELLOW}🚀 Creating App Runner service...${NC}"
SERVICE_ARN=$(aws apprunner create-service --cli-input-json file://apprunner-config.json --query 'Service.ServiceArn' --output text)

# Wait for service to be running
echo -e "${YELLOW}⏳ Waiting for service to be ready...${NC}"
aws apprunner wait service-running --service-arn $SERVICE_ARN

# Get service URL
SERVICE_URL=$(aws apprunner describe-service --service-arn $SERVICE_ARN --query 'Service.ServiceUrl' --output text)

# Clean up temporary files
rm apprunner-config.json

echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo "================================="
echo -e "${GREEN}✅ Service Name:${NC} $APP_NAME"
echo -e "${GREEN}✅ Service ARN:${NC} $SERVICE_ARN"
echo -e "${GREEN}✅ Dashboard URL:${NC} https://$SERVICE_URL"
echo -e "${GREEN}✅ API Endpoint:${NC} https://$SERVICE_URL/api/fire-detection-data"
echo -e "${GREEN}✅ ECR Repository:${NC} $ECR_URI"
echo ""
echo -e "${BLUE}🌍 Your SAAFE dashboard is now live and accessible worldwide!${NC}"
echo ""
echo -e "${YELLOW}💡 Management Commands:${NC}"
echo "   - View service: aws apprunner describe-service --service-arn $SERVICE_ARN"
echo "   - Update service: aws apprunner start-deployment --service-arn $SERVICE_ARN"
echo "   - View logs: Check CloudWatch Logs in AWS Console"
echo "   - Delete service: aws apprunner delete-service --service-arn $SERVICE_ARN"
echo ""
echo -e "${YELLOW}📊 Monitoring:${NC}"
echo "   - AWS Console: https://console.aws.amazon.com/apprunner/"
echo "   - CloudWatch Logs: Available in AWS Console"
echo "   - Service Metrics: Available in App Runner Console"
echo ""
echo -e "${GREEN}🔥 Your fire detection system is now protecting users globally!${NC}"