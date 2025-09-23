#!/bin/bash

# Saafe Fire Detection System - AWS Deployment Script
# This script deploys the Saafe fire detection system to AWS using ECS

set -e  # Exit on any error

# Configuration
AWS_REGION="us-west-2"
ECS_CLUSTER_NAME="saafe-fire-detection-cluster"
ECS_SERVICE_NAME="saafe-fire-detection-service"
ECS_TASK_FAMILY="saafe-fire-detection-task"
ECR_REPOSITORY_NAME="saafe-fire-detection"
CONTAINER_NAME="saafe-fire-detection"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Saafe Fire Detection System AWS Deployment${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install it first.${NC}"
    exit 1
fi

# Configure AWS CLI (if not already configured)
echo -e "${YELLOW}🔧 Checking AWS configuration...${NC}"
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${YELLOW}Please configure your AWS credentials:${NC}"
    aws configure
fi

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}✅ Using AWS Account: $ACCOUNT_ID${NC}"

# Create ECR repository if it doesn't exist
echo -e "${YELLOW}🔧 Checking ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names $ECR_REPOSITORY_NAME --region $AWS_REGION &> /dev/null; then
    echo -e "${YELLOW}Creating ECR repository...${NC}"
    aws ecr create-repository --repository-name $ECR_REPOSITORY_NAME --region $AWS_REGION
    echo -e "${GREEN}✅ ECR repository created${NC}"
else
    echo -e "${GREEN}✅ ECR repository already exists${NC}"
fi

# Get ECR login
echo -e "${YELLOW}🔧 Logging into ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build Docker image
echo -e "${YELLOW}🔧 Building Docker image...${NC}"
cd ..
docker build -t $ECR_REPOSITORY_NAME:latest .

# Tag image for ECR
echo -e "${YELLOW}🔧 Tagging image for ECR...${NC}"
docker tag $ECR_REPOSITORY_NAME:latest $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:latest

# Push image to ECR
echo -e "${YELLOW}🔧 Pushing image to ECR...${NC}"
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:latest
echo -e "${GREEN}✅ Image pushed to ECR${NC}"

# Create ECS cluster if it doesn't exist
echo -e "${YELLOW}🔧 Checking ECS cluster...${NC}"
if ! aws ecs describe-clusters --clusters $ECS_CLUSTER_NAME --region $AWS_REGION &> /dev/null; then
    echo -e "${YELLOW}Creating ECS cluster...${NC}"
    aws ecs create-cluster --cluster-name $ECS_CLUSTER_NAME --region $AWS_REGION
    echo -e "${GREEN}✅ ECS cluster created${NC}"
else
    echo -e "${GREEN}✅ ECS cluster already exists${NC}"
fi

# Register task definition
echo -e "${YELLOW}🔧 Registering ECS task definition...${NC}"
cat > task-definition.json << EOF
{
  "family": "$ECS_TASK_FAMILY",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "3072",
  "executionRoleArn": "arn:aws:iam::$ACCOUNT_ID:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::$ACCOUNT_ID:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "$CONTAINER_NAME",
      "image": "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "STREAMLIT_SERVER_HEADLESS",
          "value": "true"
        },
        {
          "name": "STREAMLIT_SERVER_PORT",
          "value": "8501"
        },
        {
          "name": "STREAMLIT_SERVER_ADDRESS",
          "value": "0.0.0.0"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/$ECS_TASK_FAMILY",
          "awslogs-region": "$AWS_REGION",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF

aws ecs register-task-definition --cli-input-json file://task-definition.json --region $AWS_REGION
echo -e "${GREEN}✅ Task definition registered${NC}"

# Create CloudWatch log group
echo -e "${YELLOW}🔧 Creating CloudWatch log group...${NC}"
aws logs create-log-group --log-group-name "/ecs/$ECS_TASK_FAMILY" --region $AWS_REGION || echo -e "${YELLOW}Log group may already exist${NC}"

# Create ECS service
echo -e "${YELLOW}🔧 Creating ECS service...${NC}"
# First check if service exists
if aws ecs describe-services --cluster $ECS_CLUSTER_NAME --services $ECS_SERVICE_NAME --region $AWS_REGION &> /dev/null; then
    echo -e "${YELLOW}Service already exists, updating...${NC}"
    aws ecs update-service --cluster $ECS_CLUSTER_NAME --service $ECS_SERVICE_NAME --task-definition $ECS_TASK_FAMILY --desired-count 1 --region $AWS_REGION
else
    # For new service, we need to specify network configuration
    echo -e "${YELLOW}Creating new service (you may need to update network settings)...${NC}"
    echo -e "${RED}⚠️  Please update the following command with your VPC and subnet IDs${NC}"
    echo "aws ecs create-service \\"
    echo "  --cluster $ECS_CLUSTER_NAME \\"
    echo "  --service-name $ECS_SERVICE_NAME \\"
    echo "  --task-definition $ECS_TASK_FAMILY \\"
    echo "  --desired-count 1 \\"
    echo "  --launch-type FARGATE \\"
    echo "  --network-configuration \"awsvpcConfiguration={subnets=[subnet-xxxxxxxx,subnet-yyyyyyyy],securityGroups=[sg-zzzzzzzz],assignPublicIp=ENABLED}\" \\"
    echo "  --region $AWS_REGION"
fi

echo -e "${GREEN}✅ Deployment commands completed${NC}"
echo -e "${YELLOW}📝 Next steps:${NC}"
echo "1. Update the create-service command with your VPC and subnet IDs"
echo "2. Run the create-service command shown above"
echo "3. Configure your security groups to allow traffic on port 8501"
echo "4. Access your application using the public IP assigned to the ECS service"

echo -e "${GREEN}🎉 Saafe Fire Detection System deployment initiated!${NC}"