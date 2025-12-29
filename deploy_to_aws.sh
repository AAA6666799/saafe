#!/bin/bash
# Saafe MVP - AWS ECS Deployment Script
# Deploys the Saafe Fire Detection System to AWS ECS Fargate

set -e

# Configuration
PROJECT_NAME="saafe-mvp"
AWS_REGION="eu-west-1"
AWS_ACCOUNT_ID="691595239825"
ECR_REPOSITORY="${PROJECT_NAME}"
ECS_CLUSTER="${PROJECT_NAME}-cluster"
ECS_SERVICE="${PROJECT_NAME}-service"
TASK_DEFINITION="${PROJECT_NAME}-task"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Saafe MVP AWS Deployment${NC}"
echo "=============================================="
echo "Project: ${PROJECT_NAME}"
echo "Region: ${AWS_REGION}"
echo "Account: ${AWS_ACCOUNT_ID}"
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

if ! command_exists docker; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

# Verify AWS credentials
echo -e "${YELLOW}🔐 Verifying AWS credentials...${NC}"
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}❌ AWS credentials not configured. Run 'aws configure' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Step 1: Create ECR repository if it doesn't exist
echo -e "\n${YELLOW}📦 Setting up ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} --region ${AWS_REGION} > /dev/null 2>&1; then
    echo "Creating ECR repository: ${ECR_REPOSITORY}"
    aws ecr create-repository \
        --repository-name ${ECR_REPOSITORY} \
        --region ${AWS_REGION} \
        --image-scanning-configuration scanOnPush=true
    echo -e "${GREEN}✅ ECR repository created${NC}"
else
    echo -e "${GREEN}✅ ECR repository already exists${NC}"
fi

# Step 2: Login to ECR
echo -e "\n${YELLOW}🔐 Logging into ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
echo -e "${GREEN}✅ ECR login successful${NC}"

# Step 3: Build Docker image
echo -e "\n${YELLOW}🏗️  Building Docker image...${NC}"
docker build -f Dockerfile-codeartifact -t ${PROJECT_NAME}:latest .

# Step 4: Tag and push image to ECR
echo -e "\n${YELLOW}🏷️  Tagging and pushing image...${NC}"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"
docker tag ${PROJECT_NAME}:latest ${ECR_URI}:latest
docker push ${ECR_URI}:latest
echo -e "${GREEN}✅ Image pushed to ECR${NC}"

# Step 5: Create ECS cluster if it doesn't exist
echo -e "\n${YELLOW}🐳 Setting up ECS cluster...${NC}"
if ! aws ecs describe-clusters --clusters ${ECS_CLUSTER} --region ${AWS_REGION} | grep -q "ACTIVE"; then
    echo "Creating ECS cluster: ${ECS_CLUSTER}"
    aws ecs create-cluster \
        --cluster-name ${ECS_CLUSTER} \
        --capacity-providers FARGATE FARGATE_SPOT \
        --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
        --region ${AWS_REGION}
    echo -e "${GREEN}✅ ECS cluster created${NC}"
else
    echo -e "${GREEN}✅ ECS cluster already exists${NC}"
fi

# Step 6: Create IAM execution role if it doesn't exist
echo -e "\n${YELLOW}🔑 Setting up IAM roles...${NC}"
EXECUTION_ROLE_NAME="SaafeECSExecutionRole"
EXECUTION_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${EXECUTION_ROLE_NAME}"

if ! aws iam get-role --role-name ${EXECUTION_ROLE_NAME} > /dev/null 2>&1; then
    echo "Creating ECS execution role..."
    
    # Create trust policy
    cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    aws iam create-role \
        --role-name ${EXECUTION_ROLE_NAME} \
        --assume-role-policy-document file://trust-policy.json

    aws iam attach-role-policy \
        --role-name ${EXECUTION_ROLE_NAME} \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

    aws iam attach-role-policy \
        --role-name ${EXECUTION_ROLE_NAME} \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

    rm trust-policy.json
    echo -e "${GREEN}✅ IAM execution role created${NC}"
else
    echo -e "${GREEN}✅ IAM execution role already exists${NC}"
fi

# Step 7: Create CloudWatch log group
echo -e "\n${YELLOW}📊 Setting up CloudWatch logs...${NC}"
LOG_GROUP="/ecs/${PROJECT_NAME}"
if ! aws logs describe-log-groups --log-group-name-prefix ${LOG_GROUP} --region ${AWS_REGION} | grep -q ${LOG_GROUP}; then
    aws logs create-log-group --log-group-name ${LOG_GROUP} --region ${AWS_REGION}
    echo -e "${GREEN}✅ CloudWatch log group created${NC}"
else
    echo -e "${GREEN}✅ CloudWatch log group already exists${NC}"
fi

# Step 8: Create task definition
echo -e "\n${YELLOW}📋 Creating ECS task definition...${NC}"
cat > task-definition.json << EOF
{
  "family": "${TASK_DEFINITION}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "3072",
  "executionRoleArn": "${EXECUTION_ROLE_ARN}",
  "taskRoleArn": "${EXECUTION_ROLE_ARN}",
  "containerDefinitions": [
    {
      "name": "saafe-container",
      "image": "${ECR_URI}:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "${LOG_GROUP}",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "environment": [
        {
          "name": "AWS_DEFAULT_REGION",
          "value": "${AWS_REGION}"
        },
        {
          "name": "ENV",
          "value": "production"
        }
      ],
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8501/_stcore/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
EOF

# Register task definition
aws ecs register-task-definition \
    --cli-input-json file://task-definition.json \
    --region ${AWS_REGION}

rm task-definition.json
echo -e "${GREEN}✅ Task definition registered${NC}"

# Step 9: Get default VPC and subnets
echo -e "\n${YELLOW}🌐 Getting VPC configuration...${NC}"
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query "Vpcs[0].VpcId" --output text --region ${AWS_REGION})
SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=${VPC_ID}" --query "Subnets[*].SubnetId" --output text --region ${AWS_REGION})
SUBNET_ARRAY=(${SUBNET_IDS})

echo "VPC ID: ${VPC_ID}"
echo "Subnets: ${SUBNET_IDS}"

# Step 10: Create security group
echo -e "\n${YELLOW}🔒 Setting up security group...${NC}"
SECURITY_GROUP_NAME="${PROJECT_NAME}-sg"
SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=${SECURITY_GROUP_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
    --query "SecurityGroups[0].GroupId" \
    --output text \
    --region ${AWS_REGION} 2>/dev/null || echo "None")

if [ "${SECURITY_GROUP_ID}" = "None" ]; then
    echo "Creating security group..."
    SECURITY_GROUP_ID=$(aws ec2 create-security-group \
        --group-name ${SECURITY_GROUP_NAME} \
        --description "Security group for Saafe MVP" \
        --vpc-id ${VPC_ID} \
        --query "GroupId" \
        --output text \
        --region ${AWS_REGION})

    # Allow inbound traffic on port 8501
    aws ec2 authorize-security-group-ingress \
        --group-id ${SECURITY_GROUP_ID} \
        --protocol tcp \
        --port 8501 \
        --cidr 0.0.0.0/0 \
        --region ${AWS_REGION}

    echo -e "${GREEN}✅ Security group created: ${SECURITY_GROUP_ID}${NC}"
else
    echo -e "${GREEN}✅ Security group already exists: ${SECURITY_GROUP_ID}${NC}"
fi

# Step 11: Create or update ECS service
echo -e "\n${YELLOW}🚀 Deploying ECS service...${NC}"
if aws ecs describe-services --cluster ${ECS_CLUSTER} --services ${ECS_SERVICE} --region ${AWS_REGION} | grep -q "ACTIVE"; then
    echo "Updating existing service..."
    aws ecs update-service \
        --cluster ${ECS_CLUSTER} \
        --service ${ECS_SERVICE} \
        --task-definition ${TASK_DEFINITION} \
        --region ${AWS_REGION}
else
    echo "Creating new service..."
    aws ecs create-service \
        --cluster ${ECS_CLUSTER} \
        --service-name ${ECS_SERVICE} \
        --task-definition ${TASK_DEFINITION} \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_ARRAY[0]},${SUBNET_ARRAY[1]}],securityGroups=[${SECURITY_GROUP_ID}],assignPublicIp=ENABLED}" \
        --region ${AWS_REGION}
fi

echo -e "${GREEN}✅ ECS service deployed${NC}"

# Step 12: Wait for service to be stable
echo -e "\n${YELLOW}⏳ Waiting for service to stabilize...${NC}"
aws ecs wait services-stable \
    --cluster ${ECS_CLUSTER} \
    --services ${ECS_SERVICE} \
    --region ${AWS_REGION}

# Step 13: Get service information
echo -e "\n${YELLOW}📊 Getting service information...${NC}"
TASK_ARN=$(aws ecs list-tasks \
    --cluster ${ECS_CLUSTER} \
    --service-name ${ECS_SERVICE} \
    --query "taskArns[0]" \
    --output text \
    --region ${AWS_REGION})

if [ "${TASK_ARN}" != "None" ]; then
    PUBLIC_IP=$(aws ecs describe-tasks \
        --cluster ${ECS_CLUSTER} \
        --tasks ${TASK_ARN} \
        --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" \
        --output text \
        --region ${AWS_REGION} | xargs -I {} aws ec2 describe-network-interfaces \
        --network-interface-ids {} \
        --query "NetworkInterfaces[0].Association.PublicIp" \
        --output text \
        --region ${AWS_REGION})

    echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
    echo "=============================================="
    echo -e "${BLUE}📍 Application URL: http://${PUBLIC_IP}:8501${NC}"
    echo -e "${BLUE}🔍 CloudWatch Logs: https://console.aws.amazon.com/cloudwatch/home?region=${AWS_REGION}#logsV2:log-groups/log-group/${LOG_GROUP//\//%2F}${NC}"
    echo -e "${BLUE}🐳 ECS Service: https://console.aws.amazon.com/ecs/home?region=${AWS_REGION}#/clusters/${ECS_CLUSTER}/services/${ECS_SERVICE}/details${NC}"
    echo "=============================================="
else
    echo -e "${YELLOW}⚠️  Service deployed but task information not available yet${NC}"
fi

echo -e "\n${GREEN}✅ Deployment script completed!${NC}"