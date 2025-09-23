#!/bin/bash

# Saafe Fire Detection Dashboard Deployment Script
# This script deploys the Streamlit dashboard to AWS ECS for public access

set -e  # Exit on any error

echo "🚀 Starting Saafe Fire Detection Dashboard Deployment..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null
then
    echo "❌ AWS CLI is not installed. Please install AWS CLI and configure credentials."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "❌ Docker is not installed. Please install Docker."
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null
then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

# Variables
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY="saafe-dashboard"
CLUSTER_NAME="saafe-cluster"
TASK_FAMILY="saafe-dashboard"
SERVICE_NAME="saafe-dashboard-service"
IMAGE_TAG="latest"

echo "📋 Deployment Configuration:"
echo "   AWS Account: $ACCOUNT_ID"
echo "   Region: $REGION"
echo "   ECR Repository: $ECR_REPOSITORY"
echo "   Cluster: $CLUSTER_NAME"
echo "   Task Family: $TASK_FAMILY"
echo "   Service: $SERVICE_NAME"

# Create ECR repository if it doesn't exist
echo "📦 Creating ECR repository..."
aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $ECR_REPOSITORY --region $REGION

# Login to ECR
echo "🔐 Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build Docker image
echo "🏗️ Building Docker image..."
docker build -f Dockerfile.dashboard -t $ECR_REPOSITORY .

# Tag image for ECR
echo "🏷️ Tagging image..."
docker tag $ECR_REPOSITORY:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# Push image to ECR
echo "📤 Pushing image to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# Create ECS cluster if it doesn't exist
echo "🌐 Creating ECS cluster..."
aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $REGION 2>/dev/null || echo "Cluster already exists"

# Create CloudWatch log group
echo "📊 Creating CloudWatch log group..."
aws logs create-log-group --log-group-name "/ecs/$TASK_FAMILY" --region $REGION 2>/dev/null || echo "Log group already exists"

# Create task execution role if it doesn't exist
echo "🔐 Creating task execution role..."
ROLE_NAME="SaafeDashboardExecutionRole"
ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text 2>/dev/null || echo "none")

if [ "$ROLE_ARN" == "none" ]; then
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

    # Create role
    aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file://trust-policy.json --region $REGION
    
    # Attach policies
    aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy --region $REGION
    aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess --region $REGION
    
    # Clean up
    rm trust-policy.json
    
    # Wait for role to be available
    sleep 10
    ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text --region $REGION)
fi

echo "Role ARN: $ROLE_ARN"

# Create task definition
echo "📝 Creating task definition..."
cat > task-definition.json << EOF
{
    "family": "$TASK_FAMILY",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "512",
    "memory": "1024",
    "executionRoleArn": "$ROLE_ARN",
    "containerDefinitions": [
        {
            "name": "saafe-dashboard",
            "image": "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG",
            "portMappings": [
                {
                    "containerPort": 8502,
                    "protocol": "tcp"
                }
            ],
            "essential": true,
            "environment": [
                {
                    "name": "AWS_DEFAULT_REGION",
                    "value": "$REGION"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/$TASK_FAMILY",
                    "awslogs-region": "$REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
EOF

# Register task definition
echo "💾 Registering task definition..."
aws ecs register-task-definition --cli-input-json file://task-definition.json --region $REGION

# Get default VPC
echo "🌐 Getting default VPC..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text --region $REGION)

# Get subnets
echo "🌐 Getting subnets..."
SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[*].SubnetId' --output text --region $REGION | tr '\t' ',')

# Create security group for the service
echo "🛡️ Creating security group..."
SG_ID=$(aws ec2 create-security-group --group-name saafe-dashboard-sg --description "Security group for Saafe dashboard" --vpc-id $VPC_ID --region $REGION --query 'GroupId' --output text 2>/dev/null || aws ec2 describe-security-groups --filters "Name=group-name,Values=saafe-dashboard-sg" --region $REGION --query 'SecurityGroups[0].GroupId' --output text)

# Add inbound rule for port 8502
echo "🔓 Configuring security group rules..."
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8502 --cidr 0.0.0.0/0 --region $REGION 2>/dev/null || echo "Rule already exists"

# Create ECS service
echo "サービ Creating ECS service..."
aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --task-definition $TASK_FAMILY \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --region $REGION 2>/dev/null || echo "Service already exists"

echo "✅ Deployment completed successfully!"

echo "🌐 To access your dashboard:"
echo "   1. Wait 2-3 minutes for the service to start"
echo "   2. Get the public IP address by running:"
echo "      aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks \$(aws ecs list-tasks --cluster $CLUSTER_NAME --query 'taskArns[0]' --output text) --region $REGION --query 'tasks[0].containers[0].networkInterfaces[0].privateIpv4Address' --output text"
echo "   3. Access the dashboard at: http://<PUBLIC_IP>:8502"

echo "🧹 Cleaning up temporary files..."
rm -f task-definition.json

echo "🎉 Deployment script finished!"