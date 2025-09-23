#!/bin/bash

# Fire Detection Dashboard Deployment Script
# This script deploys the Streamlit dashboard to AWS ECS for public access

set -e  # Exit on any error

echo "🚀 Starting Fire Detection Dashboard Deployment..."

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

# Variables
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY="fire-detection-dashboard"
CLUSTER_NAME="fire-detection-cluster"
TASK_FAMILY="fire-detection-dashboard"
SERVICE_NAME="fire-detection-dashboard-service"
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
docker build -t $ECR_REPOSITORY .

# Tag image for ECR
echo "🏷️ Tagging image..."
docker tag $ECR_REPOSITORY:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# Push image to ECR
echo "📤 Pushing image to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# Create ECS cluster if it doesn't exist
echo "🌐 Creating ECS cluster..."
aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $REGION 2>/dev/null || echo "Cluster already exists"

# Create task definition
echo "📝 Creating task definition..."
cat > task-definition.json << EOF
{
    "family": "$TASK_FAMILY",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "512",
    "memory": "1024",
    "executionRoleArn": "arn:aws:iam::$ACCOUNT_ID:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::$ACCOUNT_ID:role/fire-detection-dashboard-role",
    "containerDefinitions": [
        {
            "name": "fire-detection-dashboard",
            "image": "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG",
            "portMappings": [
                {
                    "containerPort": 8501,
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

# Create security group for the service
echo "🛡️ Creating security group..."
VPC_ID=$(aws ec2 describe-vpcs --region $REGION --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws ec2 create-security-group --group-name fire-detection-dashboard-sg --description "Security group for fire detection dashboard" --vpc-id $VPC_ID --region $REGION --query 'GroupId' --output text 2>/dev/null || aws ec2 describe-security-groups --filters "Name=group-name,Values=fire-detection-dashboard-sg" --region $REGION --query 'SecurityGroups[0].GroupId' --output text)

# Add inbound rule for port 8501
echo "🔓 Configuring security group rules..."
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8501 --cidr 0.0.0.0/0 --region $REGION 2>/dev/null || echo "Rule already exists"

# Create ECS service
echo "サービ Creating ECS service..."
SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --region $REGION --query 'Subnets[*].SubnetId' --output text | head -n 1)

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
echo "   2. Get the public IP address:"
echo "      aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks \$(aws ecs list-tasks --cluster $CLUSTER_NAME --query 'taskArns[0]' --output text) --region $REGION --query 'tasks[0].containers[0].networkInterfaces[0].privateIpv4Address' --output text"
echo "   3. Access the dashboard at: http://<PUBLIC_IP>:8501"

echo "🧹 Cleaning up temporary files..."
rm -f task-definition.json

echo "🎉 Deployment script finished!"