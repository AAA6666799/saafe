#!/bin/bash

# SAAFE Lovable Dashboard AWS Deployment Script with ALB
# This script deploys the dashboard to AWS ECS Fargate with Application Load Balancer

set -e  # Exit on any error

# Configuration
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region $REGION)
REPO_NAME="saafe-lovable-dashboard"
IMAGE_TAG="latest"
CLUSTER_NAME="saafe-dashboard-cluster"
SERVICE_NAME="saafe-dashboard-service"
TASK_FAMILY="saafe-dashboard-task"
VPC_NAME="saafe-dashboard-vpc"
ALB_NAME="saafe-dashboard-alb"
TG_NAME="saafe-dashboard-tg"
EXECUTION_ROLE_NAME="SaafeDashboardExecutionRole"
TASK_ROLE_NAME="SaafeDashboardTaskRole"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error_exit "AWS CLI is not installed. Please install it first."
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed. Please install it first."
    fi

    # Check if AWS credentials are configured
    if ! aws sts get-caller-identity &> /dev/null; then
        error_exit "AWS credentials are not configured. Please run 'aws configure' first."
    fi

    log_success "Prerequisites check passed"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."

    # Navigate to saafe-lovable directory
    if [ ! -d "saafe-lovable" ]; then
        error_exit "saafe-lovable directory not found. Please run this script from the project root."
    fi

    cd saafe-lovable

    # Build Docker image
    log_info "Building Docker image..."
    docker build -t $REPO_NAME:$IMAGE_TAG .

    # Get ECR login token
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

    # Create ECR repository if it doesn't exist
    if ! aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION &> /dev/null; then
        log_info "Creating ECR repository..."
        aws ecr create-repository --repository-name $REPO_NAME --region $REGION
    fi

    # Tag and push image
    IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"
    docker tag $REPO_NAME:$IMAGE_TAG $IMAGE_URI
    docker push $IMAGE_URI

    cd ..
    log_success "Docker image built and pushed: $IMAGE_URI"
    echo $IMAGE_URI
}

# Create VPC infrastructure
create_vpc_infrastructure() {
    log_info "Creating VPC infrastructure..."

    # Create VPC
    VPC_ID=$(aws ec2 create-vpc --cidr-block 10.0.0.0/16 --region $REGION --query 'Vpc.VpcId' --output text)
    aws ec2 create-tags --resources $VPC_ID --tags Key=Name,Value=$VPC_NAME --region $REGION

    # Create subnets (2 public subnets in different AZs)
    SUBNET_1=$(aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.0.1.0/24 --availability-zone ${REGION}a --region $REGION --query 'Subnet.SubnetId' --output text)
    SUBNET_2=$(aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.0.2.0/24 --availability-zone ${REGION}b --region $REGION --query 'Subnet.SubnetId' --output text)

    # Create Internet Gateway
    IGW_ID=$(aws ec2 create-internet-gateway --region $REGION --query 'InternetGateway.InternetGatewayId' --output text)
    aws ec2 attach-internet-gateway --vpc-id $VPC_ID --internet-gateway-id $IGW_ID --region $REGION

    # Create route table
    RT_ID=$(aws ec2 create-route-table --vpc-id $VPC_ID --region $REGION --query 'RouteTable.RouteTableId' --output text)

    # Create route to internet
    aws ec2 create-route --route-table-id $RT_ID --destination-cidr-block 0.0.0.0/0 --gateway-id $IGW_ID --region $REGION

    # Associate subnets with route table
    aws ec2 associate-route-table --subnet-id $SUBNET_1 --route-table-id $RT_ID --region $REGION
    aws ec2 associate-route-table --subnet-id $SUBNET_2 --route-table-id $RT_ID --region $REGION

    # Enable auto-assign public IP for subnets
    aws ec2 modify-subnet-attribute --subnet-id $SUBNET_1 --map-public-ip-on-launch --region $REGION
    aws ec2 modify-subnet-attribute --subnet-id $SUBNET_2 --map-public-ip-on-launch --region $REGION

    log_success "VPC infrastructure created"
    echo "$VPC_ID $SUBNET_1 $SUBNET_2"
}

# Create security groups
create_security_groups() {
    VPC_ID=$1

    log_info "Creating security groups..."

    # ALB Security Group (allow HTTP/HTTPS from anywhere)
    ALB_SG=$(aws ec2 create-security-group \
        --group-name saafe-dashboard-alb-sg \
        --description "ALB Security Group for SAAFE Dashboard" \
        --vpc-id $VPC_ID \
        --region $REGION \
        --query 'GroupId' \
        --output text)

    aws ec2 authorize-security-group-ingress \
        --group-id $ALB_SG \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0 \
        --region $REGION

    aws ec2 authorize-security-group-ingress \
        --group-id $ALB_SG \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0 \
        --region $REGION

    # ECS Task Security Group (allow traffic from ALB only)
    TASK_SG=$(aws ec2 create-security-group \
        --group-name saafe-dashboard-task-sg \
        --description "ECS Task Security Group for SAAFE Dashboard" \
        --vpc-id $VPC_ID \
        --region $REGION \
        --query 'GroupId' \
        --output text)

    aws ec2 authorize-security-group-ingress \
        --group-id $TASK_SG \
        --protocol tcp \
        --port 8000 \
        --source-group $ALB_SG \
        --region $REGION

    log_success "Security groups created"
    echo "$ALB_SG $TASK_SG"
}

# Create Application Load Balancer
create_load_balancer() {
    VPC_ID=$1
    SUBNET_1=$2
    SUBNET_2=$3
    ALB_SG=$4

    log_info "Creating Application Load Balancer..."

    # Create ALB
    ALB_ARN=$(aws elbv2 create-load-balancer \
        --name $ALB_NAME \
        --subnets $SUBNET_1 $SUBNET_2 \
        --security-groups $ALB_SG \
        --scheme internet-facing \
        --type application \
        --region $REGION \
        --query 'LoadBalancers[0].LoadBalancerArn' \
        --output text)

    # Create target group
    TG_ARN=$(aws elbv2 create-target-group \
        --name $TG_NAME \
        --protocol HTTP \
        --port 8000 \
        --vpc-id $VPC_ID \
        --target-type ip \
        --health-check-path /api/fire-detection-data \
        --health-check-interval-seconds 30 \
        --health-check-timeout-seconds 5 \
        --healthy-threshold-count 2 \
        --unhealthy-threshold-count 2 \
        --region $REGION \
        --query 'TargetGroups[0].TargetGroupArn' \
        --output text)

    # Create listener
    aws elbv2 create-listener \
        --load-balancer-arn $ALB_ARN \
        --protocol HTTP \
        --port 80 \
        --default-actions Type=forward,TargetGroupArn=$TG_ARN \
        --region $REGION

    # Get ALB DNS name
    ALB_DNS=$(aws elbv2 describe-load-balancers \
        --load-balancer-arns $ALB_ARN \
        --region $REGION \
        --query 'LoadBalancers[0].DNSName' \
        --output text)

    log_success "Application Load Balancer created"
    echo "$ALB_ARN $TG_ARN $ALB_DNS"
}

# Create IAM roles
create_iam_roles() {
    log_info "Creating IAM roles..."

    # Create execution role
    EXECUTION_ROLE_ARN=$(aws iam create-role \
        --role-name $EXECUTION_ROLE_NAME \
        --assume-role-policy-document '{
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
        }' \
        --query 'Role.Arn' \
        --output text)

    # Attach execution role policy
    aws iam attach-role-policy \
        --role-name $EXECUTION_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

    # Create task role
    TASK_ROLE_ARN=$(aws iam create-role \
        --role-name $TASK_ROLE_NAME \
        --assume-role-policy-document '{
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
        }' \
        --query 'Role.Arn' \
        --output text)

    # Attach task role policies
    aws iam attach-role-policy \
        --role-name $TASK_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
    aws iam attach-role-policy \
        --role-name $TASK_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

    log_success "IAM roles created"
    echo "$EXECUTION_ROLE_ARN $TASK_ROLE_ARN"
}

# Create ECS cluster and service
create_ecs_service() {
    IMAGE_URI=$1
    SUBNET_1=$2
    SUBNET_2=$3
    TASK_SG=$4
    TG_ARN=$5
    EXECUTION_ROLE_ARN=$6
    TASK_ROLE_ARN=$7

    log_info "Creating ECS cluster and service..."

    # Create cluster
    aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $REGION

    # Register task definition
    TASK_DEF_ARN=$(aws ecs register-task-definition \
        --family $TASK_FAMILY \
        --network-mode awsvpc \
        --requires-compatibilities FARGATE \
        --cpu 256 \
        --memory 512 \
        --execution-role-arn $EXECUTION_ROLE_ARN \
        --task-role-arn $TASK_ROLE_ARN \
        --container-definitions "[
            {
                \"name\": \"saafe-dashboard\",
                \"image\": \"$IMAGE_URI\",
                \"portMappings\": [
                    {
                        \"containerPort\": 8000,
                        \"hostPort\": 8000,
                        \"protocol\": \"tcp\"
                    }
                ],
                \"logConfiguration\": {
                    \"logDriver\": \"awslogs\",
                    \"options\": {
                        \"awslogs-group\": \"/ecs/$TASK_FAMILY\",
                        \"awslogs-region\": \"$REGION\",
                        \"awslogs-stream-prefix\": \"ecs\"
                    }
                },
                \"essential\": true
            }
        ]" \
        --region $REGION \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text)

    # Create CloudWatch log group
    aws logs create-log-group --log-group-name /ecs/$TASK_FAMILY --region $REGION || true

    # Create service
    SERVICE_ARN=$(aws ecs create-service \
        --cluster $CLUSTER_NAME \
        --service-name $SERVICE_NAME \
        --task-definition $TASK_FAMILY \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_1,$SUBNET_2],securityGroups=[$TASK_SG],assignPublicIp=ENABLED}" \
        --load-balancers "targetGroupArn=$TG_ARN,containerName=saafe-dashboard,containerPort=8000" \
        --region $REGION \
        --query 'service.serviceArn' \
        --output text)

    log_success "ECS cluster and service created"
    echo "$SERVICE_ARN"
}

# Wait for service to be stable
wait_for_service() {
    log_info "Waiting for service to become stable..."

    # Wait up to 10 minutes for service to stabilize
    for i in {1..60}; do
        STATUS=$(aws ecs describe-services \
            --cluster $CLUSTER_NAME \
            --services $SERVICE_NAME \
            --region $REGION \
            --query 'services[0].serviceStatus' \
            --output text)

        if [ "$STATUS" = "ACTIVE" ]; then
            RUNNING_COUNT=$(aws ecs describe-services \
                --cluster $CLUSTER_NAME \
                --services $SERVICE_NAME \
                --region $REGION \
                --query 'services[0].runningCount' \
                --output text)

            if [ "$RUNNING_COUNT" -eq 1 ]; then
                log_success "Service is stable and running"
                return 0
            fi
        fi

        log_info "Waiting for service to stabilize... (attempt $i/60)"
        sleep 10
    done

    error_exit "Service failed to stabilize within 10 minutes"
}

# Cleanup function
cleanup() {
    log_warning "Starting cleanup process..."

    # Delete service
    if aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION &> /dev/null; then
        aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --desired-count 0 --region $REGION
        aws ecs delete-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --region $REGION --force
    fi

    # Delete task definition
    if aws ecs describe-task-definition --task-definition $TASK_FAMILY --region $REGION &> /dev/null; then
        aws ecs deregister-task-definition --task-definition $TASK_FAMILY --region $REGION
    fi

    # Delete cluster
    if aws ecs describe-cluster --cluster $CLUSTER_NAME --region $REGION &> /dev/null; then
        aws ecs delete-cluster --cluster $CLUSTER_NAME --region $REGION
    fi

    # Delete ALB and target group
    if [ ! -z "$ALB_ARN" ]; then
        # Delete listeners first
        LISTENER_ARNS=$(aws elbv2 describe-listeners --load-balancer-arn $ALB_ARN --region $REGION --query 'Listeners[].ListenerArn' --output text)
        for listener in $LISTENER_ARNS; do
            aws elbv2 delete-listener --listener-arn $listener --region $REGION
        done

        aws elbv2 delete-load-balancer --load-balancer-arn $ALB_ARN --region $REGION
    fi

    if [ ! -z "$TG_ARN" ]; then
        aws elbv2 delete-target-group --target-group-arn $TG_ARN --region $REGION
    fi

    # Delete security groups
    if [ ! -z "$ALB_SG" ]; then
        aws ec2 delete-security-group --group-id $ALB_SG --region $REGION || true
    fi

    if [ ! -z "$TASK_SG" ]; then
        aws ec2 delete-security-group --group-id $TASK_SG --region $REGION || true
    fi

    # Delete VPC infrastructure
    if [ ! -z "$VPC_ID" ]; then
        # Delete subnets
        if [ ! -z "$SUBNET_1" ]; then
            aws ec2 delete-subnet --subnet-id $SUBNET_1 --region $REGION || true
        fi
        if [ ! -z "$SUBNET_2" ]; then
            aws ec2 delete-subnet --subnet-id $SUBNET_2 --region $REGION || true
        fi

        # Delete route table associations and route table
        if [ ! -z "$RT_ID" ]; then
            aws ec2 delete-route-table --route-table-id $RT_ID --region $REGION || true
        fi

        # Detach and delete internet gateway
        if [ ! -z "$IGW_ID" ]; then
            aws ec2 detach-internet-gateway --vpc-id $VPC_ID --internet-gateway-id $IGW_ID --region $REGION || true
            aws ec2 delete-internet-gateway --internet-gateway-id $IGW_ID --region $REGION || true
        fi

        # Delete VPC
        aws ec2 delete-vpc --vpc-id $VPC_ID --region $REGION || true
    fi

    # Delete IAM roles
    if aws iam get-role --role-name $EXECUTION_ROLE_NAME &> /dev/null; then
        aws iam detach-role-policy --role-name $EXECUTION_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
        aws iam delete-role --role-name $EXECUTION_ROLE_NAME
    fi

    if aws iam get-role --role-name $TASK_ROLE_NAME &> /dev/null; then
        aws iam detach-role-policy --role-name $TASK_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
        aws iam detach-role-policy --role-name $TASK_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
        aws iam delete-role --role-name $TASK_ROLE_NAME
    fi

    # Delete ECR repository
    if aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION &> /dev/null; then
        aws ecr delete-repository --repository-name $REPO_NAME --region $REGION --force
    fi

    # Delete CloudWatch log group
    aws logs delete-log-group --log-group-name /ecs/$TASK_FAMILY --region $REGION || true

    log_success "Cleanup completed"
}

# Main deployment function
deploy() {
    log_info "Starting SAAFE Dashboard deployment with ALB..."

    # Check prerequisites
    check_prerequisites

    # Build and push image
    IMAGE_URI=$(build_and_push_image)

    # Create VPC infrastructure
    VPC_INFO=$(create_vpc_infrastructure)
    read VPC_ID SUBNET_1 SUBNET_2 <<< $VPC_INFO

    # Create security groups
    SG_INFO=$(create_security_groups $VPC_ID)
    read ALB_SG TASK_SG <<< $SG_INFO

    # Create load balancer
    ALB_INFO=$(create_load_balancer $VPC_ID $SUBNET_1 $SUBNET_2 $ALB_SG)
    read ALB_ARN TG_ARN ALB_DNS <<< $ALB_INFO

    # Create IAM roles
    ROLE_INFO=$(create_iam_roles)
    read EXECUTION_ROLE_ARN TASK_ROLE_ARN <<< $ROLE_INFO

    # Create ECS service
    SERVICE_ARN=$(create_ecs_service $IMAGE_URI $SUBNET_1 $SUBNET_2 $TASK_SG $TG_ARN $EXECUTION_ROLE_ARN $TASK_ROLE_ARN)

    # Wait for service to be stable
    wait_for_service

    log_success "Deployment completed successfully!"
    log_info "Dashboard is accessible at: http://$ALB_DNS"
    log_info "It may take a few minutes for the ALB to route traffic to the service."

    # Save deployment info
    cat > deployment_info.txt << EOF
SAAFE Dashboard Deployment Information
=====================================

Dashboard URL: http://$ALB_DNS
Region: $REGION
Cluster: $CLUSTER_NAME
Service: $SERVICE_NAME
Task Family: $TASK_FAMILY
VPC ID: $VPC_ID
ALB ARN: $ALB_ARN
Target Group ARN: $TG_ARN

To check service status:
aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION

To view logs:
aws logs tail /ecs/$TASK_FAMILY --region $REGION --follow

To cleanup all resources:
./deploy_saafe_dashboard_alb.sh cleanup
EOF

    log_info "Deployment information saved to deployment_info.txt"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "cleanup")
        cleanup
        ;;
    *)
        echo "Usage: $0 [deploy|cleanup]"
        echo "  deploy  - Deploy the SAAFE dashboard with ALB (default)"
        echo "  cleanup - Clean up all deployed resources"
        exit 1
        ;;
esac