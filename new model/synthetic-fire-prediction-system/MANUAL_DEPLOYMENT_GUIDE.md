# 🔧 Manual Dashboard Deployment Guide

## Overview
This guide provides step-by-step instructions for manually deploying the Fire Detection Dashboard to AWS for public access.

## Prerequisites
1. AWS CLI installed and configured
2. Docker installed (for containerization)
3. Active AWS account with appropriate permissions

## Deployment Steps

### 1. Prepare the Environment

#### Install Docker
If Docker is not installed:
- For macOS: Download Docker Desktop from https://www.docker.com/products/docker-desktop
- For Windows: Download Docker Desktop from https://www.docker.com/products/docker-desktop
- For Linux: Follow the instructions at https://docs.docker.com/engine/install/

#### Configure AWS CLI
Ensure AWS CLI is installed and configured:
```bash
aws configure
```

### 2. Build and Push Docker Image

#### Create ECR Repository
```bash
aws ecr create-repository --repository-name fire-detection-dashboard --region us-east-1
```

#### Login to ECR
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
```
Replace `YOUR_ACCOUNT_ID` with your actual AWS account ID.

#### Build Docker Image
```bash
cd "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system"
docker build -t fire-detection-dashboard .
```

#### Tag and Push Image
```bash
docker tag fire-detection-dashboard:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fire-detection-dashboard:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fire-detection-dashboard:latest
```

### 3. Deploy to ECS

#### Create ECS Cluster
```bash
aws ecs create-cluster --cluster-name fire-detection-cluster --region us-east-1
```

#### Create Task Definition
Create a file named `task-definition.json`:
```json
{
    "family": "fire-detection-dashboard",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "512",
    "memory": "1024",
    "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "fire-detection-dashboard",
            "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fire-detection-dashboard:latest",
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
                    "value": "us-east-1"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/fire-detection-dashboard",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
```

Register the task definition:
```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json --region us-east-1
```

#### Create Security Group
```bash
VPC_ID=$(aws ec2 describe-vpcs --region us-east-1 --query 'Vpcs[0].VpcId' --output text)
aws ec2 create-security-group --group-name fire-detection-dashboard-sg --description "Security group for fire detection dashboard" --vpc-id $VPC_ID --region us-east-1
```

#### Configure Security Group Rules
```bash
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=fire-detection-dashboard-sg" --region us-east-1 --query 'SecurityGroups[0].GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8501 --cidr 0.0.0.0/0 --region us-east-1
```

#### Create ECS Service
```bash
SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --region us-east-1 --query 'Subnets[*].SubnetId' --output text | head -n 1)
aws ecs create-service \
    --cluster fire-detection-cluster \
    --service-name fire-detection-dashboard-service \
    --task-definition fire-detection-dashboard \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --region us-east-1
```

### 4. Access the Dashboard

After 2-3 minutes, get the public IP address:
```bash
aws ecs describe-tasks --cluster fire-detection-cluster --tasks $(aws ecs list-tasks --cluster fire-detection-cluster --query 'taskArns[0]' --output text) --region us-east-1 --query 'tasks[0].containers[0].networkInterfaces[0].privateIpv4Address' --output text
```

Access the dashboard at: `http://<PUBLIC_IP>:8501`

## Verification

### Check SNS Subscription
Verify your email is subscribed:
```bash
cd "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system"
python3 test_sns_configuration.py
```

### Send Test Alert
Send a test message to verify alerts:
```bash
python3 verify_sns_functionality.py --send-test
```

## Troubleshooting

### Common Issues

1. **Docker not running**: Start Docker Desktop
2. **AWS permissions**: Ensure your user has ECS, ECR, and EC2 permissions
3. **VPC issues**: Ensure your account has a default VPC
4. **Security group errors**: Check that the security group exists

### Check Logs
```bash
aws logs tail /ecs/fire-detection-dashboard --follow
```

## Cost Considerations

### ECS Fargate Pricing
- CPU: 0.5 vCPU (~$0.0208/hour)
- Memory: 1 GB (~$0.0104/hour)
- Estimated monthly cost: ~$15-20

## Next Steps

1. Monitor the dashboard for live data
2. Verify all AWS services show as "OPERATIONAL"
3. Confirm SNS alerts are received
4. Share the public URL with your team

## Support

For deployment issues:
- Check AWS CloudWatch logs
- Verify IAM permissions
- Contact your AWS administrator