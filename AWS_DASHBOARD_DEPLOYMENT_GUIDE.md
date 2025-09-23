# 🔥 Saafe Fire Detection Dashboard - AWS Deployment Guide

This guide provides step-by-step instructions for deploying the Saafe Fire Detection Dashboard to AWS using ECS Fargate.

## 📋 Prerequisites

Before deploying, ensure you have the following:

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **Docker** installed and running
4. **Python 3.8+** installed
5. **Git** installed

## 🔧 Required AWS Permissions

Your AWS user/role needs the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:*",
                "ecs:*",
                "iam:CreateRole",
                "iam:GetRole",
                "iam:AttachRolePolicy",
                "logs:CreateLogGroup",
                "ec2:DescribeVpcs",
                "ec2:DescribeSubnets",
                "ec2:CreateSecurityGroup",
                "ec2:DescribeSecurityGroups",
                "ec2:AuthorizeSecurityGroupIngress",
                "sts:GetCallerIdentity"
            ],
            "Resource": "*"
        }
    ]
}
```

## 🚀 Deployment Steps

### 1. Configure AWS CLI

```bash
aws configure
```

Enter your AWS Access Key ID, Secret Access Key, and default region (us-east-1).

### 2. Install Docker

Follow the official Docker installation guide for your operating system:
https://docs.docker.com/get-docker/

### 3. Navigate to Project Directory

```bash
cd /path/to/saafe/project
```

### 4. Run the Deployment Script

```bash
chmod +x deploy_dashboard.sh
./deploy_dashboard.sh
```

## 📁 Deployment Components

The deployment script will create the following AWS resources:

1. **ECR Repository** (`saafe-dashboard`) - Stores the Docker image
2. **ECS Cluster** (`saafe-cluster`) - Container orchestration
3. **Task Definition** (`saafe-dashboard`) - Container configuration
4. **ECS Service** (`saafe-dashboard-service`) - Runs the dashboard
5. **IAM Role** (`SaafeDashboardExecutionRole`) - ECS task permissions
6. **Security Group** (`saafe-dashboard-sg`) - Network access control
7. **CloudWatch Log Group** (`/ecs/saafe-dashboard`) - Application logs

## 🔍 Monitoring Deployment

After deployment, monitor the service status:

```bash
aws ecs describe-services --cluster saafe-cluster --services saafe-dashboard-service --region us-east-1
```

## 🌐 Accessing the Dashboard

1. Wait 2-3 minutes for the service to start
2. Get the public IP address:

```bash
aws ecs describe-tasks --cluster saafe-cluster --tasks $(aws ecs list-tasks --cluster saafe-cluster --query 'taskArns[0]' --output text) --region us-east-1 --query 'tasks[0].containers[0].networkInterfaces[0].privateIpv4Address' --output text
```

3. Access the dashboard at: `http://<PUBLIC_IP>:8502`

## 🔄 Updating the Dashboard

To deploy updates:

1. Make changes to the dashboard code
2. Re-run the deployment script:

```bash
./deploy_dashboard.sh
```

## 🧹 Cleanup

To remove all deployed resources:

```bash
# Delete ECS service
aws ecs delete-service --cluster saafe-cluster --service saafe-dashboard-service --region us-east-1 --force

# Delete ECS cluster
aws ecs delete-cluster --cluster saafe-cluster --region us-east-1

# Delete ECR repository
aws ecr delete-repository --repository-name saafe-dashboard --region us-east-1 --force

# Delete IAM role
aws iam detach-role-policy --role-name SaafeDashboardExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy --region us-east-1
aws iam detach-role-policy --role-name SaafeDashboardExecutionRole --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess --region us-east-1
aws iam delete-role --role-name SaafeDashboardExecutionRole --region us-east-1

# Delete security group
# First get the security group ID
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=saafe-dashboard-sg" --region us-east-1 --query 'SecurityGroups[0].GroupId' --output text)
aws ec2 delete-security-group --group-id $SG_ID --region us-east-1

# Delete CloudWatch log group
aws logs delete-log-group --log-group-name /ecs/saafe-dashboard --region us-east-1
```

## 🛠️ Troubleshooting

### JSON/YAML Parsing Errors

If you encounter JSON or YAML parsing errors during deployment, this is likely due to hidden macOS metadata files (files starting with `._`) that are automatically created by the operating system. These files can interfere with AWS deployment processes.

To fix this issue:

1. Use our clean deployment script that automatically excludes these files:
   ```bash
   python3 create_clean_deployment.py
   ```

2. Or manually remove hidden files before deployment:
   ```bash
   zip -d your-deployment-package.zip "__MACOSX/*" "*.DS_Store" "._*"
   ```

### Docker Not Found
Install Docker from https://docs.docker.com/get-docker/

### AWS Credentials Not Configured
Run `aws configure` and enter your credentials

### Permission Errors
Ensure your AWS user has the required permissions listed above

### Service Not Starting
Check CloudWatch logs:
```bash
aws logs describe-log-streams --log-group-name /ecs/saafe-dashboard --region us-east-1
```

## 📞 Support

For deployment issues, contact:
- Email: ch.ajay1707@gmail.com
- Project documentation: See main Saafe documentation

---
*Built with ❤️ for the Saafe Fire Detection System*