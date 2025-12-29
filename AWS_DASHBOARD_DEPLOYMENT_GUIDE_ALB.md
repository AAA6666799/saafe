# 🔥 Saafe Lovable Dashboard - AWS Deployment Guide with ALB

This guide provides step-by-step instructions for deploying the Saafe Lovable Fire Detection Dashboard to AWS using ECS Fargate with Application Load Balancer (ALB) for public access.

## 📋 Prerequisites

Before deploying, ensure you have the following:

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured (`aws configure`)
3. **Docker** installed and running
4. **Node.js 18+** installed (for local testing)
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
                "ec2:*",
                "elbv2:*",
                "iam:CreateRole",
                "iam:GetRole",
                "iam:AttachRolePolicy",
                "iam:DetachRolePolicy",
                "iam:DeleteRole",
                "logs:*",
                "sts:GetCallerIdentity"
            ],
            "Resource": "*"
        }
    ]
}
```

## 🚀 Quick Deployment

### 1. Configure AWS CLI

```bash
aws configure
```

Enter your AWS Access Key ID, Secret Access Key, and default region (us-east-1).

### 2. Run the Deployment Script

```bash
chmod +x deploy_saafe_dashboard_alb.sh
./deploy_saafe_dashboard_alb.sh
```

The script will:
- Build the Docker image for the saafe-lovable dashboard
- Create ECR repository and push the image
- Set up complete VPC infrastructure (VPC, subnets, internet gateway)
- Create security groups for ALB and ECS tasks
- Create Application Load Balancer with target group
- Create ECS cluster, task definition, and service with ALB integration
- Provide the public ALB URL upon completion

## 📁 What Gets Created

The deployment script creates the following AWS resources:

### Container & Registry
1. **ECR Repository** (`saafe-lovable-dashboard`) - Stores the Docker image
2. **Docker Image** - Multi-stage build with Node.js frontend and backend

### Networking
3. **VPC** (`saafe-dashboard-vpc`) - Isolated network environment
4. **Subnets** - 2 public subnets across different AZs
5. **Internet Gateway** - Enables internet access
6. **Route Tables** - Routing configuration

### Security
7. **ALB Security Group** (`saafe-dashboard-alb-sg`) - Allows HTTP/HTTPS from anywhere
8. **ECS Task Security Group** (`saafe-dashboard-task-sg`) - Allows traffic from ALB only

### Load Balancing
9. **Application Load Balancer** (`saafe-dashboard-alb`) - Public load balancer
10. **Target Group** (`saafe-dashboard-tg`) - Routes traffic to ECS tasks

### Compute
11. **ECS Cluster** (`saafe-dashboard-cluster`) - Container orchestration
12. **Task Definition** (`saafe-dashboard-task`) - Container configuration (256 CPU, 512 MB RAM)
13. **ECS Service** (`saafe-dashboard-service`) - Runs and maintains the dashboard

### IAM & Monitoring
14. **Execution Role** (`SaafeDashboardExecutionRole`) - ECS task execution permissions
15. **Task Role** (`SaafeDashboardTaskRole`) - Application permissions (S3, CloudWatch)
16. **CloudWatch Log Group** (`/ecs/saafe-dashboard-task`) - Application logs

## 🌐 Accessing the Dashboard

After successful deployment:

1. The script will output the ALB URL: `http://<ALB-DNS-NAME>`
2. Wait 2-3 minutes for the service to fully start
3. Access the dashboard in your browser

The ALB provides public access to the dashboard running on port 8000 inside the container.

## 🔍 Monitoring Deployment

### Check Service Status
```bash
aws ecs describe-services --cluster saafe-dashboard-cluster --services saafe-dashboard-service --region us-east-1
```

### View Application Logs
```bash
aws logs tail /ecs/saafe-dashboard-task --region us-east-1 --follow
```

### Check ALB Health
```bash
aws elbv2 describe-target-health --target-group-arn <TARGET-GROUP-ARN> --region us-east-1
```

## 🔄 Updating the Dashboard

To deploy updates:

1. Make changes to the dashboard code in `saafe-lovable/`
2. Re-run the deployment script:
   ```bash
   ./deploy_saafe_dashboard_alb.sh
   ```

The script will build a new image, push it to ECR, and update the ECS service with zero-downtime deployment.

## 🧹 Cleanup

To remove all deployed resources:

```bash
./deploy_saafe_dashboard_alb.sh cleanup
```

This will delete all AWS resources created by the deployment script.

### Manual Cleanup (if needed)

If the automated cleanup fails, you can manually delete resources in this order:

1. **ECS Service & Cluster**
   ```bash
   aws ecs delete-service --cluster saafe-dashboard-cluster --service saafe-dashboard-service --region us-east-1 --force
   aws ecs delete-cluster --cluster saafe-dashboard-cluster --region us-east-1
   ```

2. **ALB & Target Group**
   ```bash
   aws elbv2 delete-load-balancer --load-balancer-arn <ALB-ARN> --region us-east-1
   aws elbv2 delete-target-group --target-group-arn <TG-ARN> --region us-east-1
   ```

3. **EC2 Resources**
   ```bash
   aws ec2 delete-security-group --group-id <ALB-SG-ID> --region us-east-1
   aws ec2 delete-security-group --group-id <TASK-SG-ID> --region us-east-1
   aws ec2 delete-subnet --subnet-id <SUBNET-1-ID> --region us-east-1
   aws ec2 delete-subnet --subnet-id <SUBNET-2-ID> --region us-east-1
   aws ec2 delete-vpc --vpc-id <VPC-ID> --region us-east-1
   ```

4. **IAM Roles**
   ```bash
   aws iam detach-role-policy --role-name SaafeDashboardExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
   aws iam delete-role --role-name SaafeDashboardExecutionRole
   aws iam detach-role-policy --role-name SaafeDashboardTaskRole --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
   aws iam detach-role-policy --role-name SaafeDashboardTaskRole --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
   aws iam delete-role --role-name SaafeDashboardTaskRole
   ```

5. **ECR Repository**
   ```bash
   aws ecr delete-repository --repository-name saafe-lovable-dashboard --region us-east-1 --force
   ```

## 🛠️ Troubleshooting

### Common Issues

#### JSON/YAML Parsing Errors
If you encounter JSON or YAML parsing errors during deployment, this is likely due to hidden macOS metadata files (files starting with `._`) that are automatically created by the operating system. These files can interfere with AWS deployment processes.

**Fix:**
1. Use our clean deployment script that automatically excludes these files
2. Or manually remove hidden files before deployment:
   ```bash
   find . -name "._*" -delete
   ```

#### Docker Build Failures
- Ensure Docker is running
- Check that all required files are present in `saafe-lovable/`
- Verify Node.js dependencies in `package.json` files

#### AWS Permission Errors
- Ensure your AWS user has the required permissions listed above
- Check that AWS CLI is configured with correct credentials
- Verify the default region is set to us-east-1

#### Service Not Starting
- Check ECS service events:
  ```bash
  aws ecs describe-services --cluster saafe-dashboard-cluster --services saafe-dashboard-service --region us-east-1 --query 'services[0].events'
  ```
- View CloudWatch logs for detailed error messages

#### ALB Not Routing Traffic
- Verify target group health:
  ```bash
  aws elbv2 describe-target-health --target-group-arn <TG-ARN> --region us-east-1
  ```
- Check security groups allow traffic from ALB to ECS tasks
- Ensure the container is listening on port 8000

### Health Checks

The deployment includes health checks:
- **Container Health Check**: HTTP GET to `/api/fire-detection-data`
- **ALB Health Check**: Same endpoint with 30-second intervals

### Resource Limits

Current configuration:
- **CPU**: 256 units (0.25 vCPU)
- **Memory**: 512 MB
- **Tasks**: 1 (can be scaled up if needed)

## 📊 Cost Estimation

Approximate monthly costs (us-east-1):
- **ECS Fargate**: ~$10-15 (256 CPU, 512 MB, 1 task, 24/7)
- **ALB**: ~$15-20 (with low traffic)
- **ECR**: ~$1 (storage + data transfer)
- **CloudWatch Logs**: ~$1-2
- **VPC**: Free (within limits)

**Total**: ~$27-38/month for basic deployment

## 🔒 Security Considerations

- ALB allows HTTP traffic from anywhere (consider adding HTTPS with ACM certificate)
- ECS tasks are in private subnets but accessible through ALB
- IAM roles follow least-privilege principle
- Security groups restrict traffic appropriately

## 📞 Support

For deployment issues, contact:
- Email: ch.ajay1707@gmail.com
- Check CloudWatch logs for application errors
- Review AWS service events for infrastructure issues

---
*Built with ❤️ for the Saafe Fire Detection System*