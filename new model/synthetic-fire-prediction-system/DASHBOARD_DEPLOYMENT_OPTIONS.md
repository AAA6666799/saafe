# 🔥 Fire Detection Dashboard - Deployment Options

## Current Status
The dashboard is currently running locally at: **http://localhost:8502**

This is only accessible from your local machine. To make it publicly accessible, you need to deploy it to the cloud.

## Deployment Options

### Option 1: Elastic Beanstalk Deployment (Recommended - Easiest)
**Script**: [deploy_eb.sh](file:///Volumes/Ajay/saafe%20copy%203/new%20model/synthetic-fire-prediction-system/deploy_eb.sh)
**Public Access**: Via a public URL
**Complexity**: Low

#### Steps:
1. Ensure you have AWS CLI and EB CLI installed
2. Run the deployment script:
   ```bash
   cd "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system"
   ./deploy_eb.sh
   ```
3. Wait 5-10 minutes for deployment to complete
4. Get the public URL:
   ```bash
   eb status
   ```

### Option 2: ECS Deployment (More Control)
**Script**: [deploy_dashboard.sh](file:///Volumes/Ajay/saafe%20copy%203/new%20model/synthetic-fire-prediction-system/deploy_dashboard.sh)
**Public Access**: Via a public IP address
**Complexity**: Medium

#### Steps:
1. Ensure you have AWS CLI and Docker installed
2. Run the deployment script:
   ```bash
   cd "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system"
   ./deploy_dashboard.sh
   ```
3. Wait 2-3 minutes for deployment to complete
4. Get the public IP:
   ```bash
   aws ecs describe-tasks --cluster fire-detection-cluster --tasks $(aws ecs list-tasks --cluster fire-detection-cluster --query 'taskArns[0]' --output text) --region us-east-1 --query 'tasks[0].containers[0].networkInterfaces[0].privateIpv4Address' --output text
   ```
5. Access at: `http://<PUBLIC_IP>:8501`

## Why Local vs. Public?

### Local Dashboard (Current)
- **URL**: http://localhost:8502
- **Access**: Only from your machine
- **Purpose**: Development and testing
- **Security**: No public exposure
- **Cost**: None

### Public Dashboard (After Deployment)
- **URL**: Public URL or IP address
- **Access**: From anywhere in the world
- **Purpose**: Production monitoring
- **Security**: Requires proper configuration
- **Cost**: AWS service charges (~$15-30/month)

## Prerequisites for Deployment

### Required Tools
1. **AWS CLI** - For AWS service interaction
2. **Docker** - For containerization (ECS deployment)
3. **EB CLI** - For Elastic Beanstalk deployment

### AWS Permissions Needed
The dashboard requires IAM permissions to access:
- S3 bucket: `data-collector-of-first-device`
- Lambda function: `saafe-s3-data-processor`
- SageMaker endpoint: `fire-mvp-xgb-endpoint`
- SNS topic: `fire-detection-alerts`
- CloudWatch metrics

## Benefits of Public Deployment

### 1. Global Accessibility
- Access from any device with internet
- Share with team members
- Monitor from anywhere

### 2. Real-Time Monitoring
- Continuous system status updates
- Immediate alert on issues
- Historical data tracking

### 3. Professional Presentation
- Proper URL instead of localhost
- Better for demonstrations
- Suitable for production use

## Deployment Comparison

| Feature | Elastic Beanstalk | ECS |
|---------|------------------|-----|
| Complexity | Low | Medium |
| Public Access | URL | IP Address |
| Automatic Scaling | Yes | Manual |
| Cost | ~$15/month | ~$30/month |
| Setup Time | 5-10 minutes | 2-3 minutes |
| Maintenance | Low | Medium |

## Next Steps

### To Deploy via Elastic Beanstalk:
```bash
# 1. Install EB CLI (if not already installed)
pip install awsebcli

# 2. Deploy
cd "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system"
./deploy_eb.sh

# 3. Check status after 5-10 minutes
eb status
```

### To Deploy via ECS:
```bash
# 1. Ensure Docker is running
# 2. Deploy
cd "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system"
./deploy_dashboard.sh

# 3. Get public IP after 2-3 minutes
aws ecs describe-tasks --cluster fire-detection-cluster --tasks $(aws ecs list-tasks --cluster fire-detection-cluster --query 'taskArns[0]' --output text) --region us-east-1 --query 'tasks[0].containers[0].networkInterfaces[0].privateIpv4Address' --output text
```

## Troubleshooting

### Common Issues:
1. **AWS CLI not configured**: Run `aws configure`
2. **Missing permissions**: Contact your AWS administrator
3. **Docker not running**: Start Docker Desktop
4. **Port conflicts**: Stop local dashboard before deployment

### Verification:
After deployment, verify the dashboard shows:
- ✅ All AWS services as "OPERATIONAL"
- ✅ Live data detection when devices are sending data
- ✅ SNS subscription count > 0 (after setting up alerts)

## Support
For deployment issues:
- Check [DASHBOARD_DEPLOYMENT_GUIDE.md](file:///Volumes/Ajay/saafe%20copy%203/new%20model/synthetic-fire-prediction-system/DASHBOARD_DEPLOYMENT_GUIDE.md) for detailed instructions
- Ensure AWS credentials have required permissions
- Contact system administrator for access issues