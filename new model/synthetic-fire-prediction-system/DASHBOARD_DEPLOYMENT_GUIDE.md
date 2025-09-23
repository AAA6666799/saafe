# Fire Detection System Dashboard Deployment Guide

## Overview
This guide explains how to deploy the Fire Detection System Dashboard, which provides real-time monitoring of your deployed fire detection system. The dashboard shows live data from your AWS services without any dummy data.

## Prerequisites
1. AWS CLI installed and configured with appropriate credentials
2. Docker installed (for ECS deployment)
3. EB CLI installed (for Elastic Beanstalk deployment)
4. Python 3.9+
5. Active AWS account with permissions for:
   - ECS (if using ECS deployment)
   - Elastic Beanstalk (if using EB deployment)
   - ECR (if using ECS deployment)
   - IAM (to create roles if needed)

## Deployment Options

### Option 1: Elastic Beanstalk Deployment (Recommended for simplicity)

1. **Install EB CLI** (if not already installed):
   ```bash
   pip install awsebcli
   ```

2. **Deploy the dashboard**:
   ```bash
   cd "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system"
   ./deploy_eb.sh
   ```

3. **Monitor deployment**:
   ```bash
   eb status
   ```

4. **Access the dashboard**:
   Once deployment is complete (5-10 minutes), get the URL:
   ```bash
   eb status
   ```
   Then open the URL in your browser.

### Option 2: ECS Deployment (More control)

1. **Deploy the dashboard**:
   ```bash
   cd "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system"
   ./deploy_dashboard.sh
   ```

2. **Get the public URL**:
   After 2-3 minutes, get the public IP:
   ```bash
   aws ecs describe-tasks --cluster fire-detection-cluster --tasks $(aws ecs list-tasks --cluster fire-detection-cluster --query 'taskArns[0]' --output text) --region us-east-1 --query 'tasks[0].containers[0].networkInterfaces[0].privateIpv4Address' --output text
   ```

3. **Access the dashboard**:
   Open `http://<PUBLIC_IP>:8501` in your browser.

## Dashboard Features

### Real-Time Data
The dashboard connects directly to your AWS services and displays:
- ✅ Live S3 bucket status with actual file counts
- ✅ Lambda function status and configuration
- ✅ SageMaker endpoint status
- ✅ SNS topic status and subscription count
- ✅ Recent files processed
- ✅ Performance metrics

### No Dummy Data
All data shown in the dashboard is pulled directly from your AWS services in real-time:
- File counts from your actual S3 bucket
- Function configuration from your actual Lambda function
- Endpoint status from your actual SageMaker endpoint
- Subscription information from your actual SNS topic

### Global Accessibility
Once deployed, the dashboard is accessible from anywhere in the world via:
- A public URL (Elastic Beanstalk)
- A public IP address with port (ECS)

## Security Considerations

### IAM Permissions
The dashboard requires the following IAM permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetObject"
            ],
            "Resource": [
                "arn:aws:s3:::data-collector-of-first-device",
                "arn:aws:s3:::data-collector-of-first-device/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "lambda:GetFunction"
            ],
            "Resource": "arn:aws:lambda:us-east-1:691595239825:function:saafe-s3-data-processor"
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:DescribeEndpoint"
            ],
            "Resource": "arn:aws:sagemaker:us-east-1:691595239825:endpoint/fire-mvp-xgb-endpoint"
        },
        {
            "Effect": "Allow",
            "Action": [
                "sns:GetTopicAttributes",
                "sns:ListSubscriptionsByTopic"
            ],
            "Resource": "arn:aws:sns:us-east-1:691595239825:fire-detection-alerts"
        },
        {
            "Effect": "Allow",
            "Action": [
                "cloudwatch:GetMetricStatistics"
            ],
            "Resource": "*"
        }
    ]
}
```

### Network Security
- The dashboard is deployed with appropriate security groups
- Only necessary ports are exposed
- HTTPS can be configured through AWS Load Balancer

## Updating the Dashboard

### Elastic Beanstalk
```bash
eb deploy
```

### ECS
Re-run the deployment script:
```bash
./deploy_dashboard.sh
```

## Troubleshooting

### Common Issues

1. **Permission Errors**
   - Ensure your AWS credentials have the required permissions
   - Check IAM roles for the deployed service

2. **Dashboard Not Loading**
   - Check CloudWatch logs for the deployed service
   - Verify security groups allow inbound traffic on port 8501

3. **Stale Data**
   - Dashboard refreshes every 30 seconds automatically
   - Manual refresh available with the refresh button

### Checking Logs

#### Elastic Beanstalk
```bash
eb logs
```

#### ECS
```bash
aws logs tail /ecs/fire-detection-dashboard --follow
```

## Cost Considerations

### Elastic Beanstalk
- t3.small instance: ~$0.0208 per hour
- Estimated monthly cost: ~$15

### ECS
- Fargate (0.5 vCPU, 1 GB): ~$0.04048 per hour
- Estimated monthly cost: ~$30

## Maintenance

### Regular Tasks
1. Monitor CloudWatch logs for errors
2. Update dependencies periodically
3. Review security groups and IAM permissions
4. Check for new versions of the dashboard

### Scaling
The dashboard automatically scales based on demand when deployed with Elastic Beanstalk.

## Support

For issues with the dashboard deployment:
- AWS Account: 691595239825
- Region: us-east-1
- Contact: [Add your contact information here]

## Verification

To verify the dashboard is working correctly:
1. Check that all system components show as operational
2. Verify recent files are shown in the S3 section
3. Confirm Lambda function details are displayed
4. Ensure SageMaker endpoint shows as "InService"
5. Check SNS subscription count (0 indicates alerts won't be sent)

The dashboard provides a real-time view of your fire detection system status, with no dummy data and global accessibility.