# 🔥 Saafe Fire Detection - AWS Dashboard

A real-time dashboard for monitoring sensor data and fire detection scores from your AWS-deployed Saafe system.

## 📋 Features

- Real-time sensor data visualization (temperature, PM2.5, CO₂, audio levels)
- Fire detection risk scoring
- System component status monitoring (S3, Lambda, SageMaker)
- Historical data trends
- Responsive design for all devices

## 🚀 Quick Start

1. **Ensure AWS credentials are configured:**
   ```bash
   # Using AWS CLI
   aws configure
   
   # Or set environment variables
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-east-1
   ```

2. **Run the dashboard using the startup script:**
   ```bash
   cd /path/to/saafe/project
   ./start_aws_dashboard.sh
   ```

   Or run directly:
   ```bash
   python run_aws_dashboard.py
   ```

3. **Access the dashboard:**
   Open your browser to http://localhost:8502

## 📊 Dashboard Components

### System Status
- Overall system health indicator
- Individual component status (S3, Lambda, SageMaker)
- Last update timestamp

### Sensor Readings
- Real-time temperature measurements
- PM2.5 particulate matter levels
- CO₂ concentration readings
- Audio level detection

### Fire Detection Score
- Real-time fire risk assessment
- Risk level classification (Low, Medium, High)
- Historical trend visualization

### Recent Data
- Tabular view of recent sensor readings
- Timestamped data for analysis

## ⚙️ Configuration

The dashboard connects to the following AWS services:

- **S3 Bucket**: `data-collector-of-first-device`
- **Lambda Function**: `saafe-s3-data-processor`
- **SageMaker Endpoint**: `fire-mvp-xgb-endpoint`

Ensure these resources exist in your AWS account and your credentials have appropriate permissions.

## 🔐 AWS Permissions Required

Your AWS credentials need the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket",
                "s3:HeadBucket"
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
            "Resource": "arn:aws:lambda:us-east-1:YOUR_ACCOUNT:function:saafe-s3-data-processor"
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:DescribeEndpoint"
            ],
            "Resource": "arn:aws:sagemaker:us-east-1:YOUR_ACCOUNT:endpoint/fire-mvp-xgb-endpoint"
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

##  troubleshoot

### No Data Displayed
1. Verify devices are sending data to S3
2. Check S3 bucket permissions
3. Confirm file naming conventions match expectations

### Connection Errors
1. Verify AWS credentials are properly configured
2. Check network connectivity to AWS services
3. Ensure IAM permissions are correctly set

### Dashboard Not Loading
1. Confirm all required Python packages are installed
2. Check that AWS services are running
3. Verify the dashboard can access the required endpoints

## 📞 Support

For issues with the dashboard, contact:
- Email: ch.ajay1707@gmail.com
- Project documentation: See main Saafe documentation

---
*Built with ❤️ for the Saafe Fire Detection System*