# Fire Detection System - Next Steps for Operations

## System Status
✅ Devices have been successfully deployed and are sending data to the cloud
✅ All cloud components (S3, Lambda, SageMaker, SNS) are properly configured
✅ End-to-end data flow is established

## Next Steps for Full System Operation

### 1. Configure Alerting and Notifications

The system is currently sending alerts to an SNS topic, but no subscriptions are configured. To receive alerts:

1. Run the SNS configuration script:
   ```bash
   python3 configure_sns_alerts.py
   ```

2. Modify the script to add your email or SMS subscriptions:
   ```python
   # Add email subscription
   sns_client.subscribe(
       TopicArn=topic_arn,
       Protocol='email',
       Endpoint='your-email@example.com'
   )
   
   # Add SMS subscription
   sns_client.subscribe(
       TopicArn=topic_arn,
       Protocol='sms',
       Endpoint='+1234567890'
   )
   ```

### 2. Monitor System Processing

Monitor the system to ensure data is being processed correctly:

1. Check CloudWatch logs for the Lambda function:
   ```bash
   aws logs describe-log-groups --log-group-name-prefix /aws/lambda/saafe-s3-data-processor
   ```

2. Run the monitoring dashboard:
   ```bash
   python3 system_monitor_dashboard.py
   ```

### 3. Verify End-to-End Data Flow

Test the complete system with sample data:

1. Run the high-frequency data processing script:
   ```bash
   python3 process_high_frequency_data.py
   ```

2. Check that:
   - Data is read from S3
   - Features are extracted correctly
   - Predictions are made using SageMaker
   - Alerts are sent via SNS (if risk threshold is met)

### 4. Set Up Automated Monitoring

For continuous operation, set up automated monitoring:

1. Create a CloudWatch alarm for Lambda errors:
   ```bash
   aws cloudwatch put-metric-alarm \
       --alarm-name "FireDetectionLambdaErrors" \
       --alarm-description "Alarm when Lambda function has errors" \
       --metric-name Errors \
       --namespace AWS/Lambda \
       --statistic Sum \
       --period 300 \
       --threshold 1 \
       --comparison-operator GreaterThanOrEqualToThreshold \
       --dimensions Name=FunctionName,Value=saafe-s3-data-processor \
       --evaluation-periods 1 \
       --alarm-actions arn:aws:sns:us-east-1:691595239825:fire-detection-alerts
   ```

2. Set up a scheduled Lambda function to run periodic checks

### 5. Performance Optimization

To optimize the system for high-frequency processing:

1. **Adjust Lambda concurrency**:
   - Set reserved concurrency to handle peak loads
   - Configure provisioned concurrency for consistent performance

2. **Optimize batch processing**:
   - Process multiple files in a single Lambda invocation
   - Use SQS for queueing high volumes of data

3. **Monitor costs**:
   - Track SageMaker endpoint usage
   - Monitor S3 requests and data transfer
   - Set up billing alerts

## Key System Components

### Data Flow
1. **Devices** → S3 (data-collector-of-first-device)
2. **S3** → Lambda (saafe-s3-data-processor)
3. **Lambda** → SageMaker (fire-mvp-xgb-endpoint)
4. **SageMaker** → SNS (fire-detection-alerts)
5. **SNS** → Subscribers (email, SMS, etc.)

### File Naming Convention
- Thermal data: `thermal_data_YYYYMMDD_HHMMSS.csv`
- Gas data: `gas_data_YYYYMMDD_HHMMSS.csv`

### Feature Engineering
The system expects 18 features for the XGBoost model:
- 15 thermal features (t_mean, t_std, t_max, etc.)
- 3 gas features (gas_val, gas_delta, gas_vel)

## Troubleshooting

### No Alerts Being Sent
1. Check SNS subscriptions are configured
2. Verify SageMaker endpoint is InService
3. Check Lambda logs for processing errors

### Lambda Processing Failures
1. Check CloudWatch logs for error messages
2. Verify S3 bucket permissions
3. Check SageMaker endpoint availability

### High Latency
1. Monitor Lambda duration metrics
2. Check SageMaker endpoint response times
3. Consider increasing Lambda memory allocation

## Security Considerations

1. **IAM Roles**: Ensure Lambda functions have minimal required permissions
2. **S3 Bucket Policies**: Restrict access to only necessary services
3. **SageMaker Endpoint**: Use VPC endpoints for private access
4. **SNS Topics**: Restrict publishing and subscription permissions

## Maintenance

1. **Regular Updates**: Update Lambda functions with latest code
2. **Model Retraining**: Schedule periodic model retraining with new data
3. **Cost Monitoring**: Review AWS billing regularly
4. **Backup Strategy**: Implement backup for critical data and configurations

## Contact Information

For system administration and support:
- AWS Account: 691595239825
- Region: us-east-1
- Primary Contact: [Add your contact information here]