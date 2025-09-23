# S3 Data Processor for High-Frequency Sensor Data

## Overview
This document describes the S3 Data Processor Lambda function that handles real-time processing of high-frequency sensor data (collected every second/minute) from S3 for fire prediction.

## Architecture

### Data Flow
1. **Sensor Data Collection**: Grove sensors collect data every second/minute and upload CSV files to S3
2. **S3 Event Trigger**: Object creation events in S3 trigger the Lambda function
3. **Feature Engineering**: Raw sensor data is processed to extract 18 features
4. **Fire Prediction**: Features are sent to the SageMaker endpoint for prediction
5. **Alerting**: High-risk predictions trigger SNS alerts

### Components
- **Lambda Function**: `saafe-s3-data-processor` - Processes S3 events
- **S3 Bucket**: `data-collector-of-first-device` - Stores raw sensor data
- **SageMaker Endpoint**: `fire-mvp-xgb-endpoint` - Provides fire risk predictions
- **SNS Topic**: `fire-detection-alerts` - Sends alerts for high-risk detections

## Implementation Details

### File Processing
The Lambda function handles two types of CSV files:
1. **Thermal Data Files**: `thermal_data_YYYYMMDD_HHMMSS.csv` - Contains 768 pixel temperature values
2. **Gas Data Files**: `gas_data_YYYYMMDD_HHMMSS.csv` - Contains CO, NO2, VOC readings

### Feature Engineering
The function extracts 18 features required by the XGBoost model:

#### Thermal Features (15 features)
- `t_mean`: Mean temperature
- `t_std`: Temperature standard deviation
- `t_max`: Maximum temperature
- `t_p95`: 95th percentile temperature
- `t_hot_area_pct`: Percentage of hot area (>40°C)
- `t_hot_largest_blob_pct`: Largest hot blob percentage
- `t_grad_mean`: Mean temperature gradient
- `t_grad_std`: Temperature gradient standard deviation
- `t_diff_mean`: Mean temporal difference
- `t_diff_std`: Temporal difference standard deviation
- `flow_mag_mean`: Mean optical flow magnitude
- `flow_mag_std`: Optical flow magnitude standard deviation
- `tproxy_val`: Temperature proxy value
- `tproxy_delta`: Temperature proxy delta
- `tproxy_vel`: Temperature proxy velocity

#### Gas Features (3 features)
- `gas_val`: Gas concentration value
- `gas_delta`: Change in gas concentration
- `gas_vel`: Rate of change in gas concentration

### Fire Risk Prediction
The extracted features are sent to the SageMaker endpoint for prediction. The model returns a probability score between 0 and 1, where:
- 0.0-0.4: Low risk
- 0.4-0.6: Medium risk (WARNING level)
- 0.6-0.8: High risk (ALERT level)
- 0.8-1.0: Critical risk (EMERGENCY level)

### Alerting
When the fire risk probability exceeds 0.6, an alert is sent via SNS with:
- Timestamp
- Risk level
- Risk score
- Source file name

## Deployment

### Prerequisites
1. AWS CLI configured with appropriate permissions
2. Existing SageMaker endpoint (`fire-mvp-xgb-endpoint`)
3. SNS topic (`fire-detection-alerts`)
4. S3 bucket (`data-collector-of-first-device`)

### Deployment Steps
1. Run the deployment script:
   ```bash
   ./deploy_s3_processor.sh
   ```

2. The script will:
   - Package the Lambda function with dependencies
   - Deploy or update the Lambda function
   - Configure S3 event triggers
   - Set up necessary permissions

### Configuration
The Lambda function uses the following environment variables:
- `ENDPOINT_NAME`: SageMaker endpoint name (default: `fire-mvp-xgb-endpoint`)
- `ALERT_TOPIC_ARN`: SNS topic ARN (default: `arn:aws:sns:us-east-1:691595239825:fire-detection-alerts`)
- `ALERT_THRESHOLD`: Minimum risk score to trigger alerts (default: 0.6)

## Testing

### Local Testing
Run the test script to verify functionality:
```bash
python test_s3_processor.py
```

### Remote Testing
After deployment, the function can be tested by:
1. Uploading sample CSV files to the S3 bucket
2. Using the AWS Lambda console to test with sample events
3. Monitoring CloudWatch logs for execution results

## Performance Considerations

### Concurrency
The Lambda function is configured with:
- Timeout: 900 seconds (15 minutes)
- Memory: 1024 MB
- Concurrency: Managed by AWS (can be limited if needed)

### Scaling
AWS Lambda automatically scales to handle the high-frequency data stream, processing each file independently.

### Error Handling
The function includes comprehensive error handling:
- S3 download errors
- CSV parsing errors
- Feature extraction errors
- SageMaker invocation errors
- SNS publishing errors

## Monitoring

### CloudWatch Metrics
Monitor the following metrics:
- Invocation count
- Error rate
- Duration
- Throttles

### Logging
All processing steps are logged with appropriate log levels:
- INFO: Processing steps and results
- WARNING: Non-critical issues
- ERROR: Processing failures

## Troubleshooting

### Common Issues
1. **Permission Errors**: Ensure the Lambda execution role has necessary permissions
2. **Timeout Errors**: For very large files, consider increasing the timeout
3. **Memory Errors**: For complex processing, consider increasing memory allocation
4. **SageMaker Errors**: Verify the endpoint is active and accessible

### Debugging
1. Check CloudWatch logs for detailed error messages
2. Test the function locally with sample data
3. Verify S3 event triggers are configured correctly
4. Confirm SageMaker endpoint is functioning

## Conclusion
The S3 Data Processor Lambda function provides a robust solution for real-time processing of high-frequency sensor data, enabling the fire prediction system to respond quickly to potential fire risks. The implementation handles both thermal and gas data files, extracts the required features, makes predictions using the deployed model, and sends alerts when necessary.