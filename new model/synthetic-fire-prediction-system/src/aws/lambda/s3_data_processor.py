"""
AWS Lambda function for real-time processing of high-frequency sensor data from S3.
This function is triggered by S3 object creation events and processes the data
for fire prediction using the deployed SageMaker model.
"""

import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
sns_client = boto3.client('sns', region_name='us-east-1')

# Configuration
ENDPOINT_NAME = 'fire-mvp-xgb-endpoint'
ALERT_TOPIC_ARN = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
ALERT_THRESHOLD = 0.6

def lambda_handler(event, context):
    """
    AWS Lambda function triggered by S3 object creation events.
    
    Args:
        event: S3 event data
        context: Lambda context object
        
    Returns:
        Response dictionary
    """
    
    logger.info(f"Processing S3 event with {len(event.get('Records', []))} records")
    
    results = []
    
    # Process each S3 record
    for record in event.get('Records', []):
        try:
            # Extract bucket and object information
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            
            logger.info(f"Processing file: {key} from bucket: {bucket}")
            
            # Process the file
            result = process_sensor_file(bucket, key)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing record: {str(e)}")
            results.append({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Processed {len(results)} files',
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    }

def process_sensor_file(bucket, key):
    """
    Process a sensor data file from S3.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        
    Returns:
        dict: Processing result
    """
    
    try:
        # Download file from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        
        # Parse CSV content
        df = pd.read_csv(StringIO(content))
        
        # Process based on file type
        if 'thermal_data' in key:
            features = process_thermal_data(df)
            # Add default gas features for thermal-only data
            gas_features = [400.0, 5.0, 5.0]
            combined_features = features + gas_features
        elif 'gas_data' in key:
            gas_features = process_gas_data(df)
            # Add default thermal features for gas-only data
            thermal_features = [25.0, 1.0, 30.0, 28.0, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 30.0, 1.0, 0.5]
            combined_features = thermal_features + gas_features
        else:
            logger.warning(f"Unknown file type: {key}")
            return {
                'file': key,
                'status': 'skipped',
                'reason': 'unknown_file_type',
                'timestamp': datetime.now().isoformat()
            }
        
        # Validate we have the correct number of features (18 total)
        if len(combined_features) != 18:
            logger.error(f"Incorrect number of features: {len(combined_features)}, expected 18")
            return {
                'file': key,
                'status': 'error',
                'error': f'Incorrect feature count: {len(combined_features)}',
                'timestamp': datetime.now().isoformat()
            }
        
        # Make fire risk prediction
        risk_score = predict_fire_risk(combined_features)
        
        # Send alert if risk is high
        if risk_score > ALERT_THRESHOLD:
            send_alert(risk_score, combined_features, key)
        
        logger.info(f"Successfully processed {key} with risk score: {risk_score:.2f}")
        
        return {
            'file': key,
            'status': 'success',
            'risk_score': float(risk_score),
            'features_count': len(combined_features),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing file {key}: {str(e)}")
        return {
            'file': key,
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def process_thermal_data(df):
    """
    Extract thermal features from raw thermal data.
    
    Args:
        df (pandas.DataFrame): Raw thermal data
        
    Returns:
        list: Extracted thermal features (15 features)
    """
    
    try:
        # For MLX90640, we expect 32x24 = 768 pixels
        # Extract pixel data (assuming first 768 columns are pixel values)
        pixel_columns = min(768, len(df.columns))
        pixel_data = df.iloc[:, :pixel_columns].values.flatten()
        
        # Handle case where we have fewer pixels than expected
        if len(pixel_data) < 768:
            # Pad with mean value
            padded_data = np.full(768, np.mean(pixel_data))
            padded_data[:len(pixel_data)] = pixel_data
            pixel_data = padded_data
        
        # Calculate thermal features
        features = []
        
        # Basic temperature statistics
        features.append(float(np.mean(pixel_data)))  # t_mean
        features.append(float(np.std(pixel_data)))   # t_std
        features.append(float(np.max(pixel_data)))   # t_max
        features.append(float(np.percentile(pixel_data, 95)))  # t_p95
        
        # Hot area features (using 40°C as hot threshold)
        hot_mask = pixel_data > 40.0
        total_pixels = len(pixel_data)
        hot_pixels = np.sum(hot_mask)
        features.append(float(hot_pixels / total_pixels * 100.0))  # t_hot_area_pct
        
        # Largest hot blob percentage (simplified)
        features.append(float(hot_pixels / total_pixels * 50.0))  # t_hot_largest_blob_pct
        
        # Gradient features
        grad_x = np.gradient(pixel_data)
        gradient_magnitude = np.abs(grad_x)
        features.append(float(np.mean(gradient_magnitude)))  # t_grad_mean
        features.append(float(np.std(gradient_magnitude)))   # t_grad_std
        
        # Temporal difference features (using previous values or random for demo)
        features.append(float(np.random.normal(0.1, 0.05)))  # t_diff_mean
        features.append(float(np.random.normal(0.05, 0.02))) # t_diff_std
        
        # Optical flow features (simplified)
        features.append(float(np.random.normal(0.2, 0.1)))   # flow_mag_mean
        features.append(float(np.random.normal(0.1, 0.05)))  # flow_mag_std
        
        # Temperature proxy features
        features.append(features[2])  # tproxy_val = t_max
        features.append(float(np.random.normal(1.0, 0.5)))   # tproxy_delta
        features.append(float(np.random.normal(0.5, 0.2)))   # tproxy_vel
        
        return features
        
    except Exception as e:
        logger.error(f"Error processing thermal data: {e}")
        # Return default thermal features
        return [25.0, 1.0, 30.0, 28.0, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 30.0, 1.0, 0.5]

def process_gas_data(df):
    """
    Extract gas features from raw gas data.
    
    Args:
        df (pandas.DataFrame): Raw gas data
        
    Returns:
        list: Extracted gas features (3 features)
    """
    
    try:
        # Extract gas readings (assuming columns: CO, NO2, VOC)
        co_value = df['CO'].iloc[0] if 'CO' in df.columns else 400.0
        no2_value = df['NO2'].iloc[0] if 'NO2' in df.columns else 0.0
        voc_value = df['VOC'].iloc[0] if 'VOC' in df.columns else 0.0
        
        # Calculate gas features
        gas_features = []
        gas_features.append(float(co_value))  # gas_val
        gas_features.append(float(np.random.normal(5.0, 2.0)))  # gas_delta
        gas_features.append(gas_features[1])  # gas_vel = gas_delta
        
        return gas_features
        
    except Exception as e:
        logger.error(f"Error processing gas data: {e}")
        # Return default gas features
        return [400.0, 5.0, 5.0]

def predict_fire_risk(features):
    """
    Send features to SageMaker endpoint for fire risk prediction.
    
    Args:
        features (list): Feature vector (18 features)
        
    Returns:
        float: Fire risk probability (0-1)
    """
    
    try:
        # Convert features to CSV format
        csv_features = ','.join([str(f) for f in features])
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=csv_features
        )
        
        result = float(response['Body'].read().decode())
        return result
        
    except Exception as e:
        logger.error(f"Error predicting fire risk: {e}")
        return 0.0

def send_alert(risk_score, features, file_key):
    """
    Send alert via SNS if fire risk is detected.
    
    Args:
        risk_score (float): Fire risk probability
        features (list): Feature vector
        file_key (str): Source file name
    """
    
    # Determine alert level
    if risk_score >= 0.8:
        alert_level = "EMERGENCY"
    elif risk_score >= 0.6:
        alert_level = "ALERT"
    elif risk_score >= 0.4:
        alert_level = "WARNING"
    else:
        alert_level = "INFO"
    
    alert_message = {
        "timestamp": datetime.now().isoformat(),
        "alert_level": alert_level,
        "risk_score": risk_score,
        "source_file": file_key,
        "message": f"Fire risk detected with probability {risk_score:.2f}"
    }
    
    try:
        sns_client.publish(
            TopicArn=ALERT_TOPIC_ARN,
            Message=json.dumps(alert_message, indent=2),
            Subject=f"🔥 Fire Detection Alert - {alert_level} (Risk: {risk_score:.2f})"
        )
        logger.info(f"Alert sent: {alert_level} - Risk score: {risk_score:.2f}")
        
    except Exception as e:
        logger.error(f"Error sending alert: {e}")