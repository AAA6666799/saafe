"""
AWS Lambda function for Analysis Agent implementation.
This function performs fire pattern analysis using SageMaker endpoints.
"""

import json
import boto3
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_runtime = boto3.client('sagemaker-runtime')
sns = boto3.client('sns')

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda function for Analysis Agent.
    
    Args:
        event: Event data containing sensor features
        context: Lambda context object
        
    Returns:
        Response dictionary with analysis results
    """
    
    try:
        logger.info(f"Analysis Agent triggered with event: {json.dumps(event)}")
        
        # Extract features from event
        features = event.get('features', {})
        
        # Validate required features
        if not features:
            raise ValueError("No features provided in event")
        
        # Perform fire pattern analysis using SageMaker endpoint
        analysis_results = perform_fire_analysis(features)
        
        # Send results to monitoring queue
        send_results_to_monitoring(analysis_results)
        
        # Log analysis results
        logger.info(f"Analysis completed: {json.dumps(analysis_results)}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Analysis completed successfully',
                'timestamp': datetime.now().isoformat(),
                'results': analysis_results
            })
        }
        
    except Exception as e:
        logger.error(f"Error in analysis agent: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }

def perform_fire_analysis(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform fire analysis using SageMaker endpoint.
    
    Args:
        features: Dictionary containing 18 features (15 thermal + 3 gas)
        
    Returns:
        Dictionary with analysis results
    """
    
    # Validate that we have all required features
    required_features = [
        't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
        't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
        't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
        'tproxy_val', 'tproxy_delta', 'tproxy_vel',
        'gas_val', 'gas_delta', 'gas_vel'
    ]
    
    missing_features = [f for f in required_features if f not in features]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Call SageMaker endpoint for prediction
    # Using the actual deployed endpoint name
    endpoint_name = 'flir-scd41-fire-detection-corrected-v3-20250901-121555'
    
    # Send features as individual keys (matching the serve.py format)
    payload = features
    
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    # Parse prediction results
    result = json.loads(response['Body'].read().decode())
    
    # Extract prediction and confidence
    predictions = result.get('predictions', [])
    if not predictions:
        raise ValueError("No predictions returned from SageMaker endpoint")
    
    prediction = predictions[0]
    fire_probability = prediction if isinstance(prediction, (int, float)) else prediction.get('score', 0.0)
    
    # Determine fire detection based on confidence threshold
    confidence_threshold = 0.7
    fire_detected = fire_probability >= confidence_threshold
    
    return {
        'timestamp': datetime.now().isoformat(),
        'fire_detected': fire_detected,
        'confidence_score': float(fire_probability),
        'prediction_details': {
            'raw_prediction': prediction,
            'confidence_threshold': confidence_threshold
        },
        'features_used': features
    }

def send_results_to_monitoring(analysis_results: Dict[str, Any]) -> None:
    """
    Send analysis results to monitoring SNS topic.
    
    Args:
        analysis_results: Results from fire analysis
    """
    
    try:
        topic_arn = 'arn:aws:sns:us-east-1:691595239825:fire-detection-analysis-results'
        
        message = {
            'message_type': 'analysis_results',
            'timestamp': analysis_results['timestamp'],
            'results': analysis_results
        }
        
        sns.publish(
            TopicArn=topic_arn,
            Message=json.dumps(message),
            Subject='Fire Analysis Results'
        )
        
        logger.info("Analysis results sent to monitoring topic")
        
    except Exception as e:
        logger.error(f"Failed to send analysis results: {str(e)}")

def validate_features(features: Dict[str, Any]) -> bool:
    """
    Validate that features are within expected ranges.
    
    Args:
        features: Dictionary of features to validate
        
    Returns:
        True if features are valid, False otherwise
    """
    
    # Basic validation ranges (these would be more sophisticated in production)
    validation_ranges = {
        't_mean': (-50, 1000),
        't_std': (0, 100),
        't_max': (-50, 1000),
        't_p95': (-50, 1000),
        't_hot_area_pct': (0, 100),
        't_hot_largest_blob_pct': (0, 100),
        't_grad_mean': (-100, 100),
        't_grad_std': (0, 100),
        't_diff_mean': (-100, 100),
        't_diff_std': (0, 100),
        'flow_mag_mean': (0, 100),
        'flow_mag_std': (0, 100),
        'tproxy_val': (-50, 1000),
        'tproxy_delta': (-100, 100),
        'tproxy_vel': (-100, 100),
        'gas_val': (100, 100000),
        'gas_delta': (-10000, 10000),
        'gas_vel': (-1000, 1000)
    }
    
    for feature_name, (min_val, max_val) in validation_ranges.items():
        if feature_name in features:
            value = features[feature_name]
            if not (min_val <= value <= max_val):
                logger.warning(f"Feature {feature_name} value {value} outside expected range [{min_val}, {max_val}]")
                return False
    
    return True