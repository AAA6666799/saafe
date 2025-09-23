"""
AWS Lambda function for Monitoring Agent implementation.
This function monitors system health and sensor data quality.
"""

import json
import boto3
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda function for Monitoring Agent.
    
    Args:
        event: Event data from CloudWatch or other sources
        context: Lambda context object
        
    Returns:
        Response dictionary with status and results
    """
    
    try:
        logger.info(f"Monitoring Agent triggered with event: {json.dumps(event)}")
        
        # Extract sensor data from event
        sensor_data = event.get('sensor_data', {})
        
        # Perform basic monitoring checks
        monitoring_results = perform_monitoring_checks(sensor_data)
        
        # Send results to SNS topic if issues detected
        if monitoring_results.get('issues_detected', False):
            send_alert(monitoring_results)
        
        # Log results
        logger.info(f"Monitoring completed: {json.dumps(monitoring_results)}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Monitoring completed successfully',
                'timestamp': datetime.now().isoformat(),
                'results': monitoring_results
            })
        }
        
    except Exception as e:
        logger.error(f"Error in monitoring agent: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }

def perform_monitoring_checks(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform monitoring checks on sensor data.
    
    Args:
        sensor_data: Dictionary containing sensor data
        
    Returns:
        Dictionary with monitoring results
    """
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'issues_detected': False,
        'issues': []
    }
    
    # Check FLIR thermal data
    if 'flir' in sensor_data:
        flir_data = sensor_data['flir']
        if 'temperature' in flir_data:
            temp = flir_data['temperature']
            # Check for extreme temperatures that might indicate sensor issues
            if temp > 1000 or temp < -50:
                results['issues_detected'] = True
                results['issues'].append({
                    'type': 'flir_temperature_anomaly',
                    'value': temp,
                    'description': f'Extreme temperature reading: {temp}°C'
                })
    
    # Check SCD41 gas data
    if 'scd41' in sensor_data:
        gas_data = sensor_data['scd41']
        if 'co2_concentration' in gas_data:
            co2 = gas_data['co2_concentration']
            # Check for extreme CO2 levels
            if co2 > 10000 or co2 < 100:
                results['issues_detected'] = True
                results['issues'].append({
                    'type': 'scd41_co2_anomaly',
                    'value': co2,
                    'description': f'Extreme CO2 reading: {co2} ppm'
                })
    
    # Check data freshness
    if 'timestamp' in sensor_data:
        try:
            data_time = datetime.fromisoformat(sensor_data['timestamp'].replace('Z', '+00:00'))
            current_time = datetime.now(data_time.tzinfo)
            age_seconds = (current_time - data_time).total_seconds()
            
            # Alert if data is older than 5 minutes
            if age_seconds > 300:
                results['issues_detected'] = True
                results['issues'].append({
                    'type': 'data_stale',
                    'age_seconds': age_seconds,
                    'description': f'Data is {age_seconds} seconds old'
                })
        except Exception as e:
            logger.warning(f"Could not parse timestamp: {e}")
    
    return results

def send_alert(monitoring_results: Dict[str, Any]) -> None:
    """
    Send alert via SNS if issues are detected.
    
    Args:
        monitoring_results: Results from monitoring checks
    """
    
    try:
        sns = boto3.client('sns')
        
        # Use the correct SNS topic ARN
        topic_arn = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
        
        message = {
            'alert_type': 'system_monitoring',
            'timestamp': monitoring_results['timestamp'],
            'issues': monitoring_results['issues']
        }
        
        sns.publish(
            TopicArn=topic_arn,
            Message=json.dumps(message),
            Subject='Fire Detection System Monitoring Alert'
        )
        
        logger.info("Alert sent successfully")
        
    except Exception as e:
        logger.error(f"Failed to send alert: {str(e)}")