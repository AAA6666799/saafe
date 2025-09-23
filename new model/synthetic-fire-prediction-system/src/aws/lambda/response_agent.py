"""
AWS Lambda function for Response Agent implementation.
This function handles emergency responses based on fire detection results.
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
    AWS Lambda function for Response Agent.
    
    Args:
        event: Event data containing fire detection results
        context: Lambda context object
        
    Returns:
        Response dictionary with action results
    """
    
    try:
        logger.info(f"Response Agent triggered with event: {json.dumps(event)}")
        
        # Extract detection result from event
        detection_result = event.get('detection_result', {})
        
        # Determine response level based on confidence
        response_level = determine_response_level(detection_result)
        
        # Execute appropriate response actions
        actions_taken = execute_response_actions(response_level, detection_result)
        
        # Send notification about response actions
        send_response_notification(response_level, actions_taken)
        
        # Log response actions
        logger.info(f"Response actions completed: {json.dumps(actions_taken)}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Response actions completed successfully',
                'timestamp': datetime.now().isoformat(),
                'response_level': response_level,
                'actions_taken': actions_taken
            })
        }
        
    except Exception as e:
        logger.error(f"Error in response agent: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }

def determine_response_level(detection_result: Dict[str, Any]) -> str:
    """
    Determine response level based on detection confidence.
    
    Args:
        detection_result: Dictionary containing fire detection results
        
    Returns:
        Response level as string (NONE/LOW/MEDIUM/HIGH/CRITICAL)
    """
    
    confidence = detection_result.get('confidence', 0.0)
    fire_detected = detection_result.get('fire_detected', False)
    
    if not fire_detected:
        return 'NONE'
    elif confidence < 0.5:
        return 'LOW'
    elif confidence < 0.7:
        return 'MEDIUM'
    elif confidence < 0.9:
        return 'HIGH'
    else:
        return 'CRITICAL'

def execute_response_actions(response_level: str, detection_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute response actions based on response level.
    
    Args:
        response_level: Level of response to execute
        detection_result: Detection results that triggered the response
        
    Returns:
        Dictionary with actions taken
    """
    
    actions = {
        'timestamp': datetime.now().isoformat(),
        'response_level': response_level,
        'actions': []
    }
    
    if response_level == 'NONE':
        actions['actions'].append('no_action_required')
    elif response_level == 'LOW':
        actions['actions'].extend([
            'log_detection_event',
            'increase_monitoring_frequency'
        ])
    elif response_level == 'MEDIUM':
        actions['actions'].extend([
            'log_detection_event',
            'increase_monitoring_frequency',
            'send_alert_to_operations'
        ])
    elif response_level == 'HIGH':
        actions['actions'].extend([
            'log_detection_event',
            'increase_monitoring_frequency',
            'send_alert_to_operations',
            'notify_emergency_personnel',
            'activate_pre_emergency_protocols'
        ])
    elif response_level == 'CRITICAL':
        actions['actions'].extend([
            'log_detection_event',
            'increase_monitoring_frequency',
            'send_alert_to_operations',
            'notify_emergency_personnel',
            'activate_pre_emergency_protocols',
            'initiate_evacuation_procedures'
        ])
    
    return actions

def send_response_notification(response_level: str, actions_taken: Dict[str, Any]) -> None:
    """
    Send response notification via SNS.
    
    Args:
        response_level: Level of response executed
        actions_taken: Actions that were taken
    """
    
    try:
        sns = boto3.client('sns')
        
        # Use the correct SNS topic ARN
        topic_arn = 'arn:aws:sns:us-east-1:691595239825:fire-detection-emergency-response'
        
        message = {
            'notification_type': 'emergency_response',
            'timestamp': actions_taken['timestamp'],
            'response_level': response_level,
            'actions': actions_taken['actions']
        }
        
        sns.publish(
            TopicArn=topic_arn,
            Message=json.dumps(message),
            Subject=f'Fire Detection Emergency Response - Level {response_level}'
        )
        
        logger.info("Response notification sent successfully")
        
    except Exception as e:
        logger.error(f"Failed to send response notification: {str(e)}")