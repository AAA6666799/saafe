#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Endpoint Monitor
This script monitors the status of a deployed endpoint until it's ready for use.
"""

import boto3
import time
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'

def monitor_endpoint(endpoint_name, check_interval=60):
    """Monitor endpoint status until it's ready."""
    print(f"Monitoring endpoint: {endpoint_name}")
    print("=" * 50)
    
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    last_status = None
    
    while True:
        try:
            # Get endpoint status
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            # Print status updates
            if status != last_status:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Endpoint status: {status}")
                last_status = status
            
            # Check if endpoint is ready
            if status == 'InService':
                print(f"\nEndpoint is ready for use!")
                print(f"Endpoint name: {endpoint_name}")
                return True
            elif status in ['Failed', 'OutOfService']:
                failure_reason = response.get('FailureReason', 'No reason provided')
                print(f"\nEndpoint creation failed: {failure_reason}")
                return False
            
            # Wait before next check
            time.sleep(check_interval)
            
        except Exception as e:
            print(f"Error monitoring endpoint: {e}")
            return False

def main():
    """Main function to monitor endpoint."""
    print("FLIR+SCD41 Fire Detection - Endpoint Monitor")
    print("=" * 45)
    
    # Endpoint name from our deployment
    endpoint_name = "flir-scd41-xgboost-model-20250828-155100-endpoint"
    
    # Monitor endpoint
    success = monitor_endpoint(endpoint_name, check_interval=30)
    
    if success:
        print("\n🎉 Endpoint is ready for real-time fire detection predictions!")
        print("\nTo test the endpoint, you can use the test_endpoint.py script.")
    else:
        print("\n❌ Endpoint creation failed. Please check the SageMaker console for details.")

if __name__ == "__main__":
    main()