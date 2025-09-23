#!/usr/bin/env python3
"""
Simple Training Progress Checker
This script checks the progress of training jobs and provides updates.
"""

import boto3
import time
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'

def check_training_progress(job_name):
    """Check the progress of a training job."""
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    try:
        # Get training job status
        response = sagemaker.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        secondary_status = response.get('SecondaryStatus', 'N/A')
        
        # Print status
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {job_name}: {status} ({secondary_status})")
        
        # If completed, show metrics
        if status == 'Completed':
            if 'FinalMetricDataList' in response:
                print("Final metrics:")
                for metric in response['FinalMetricDataList']:
                    print(f"  {metric['MetricName']}: {metric['Value']}")
            print(f"Model artifacts: {response['ModelArtifacts']['S3ModelArtifacts']}")
            
        # If failed, show reason
        elif status in ['Failed', 'Stopped']:
            if 'FailureReason' in response:
                print(f"Failure reason: {response['FailureReason']}")
                
        return status
        
    except Exception as e:
        print(f"Error checking training progress: {e}")
        return None

if __name__ == "__main__":
    # Training job name
    training_job_name = "flir-scd41-sklearn-training-20250828-153039"
    
    print("FLIR+SCD41 Fire Detection - Training Progress Checker")
    print("=" * 55)
    print(f"Monitoring job: {training_job_name}")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            status = check_training_progress(training_job_name)
            
            # If job is completed or failed, exit
            if status in ['Completed', 'Failed', 'Stopped']:
                break
                
            # Wait before next check
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError during monitoring: {e}")