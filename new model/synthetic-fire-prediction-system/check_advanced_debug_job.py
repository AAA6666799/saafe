#!/usr/bin/env python3
"""
Check the status of the advanced debug job
"""

import boto3

def check_advanced_debug_job():
    sagemaker = boto3.client('sagemaker', region_name='us-east-1')
    
    job_name = 'flir-scd41-advanced-debug-20250901-101500'
    
    try:
        response = sagemaker.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        secondary_status = response['SecondaryStatus']
        
        print(f"Job: {job_name}")
        print(f"Status: {status}")
        print(f"Secondary Status: {secondary_status}")
        
        if status == 'Failed':
            failure_reason = response.get('FailureReason', 'No failure reason provided')
            print(f"Failure Reason: {failure_reason}")
            
    except sagemaker.exceptions.ResourceNotFound:
        print(f"Job {job_name} not found")
    except Exception as e:
        print(f"Error checking job status: {e}")

if __name__ == "__main__":
    check_advanced_debug_job()