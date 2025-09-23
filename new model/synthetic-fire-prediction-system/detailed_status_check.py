#!/usr/bin/env python3
"""
Detailed status check of training jobs
"""

import boto3
import sys

def detailed_status_check():
    """Detailed check of training job status"""
    sagemaker = boto3.client('sagemaker', region_name='us-east-1')
    
    # Updated job names from the latest pipeline execution
    job_names = [
        "flir-scd41-rf-100k-20250831-112304",
        "flir-scd41-gb-100k-20250831-112304", 
        "flir-scd41-lr-100k-20250831-112304"
    ]
    
    print("FLIR+SCD41 Fire Detection System - Detailed Training Job Status")
    print("=" * 65)
    
    for job_name in job_names:
        try:
            response = sagemaker.describe_training_job(TrainingJobName=job_name)
            
            status = response['TrainingJobStatus']
            secondary_status = response.get('SecondaryStatus', 'N/A')
            creation_time = response.get('CreationTime', 'N/A')
            
            print(f"\nJob Name: {job_name}")
            print(f"Primary Status: {status}")
            print(f"Secondary Status: {secondary_status}")
            print(f"Creation Time: {creation_time}")
            
            # Additional details based on status
            if status == 'InProgress':
                print("✅ Job is running normally")
            elif status == 'Completed':
                print("✅ Job completed successfully")
            elif status == 'Failed':
                failure_reason = response.get('FailureReason', 'No reason provided')
                print(f"❌ Job failed: {failure_reason}")
            else:
                print(f"ℹ️  Job status: {status}")
                
        except sagemaker.exceptions.ResourceNotFound:
            print(f"\n❌ {job_name}: NOT FOUND")
        except Exception as e:
            print(f"\n❌ Error checking {job_name}: {e}")

if __name__ == "__main__":
    detailed_status_check()