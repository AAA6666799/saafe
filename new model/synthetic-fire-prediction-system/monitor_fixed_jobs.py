#!/usr/bin/env python3
"""
Monitor the fixed training jobs
"""

import boto3
import time
import sys

def monitor_training_job(job_name, region='us-east-1'):
    """Monitor a training job until completion"""
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    print(f"Monitoring training job: {job_name}")
    print("=" * 50)
    
    completed_statuses = ['Completed', 'Failed', 'Stopped']
    
    while True:
        try:
            response = sagemaker.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            secondary_status = response.get('SecondaryStatus', 'N/A')
            
            print(f"{job_name}: {status} ({secondary_status})")
            
            if status in completed_statuses:
                if status == 'Completed':
                    print(f"✅ Training job completed successfully!")
                    print(f"Model artifacts saved to: {response['ModelArtifacts']['S3ModelArtifacts']}")
                elif status == 'Failed':
                    print(f"❌ Training job failed!")
                    failure_reason = response.get('FailureReason', 'No failure reason provided')
                    print(f"Failure reason: {failure_reason}")
                else:
                    print(f"⏹️  Training job stopped")
                break
            
            # Wait before checking again
            time.sleep(30)
            
        except sagemaker.exceptions.ResourceNotFound:
            print(f"Job {job_name} not found")
            break
        except Exception as e:
            print(f"Error monitoring job: {e}")
            time.sleep(30)

def list_recent_jobs():
    """List recent training jobs"""
    sagemaker = boto3.client('sagemaker', region_name='us-east-1')
    
    response = sagemaker.list_training_jobs(
        MaxResults=10,
        SortBy='CreationTime',
        SortOrder='Descending'
    )
    
    print("Recent Training Jobs:")
    print("=" * 50)
    
    fixed_jobs = []
    for job in response['TrainingJobSummaries']:
        job_name = job['TrainingJobName']
        status = job['TrainingJobStatus']
        creation_time = job['CreationTime']
        
        print(f"Job: {job_name}")
        print(f"  Status: {status}")
        print(f"  Created: {creation_time}")
        print()
        
        if 'fixed-ensemble' in job_name:
            fixed_jobs.append(job_name)
    
    return fixed_jobs

def main():
    """Main function"""
    print("🔍 Checking for fixed ensemble training jobs...")
    print()
    
    fixed_jobs = list_recent_jobs()
    
    if fixed_jobs:
        print(f"Found {len(fixed_jobs)} fixed ensemble job(s)")
        for job_name in fixed_jobs:
            print(f"\nMonitoring job: {job_name}")
            monitor_training_job(job_name)
    else:
        print("No fixed ensemble jobs found. Please create one first.")
        print("\nTo create a fixed ensemble training job, run:")
        print("python create_fixed_ensemble_job.py")

if __name__ == "__main__":
    main()