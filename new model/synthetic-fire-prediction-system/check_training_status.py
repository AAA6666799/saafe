#!/usr/bin/env python3
"""
Check the current status of training jobs to confirm they're running properly
"""

import boto3
from datetime import datetime

def check_training_status():
    """Check the status of our training jobs"""
    
    print("FLIR+SCD41 Fire Detection System - Training Job Status")
    print("=" * 55)
    
    # AWS Configuration
    AWS_REGION = 'us-east-1'
    
    # Training job names from the NEW pipeline execution (just created)
    job_names = [
        "flir-scd41-rf-100k-20250829-164912",
        "flir-scd41-gb-100k-20250829-164912", 
        "flir-scd41-lr-100k-20250829-164912"
    ]
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Track job status
    completed_jobs = []
    failed_jobs = []
    in_progress_jobs = []
    
    for job_name in job_names:
        try:
            response = sagemaker.describe_training_job(TrainingJobName=job_name)
            
            status = response['TrainingJobStatus']
            secondary_status = response.get('SecondaryStatus', 'N/A')
            creation_time = response.get('CreationTime', 'N/A')
            
            print(f"\n📋 Job: {job_name}")
            print(f"   Status: {status}")
            print(f"   Secondary: {secondary_status}")
            print(f"   Created: {creation_time}")
            
            # Collect status information
            if status == 'Completed':
                completed_jobs.append(job_name)
                training_time = response.get('TrainingTimeInSeconds', 'N/A')
                print(f"   🕐 Training time: {training_time} seconds")
            elif status == 'Failed':
                failed_jobs.append(job_name)
                failure_reason = response.get('FailureReason', 'Unknown')
                print(f"   ❌ Failure: {failure_reason}")
            elif status == 'InProgress':
                in_progress_jobs.append(job_name)
                print(f"   🔄 Job is currently running")
            else:
                print(f"   ℹ️  Other status: {status}")
                
        except sagemaker.exceptions.ResourceNotFound:
            print(f"\n❌ {job_name}: NOT FOUND")
        except Exception as e:
            print(f"\n❌ Error checking {job_name}: {e}")
    
    # Summary
    print("\n" + "=" * 55)
    print("📊 TRAINING STATUS SUMMARY")
    print("=" * 55)
    
    print(f"✅ Completed: {len(completed_jobs)} jobs")
    for job in completed_jobs:
        print(f"   • {job}")
    
    print(f"❌ Failed: {len(failed_jobs)} jobs")
    for job in failed_jobs:
        print(f"   • {job}")
    
    print(f"🔄 In Progress: {len(in_progress_jobs)} jobs")
    for job in in_progress_jobs:
        print(f"   • {job}")
    
    # Overall status
    if len(completed_jobs) == len(job_names):
        print("\n🎉 ALL TRAINING JOBS COMPLETED SUCCESSFULLY!")
        print("   You can now proceed with model deployment.")
    elif len(failed_jobs) > 0:
        print("\n❌ SOME JOBS FAILED!")
        print("   Check the error messages above for details.")
    elif len(in_progress_jobs) > 0:
        print(f"\n⏳ {len(in_progress_jobs)} JOBS STILL RUNNING")
        print("   Training is in progress. Check back later.")
        print("   Use 'python continuous_monitor.py' for continuous monitoring.")
    else:
        print("\n⚠️  UNEXPECTED STATUS")
        print("   Jobs are in an unexpected state.")

if __name__ == "__main__":
    check_training_status()