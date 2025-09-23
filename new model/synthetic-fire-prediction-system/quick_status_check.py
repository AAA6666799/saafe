#!/usr/bin/env python3
"""
Quick status check of training jobs
"""

import boto3
import sys

def quick_status_check():
    """Quick check of training job status"""
    sagemaker = boto3.client('sagemaker', region_name='us-east-1')
    
    # Updated job names from the latest pipeline execution
    job_names = [
        "flir-scd41-rf-100k-20250831-112304",
        "flir-scd41-gb-100k-20250831-112304", 
        "flir-scd41-lr-100k-20250831-112304"
    ]
    
    print("FLIR+SCD41 Fire Detection System - Quick Status Check")
    print("=" * 55)
    
    completed = 0
    failed = 0
    in_progress = 0
    
    from datetime import datetime
    print(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 30)
    
    for job_name in job_names:
        try:
            response = sagemaker.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            secondary_status = response.get('SecondaryStatus', 'N/A')
            print(f"{job_name}: {status} ({secondary_status})")
            
            if status == 'Completed':
                completed += 1
            elif status == 'Failed':
                failed += 1
            else:
                in_progress += 1
                
        except sagemaker.exceptions.ResourceNotFound:
            print(f"{job_name}: NOT FOUND")
            failed += 1
        except Exception as e:
            print(f"{job_name}: ERROR - {e}")
            failed += 1
    
    print("\n" + "=" * 30)
    print("SUMMARY")
    print("=" * 30)
    print(f"✅ Completed: {completed}")
    print(f"❌ Failed: {failed}")
    print(f"🔄 In Progress: {in_progress}")

if __name__ == "__main__":
    quick_status_check()