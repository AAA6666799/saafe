#!/usr/bin/env python3
"""
Check status of all recent training jobs
"""

import boto3
from datetime import datetime

def check_all_training_jobs():
    """Check status of all recent training jobs"""
    sagemaker = boto3.client('sagemaker', region_name='us-east-1')
    
    print("FLIR+SCD41 Fire Detection System - Training Jobs Status")
    print("=" * 60)
    
    try:
        response = sagemaker.list_training_jobs(
            MaxResults=20,
            SortBy='CreationTime',
            SortOrder='Descending'
        )
        
        job_count = 0
        completed_count = 0
        failed_count = 0
        in_progress_count = 0
        
        for job in response['TrainingJobSummaries']:
            job_name = job['TrainingJobName']
            status = job['TrainingJobStatus']
            creation_time = job['CreationTime']
            
            # Count job statuses
            job_count += 1
            if status == 'Completed':
                completed_count += 1
            elif status == 'Failed':
                failed_count += 1
            else:
                in_progress_count += 1
            
            # Format creation time
            formatted_time = creation_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(creation_time, datetime) else str(creation_time)
            
            # Add status indicator
            status_indicator = {
                'Completed': '✅',
                'Failed': '❌',
                'InProgress': '🔄',
                'Stopping': '⏹️',
                'Stopped': '⏹️'
            }.get(status, '❓')
            
            print(f"{status_indicator} {job_name}")
            print(f"   Status: {status}")
            print(f"   Created: {formatted_time}")
            print()
        
        # Print summary
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total Jobs: {job_count}")
        print(f"Completed: {completed_count} ✅")
        print(f"Failed: {failed_count} ❌")
        print(f"In Progress: {in_progress_count} 🔄")
        
        # Success rate
        if job_count > 0:
            success_rate = (completed_count / job_count) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"Error checking training jobs: {e}")

if __name__ == "__main__":
    check_all_training_jobs()