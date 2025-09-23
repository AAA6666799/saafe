#!/usr/bin/env python3
"""
Check the latest training jobs
"""

import boto3

def check_latest_jobs():
    sagemaker = boto3.client('sagemaker', region_name='us-east-1')
    
    response = sagemaker.list_training_jobs(
        MaxResults=10,
        SortBy='CreationTime',
        SortOrder='Descending'
    )
    
    print("Latest Training Jobs:")
    print("=" * 50)
    
    for job in response['TrainingJobSummaries']:
        job_name = job['TrainingJobName']
        status = job['TrainingJobStatus']
        creation_time = job['CreationTime']
        
        print(f"Job: {job_name}")
        print(f"  Status: {status}")
        print(f"  Created: {creation_time}")
        print()

if __name__ == "__main__":
    check_latest_jobs()