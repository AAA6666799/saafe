#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Monitor 100K Training Jobs
This script monitors the status of SageMaker training jobs for 100K samples.
"""

import boto3
import argparse
import time
import sys
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'

def monitor_training_jobs(job_names, wait=False, check_interval=60):
    """
    Monitor the status of training jobs.
    
    Args:
        job_names (list): List of training job names to monitor
        wait (bool): Whether to wait until all jobs complete
        check_interval (int): Seconds between status checks when waiting
    """
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    print(f"🔍 Monitoring {len(job_names)} training jobs...")
    print("=" * 50)
    
    completed_jobs = []
    failed_jobs = []
    in_progress_jobs = job_names.copy()
    
    while in_progress_jobs:
        print(f"\\n🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 30)
        
        # Check status of each job
        for job_name in in_progress_jobs[:]:  # Create a copy to iterate over
            try:
                response = sagemaker.describe_training_job(
                    TrainingJobName=job_name
                )
                
                status = response['TrainingJobStatus']
                print(f"{job_name}: {status}")
                
                # Get additional info for completed/failed jobs
                if status == 'Completed':
                    completed_jobs.append(job_name)
                    in_progress_jobs.remove(job_name)
                    training_time = response.get('TrainingTimeInSeconds', 'N/A')
                    print(f"  🕐 Training time: {training_time} seconds")
                    
                elif status == 'Failed':
                    failed_jobs.append(job_name)
                    in_progress_jobs.remove(job_name)
                    failure_reason = response.get('FailureReason', 'Unknown')
                    print(f"  ❌ Failure reason: {failure_reason}")
                    
                elif status == 'InProgress':
                    # Show training progress if available
                    secondary_status = response.get('SecondaryStatus', 'N/A')
                    print(f"  🔄 Secondary status: {secondary_status}")
                    
            except Exception as e:
                print(f"❌ Error checking {job_name}: {e}")
                # Keep job in monitoring list to retry
        
        # Print summary
        print(f"\\n📊 Status Summary:")
        print(f"  Completed: {len(completed_jobs)}")
        print(f"  Failed: {len(failed_jobs)}")
        print(f"  In Progress: {len(in_progress_jobs)}")
        
        # Exit if not waiting for completion
        if not wait:
            break
        
        # Exit if all jobs completed or failed
        if not in_progress_jobs:
            break
            
        # Wait before next check
        print(f"\\n⏳ Waiting {check_interval} seconds before next check...")
        time.sleep(check_interval)
    
    # Final summary
    print("\\n" + "=" * 50)
    print("FINAL STATUS SUMMARY")
    print("=" * 50)
    
    if completed_jobs:
        print("✅ COMPLETED JOBS:")
        for job_name in completed_jobs:
            print(f"  • {job_name}")
    
    if failed_jobs:
        print("❌ FAILED JOBS:")
        for job_name in failed_jobs:
            print(f"  • {job_name}")
    
    if in_progress_jobs:
        print("🔄 IN PROGRESS JOBS:")
        for job_name in in_progress_jobs:
            print(f"  • {job_name}")
    
    return {
        'completed': completed_jobs,
        'failed': failed_jobs,
        'in_progress': in_progress_jobs
    }

def get_job_metrics(job_name):
    """
    Get detailed metrics for a completed training job.
    
    Args:
        job_name (str): Name of the training job
    """
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    try:
        response = sagemaker.describe_training_job(
            TrainingJobName=job_name
        )
        
        print(f"\\n📈 Metrics for {job_name}:")
        print("-" * 30)
        
        # Training time
        training_time = response.get('TrainingTimeInSeconds', 'N/A')
        print(f"🕐 Training time: {training_time} seconds")
        
        # Billable time
        billable_time = response.get('BillableTimeInSeconds', 'N/A')
        print(f"💰 Billable time: {billable_time} seconds")
        
        # Instance type
        resource_config = response.get('ResourceConfig', {})
        instance_type = resource_config.get('InstanceType', 'N/A')
        instance_count = resource_config.get('InstanceCount', 'N/A')
        print(f"🖥️  Instance: {instance_count} x {instance_type}")
        
        # Model artifacts
        model_artifacts = response.get('ModelArtifacts', {})
        s3_model_uri = model_artifacts.get('S3ModelArtifacts', 'N/A')
        print(f"📦 Model artifacts: {s3_model_uri}")
        
        # Final metrics (if available)
        final_metric_data = response.get('FinalMetricDataList', [])
        if final_metric_data:
            print("\\n📊 Final Metrics:")
            for metric in final_metric_data:
                metric_name = metric.get('MetricName', 'N/A')
                metric_value = metric.get('Value', 'N/A')
                print(f"  {metric_name}: {metric_value}")
        
    except Exception as e:
        print(f"❌ Error getting metrics for {job_name}: {e}")

def main():
    """Main function to monitor training jobs."""
    parser = argparse.ArgumentParser(description='Monitor FLIR+SCD41 training jobs')
    parser.add_argument('job_names', nargs='+', help='Names of training jobs to monitor')
    parser.add_argument('--wait', action='store_true', help='Wait until all jobs complete')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds (default: 60)')
    parser.add_argument('--metrics', action='store_true', help='Show detailed metrics for completed jobs')
    
    args = parser.parse_args()
    
    print("🔥 FLIR+SCD41 Fire Detection System - Training Job Monitor")
    print("=" * 60)
    
    # Monitor jobs
    results = monitor_training_jobs(args.job_names, args.wait, args.interval)
    
    # Show detailed metrics if requested
    if args.metrics and results['completed']:
        print("\\n" + "=" * 60)
        print("DETAILED METRICS")
        print("=" * 60)
        
        for job_name in results['completed']:
            get_job_metrics(job_name)
    
    # Exit with appropriate code
    if results['failed']:
        print(f"\\n❌ {len(results['failed'])} job(s) failed")
        return 1
    elif results['in_progress']:
        print(f"\\n🔄 {len(results['in_progress'])} job(s) still in progress")
        return 0
    else:
        print(f"\\n✅ All {len(results['completed'])} job(s) completed successfully")
        return 0

if __name__ == "__main__":
    sys.exit(main())