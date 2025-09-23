#!/usr/bin/env python3
"""
Monitor the latest training jobs until completion
"""

import subprocess
import sys
import time

def monitor_latest_jobs():
    """Monitor the latest training jobs"""
    # Training job names from the latest pipeline execution
    job_names = [
        "flir-scd41-rf-100k-20250829-164341",
        "flir-scd41-gb-100k-20250829-164341", 
        "flir-scd41-lr-100k-20250829-164341"
    ]
    
    print("Monitoring latest training jobs...")
    print("=" * 50)
    
    # Monitor the jobs with our monitoring script
    try:
        cmd = [
            "python", 
            "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system/monitor_100k_training.py",
            "--wait",
            "--interval", "180"  # Check every 3 minutes
        ] + job_names
        
        print("Monitoring jobs (will check every 3 minutes until completion)...")
        print("This will take 2-4 hours. Feel free to detach and check back later.")
        print("Press Ctrl+C to stop monitoring (training will continue in AWS)...")
        
        # Run the monitoring script
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Print output as it comes
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        print(f"\nMonitoring process completed with return code: {rc}")
        
        if rc == 0:
            print("✅ All training jobs completed successfully!")
            print("\nNext steps:")
            print("1. Check the model artifacts in S3")
            print("2. Deploy the best performing model using deploy_100k_model.py")
            print("3. Test the deployed endpoint")
        else:
            print("⚠️  Some training jobs failed or are still in progress")
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user. Training jobs continue running in AWS.")
    except Exception as e:
        print(f"Error running monitoring: {e}")

if __name__ == "__main__":
    monitor_latest_jobs()