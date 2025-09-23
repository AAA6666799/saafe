#!/usr/bin/env python3
"""
Monitor the new training jobs created with the fixed packaging
"""

import subprocess
import sys
import time

def monitor_new_jobs():
    """Monitor the newly created training jobs"""
    # Training job names from the new pipeline execution
    job_names = [
        "flir-scd41-rf-100k-20250829-161531",
        "flir-scd41-gb-100k-20250829-161531", 
        "flir-scd41-lr-100k-20250829-161531"
    ]
    
    print("Monitoring new training jobs with fixed packaging...")
    print("=" * 50)
    
    # Monitor the jobs with our monitoring script
    try:
        cmd = [
            "python", 
            "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system/monitor_100k_training.py"
        ] + job_names
        
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
        
    except Exception as e:
        print(f"Error running monitoring: {e}")

if __name__ == "__main__":
    monitor_new_jobs()