#!/usr/bin/env python3
"""
Test script to demonstrate monitoring of training jobs
"""

import subprocess
import sys

def test_monitoring():
    """Test the monitoring functionality"""
    # Training job names from the pipeline execution
    job_names = [
        "flir-scd41-rf-100k-20250829-160706",
        "flir-scd41-gb-100k-20250829-160706", 
        "flir-scd41-lr-100k-20250829-160706"
    ]
    
    print("Testing monitoring of training jobs...")
    print("=" * 50)
    
    # Try to monitor the jobs (without waiting)
    try:
        cmd = ["python", "monitor_100k_training.py"] + job_names
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Return code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("Monitoring command timed out (this is expected for long-running processes)")
    except Exception as e:
        print(f"Error running monitoring test: {e}")

if __name__ == "__main__":
    test_monitoring()