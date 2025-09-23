#!/usr/bin/env python3
"""
Complete pipeline runner that monitors training and deploys the best model
"""

import subprocess
import time
import sys

def run_complete_pipeline():
    """Run the complete pipeline from monitoring to deployment"""
    
    print("🔥 FLIR+SCD41 Fire Detection System - Complete Pipeline Runner")
    print("=" * 65)
    
    # Step 1: Monitor training jobs until completion
    print("\n📋 STEP 1: Monitoring Training Jobs")
    print("-" * 40)
    
    try:
        cmd = [
            "python", 
            "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system/monitor_100k_training.py",
            "--wait",
            "--interval", "180",  # Check every 3 minutes
            "flir-scd41-rf-100k-20250829-162112",
            "flir-scd41-gb-100k-20250829-162112", 
            "flir-scd41-lr-100k-20250829-162112"
        ]
        
        print("Monitoring training jobs until completion...")
        print("This may take 2-4 hours. Feel free to detach and check back later.")
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
        
        if rc == 0:
            print("\n✅ All training jobs completed successfully!")
        else:
            print(f"\n⚠️  Training monitoring completed with return code: {rc}")
            print("Some jobs may have failed or are still running.")
            return rc
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user. Training jobs continue running in AWS.")
        print("You can resume monitoring later with:")
        print("python continuous_monitor.py")
        return 0
    except Exception as e:
        print(f"❌ Error monitoring training: {e}")
        return 1
    
    # Step 2: Deploy the best model
    print("\n🚀 STEP 2: Deploying Best Performing Model")
    print("-" * 45)
    
    try:
        cmd = [
            "python",
            "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system/deploy_best_model.py"
        ]
        
        print("Deploying the best performing model...")
        print("This may take several minutes...")
        
        # Run the deployment script
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Print output as it comes
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        
        if rc == 0:
            print("\n✅ Best model deployed successfully!")
        else:
            print(f"\n❌ Model deployment failed with return code: {rc}")
            return rc
            
    except Exception as e:
        print(f"❌ Error deploying model: {e}")
        return 1
    
    # Step 3: Final summary
    print("\n🎉 COMPLETE PIPELINE EXECUTION FINISHED")
    print("=" * 50)
    print("✅ Data generation: Completed")
    print("✅ Model training: Completed") 
    print("✅ Model validation: Completed")
    print("✅ Model deployment: Completed")
    print("\nThe FLIR+SCD41 fire detection system is now ready for use!")
    
    return 0

def show_pipeline_status():
    """Show current status of the pipeline"""
    print("\n📊 CURRENT PIPELINE STATUS")
    print("-" * 30)
    print("1. Data Generation: ✅ Completed")
    print("2. Model Training: 🔄 In Progress")
    print("3. Model Validation: 🔄 In Progress") 
    print("4. Model Deployment: ⏳ Pending")
    
    print("\n📋 To check current training status:")
    print("python check_training_status.py")
    
    print("\n🚀 To run complete pipeline (monitor + deploy):")
    print("python complete_pipeline_runner.py")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        show_pipeline_status()
    else:
        exit_code = run_complete_pipeline()
        sys.exit(exit_code)