#!/usr/bin/env python3
"""
Deploy the best performing model once training is complete
"""

import boto3
import subprocess
import sys

def deploy_best_model():
    """Deploy the best performing model based on training results"""
    
    print("Preparing to deploy the best performing model...")
    print("=" * 50)
    
    # For now, we'll deploy the Random Forest model as an example
    # In a production scenario, you would select based on validation metrics
    
    model_s3_uri = "s3://fire-detection-training-691595239825/flir_scd41_training/models/flir-scd41-rf-100k-20250829-161531/output/model.tar.gz"
    model_name = "flir-scd41-rf-best-20250829"
    
    print(f"Model S3 URI: {model_s3_uri}")
    print(f"Model Name: {model_name}")
    
    try:
        # Deploy using our existing deployment script
        cmd = [
            "python",
            "/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system/deploy_100k_model.py",
            "--model-uri", model_s3_uri,
            "--model-name", model_name,
            "--instance-type", "ml.t2.medium",
            "--wait",
            "--test"
        ]
        
        print("\nDeploying model (this may take several minutes)...")
        print("Command:", " ".join(cmd))
        
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
            print("\n✅ Model deployed successfully!")
            print("The endpoint is now ready to receive predictions.")
        else:
            print(f"\n❌ Model deployment failed with return code: {rc}")
            
    except Exception as e:
        print(f"Error deploying model: {e}")

def show_deployment_instructions():
    """Show instructions for deploying models manually"""
    print("\n" + "=" * 60)
    print("MANUAL DEPLOYMENT INSTRUCTIONS")
    print("=" * 60)
    
    print("\nOnce training is complete, you can deploy any model using:")
    print("\n1. Deploy a single model:")
    print("python deploy_100k_model.py \\")
    print("  --model-uri s3://fire-detection-training-691595239825/flir_scd41_training/models/<JOB_NAME>/output/model.tar.gz \\")
    print("  --model-name my-fire-detection-model \\")
    print("  --instance-type ml.t2.medium \\")
    print("  --wait \\")
    print("  --test")
    
    print("\n2. Find completed job names:")
    print("aws sagemaker list-training-jobs --region us-east-1")
    
    print("\n3. Check model artifacts location:")
    print("aws sagemaker describe-training-job --training-job-name <JOB_NAME> --region us-east-1")

if __name__ == "__main__":
    print("FLIR+SCD41 Fire Detection System - Model Deployment")
    print("=" * 55)
    
    print("\nThis script will be ready to use once training is complete.")
    print("For now, showing deployment instructions...")
    
    show_deployment_instructions()
    
    # Uncomment the following line when you want to actually deploy
    # deploy_best_model()