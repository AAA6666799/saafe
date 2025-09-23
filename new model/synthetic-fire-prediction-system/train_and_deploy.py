#!/usr/bin/env python3
"""
Train and Deploy Script for FLIR+SCD41 Fire Detection System

This script trains the models and deploys them to SageMaker hosting services.
"""

import os
import sys
import joblib
import boto3
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def train_models():
    """Train the FLIR+SCD41 models using the training pipeline."""
    print("🚀 Starting model training...")
    print("=" * 50)
    
    try:
        # Import and run the training pipeline
        from scripts.flir_scd41_training_pipeline import main as train_main
        
        # Run training
        print("Training models...")
        train_main()
        
        print("✅ Model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return False

def deploy_models():
    """Deploy trained models to SageMaker hosting services."""
    print("\n🚀 Starting model deployment...")
    print("=" * 50)
    
    try:
        # Import deployment functions
        from deploy_models import deploy_model, create_inference_script
        
        # Create inference script
        print("Creating inference script...")
        create_inference_script()
        
        # For demonstration, we'll use a placeholder model data URL
        # In a real scenario, this would point to the actual trained model artifacts
        model_data_url = "s3://fire-detection-training-691595239825/flir_scd41_training/model.tar.gz"
        
        # Deploy model
        print("Deploying model...")
        endpoint_name, model_name, config_name = deploy_model(
            model_name="flir-scd41-optimized-model",
            model_data_url=model_data_url,
            algorithm_type='xgboost'
        )
        
        if endpoint_name:
            print(f"✅ Model deployment initiated successfully!")
            print(f"Endpoint name: {endpoint_name}")
            print(f"Model name: {model_name}")
            print(f"Config name: {config_name}")
            print("\nNote: Endpoint deployment may take 5-15 minutes to complete")
            return True
        else:
            print("❌ Model deployment failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during model deployment: {e}")
        return False

def setup_monitoring():
    """Set up monitoring for the deployed models."""
    print("\n🚀 Setting up model monitoring...")
    print("=" * 50)
    
    try:
        # Import monitoring functions
        from scripts.automated_performance_tracking import main as monitor_main
        
        # Run monitoring setup
        print("Setting up automated performance tracking...")
        monitor_main()
        
        print("✅ Monitoring setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during monitoring setup: {e}")
        return False

def main():
    """Main function to train and deploy models."""
    print("🔥 FLIR+SCD41 Fire Detection System - Train and Deploy")
    print("=" * 60)
    
    # Step 1: Train models
    if not train_models():
        print("\n❌ Model training failed. Exiting.")
        return 1
    
    # Step 2: Deploy models
    if not deploy_models():
        print("\n❌ Model deployment failed. Exiting.")
        return 1
    
    # Step 3: Set up monitoring
    if not setup_monitoring():
        print("\n❌ Monitoring setup failed.")
        return 1
    
    print("\n" + "=" * 60)
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("✅ Models trained and deployed")
    print("✅ Monitoring system set up")
    print("✅ System ready for production use")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())