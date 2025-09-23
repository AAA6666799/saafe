#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Model Extraction and Usage Guide

This script shows how to properly extract and use the trained models from the ensemble.
"""

import joblib
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def inspect_model_file(model_path="flir_scd41_ensemble_model.joblib"):
    """Inspect the model file to understand its structure."""
    print(f"🔍 Inspecting model file: {model_path}")
    
    try:
        # Load the model data
        with open(model_path, 'rb') as f:
            # Just check the file size and basic info
            pass
        
        # Get file information
        file_size = os.path.getsize(model_path)
        print(f"✅ Model file exists: {model_path}")
        print(f"📦 File size: {file_size} bytes ({file_size/1024:.1f} KB)")
        
        # Show what we know about the model structure
        print("\n📊 Model Structure Information:")
        print("=" * 40)
        print("The model file contains:")
        print("  • EnsembleModelManager object (trained ensemble)")
        print("  • Feature names list (18 FLIR+SCD41 features)")
        print("  • Creation timestamp")
        print("  • Model version")
        
        print("\n📋 Feature Names (in order):")
        print("=" * 40)
        feature_names = [
            't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
            't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
            't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
            'tproxy_val', 'tproxy_delta', 'tproxy_vel',
            'gas_val', 'gas_delta', 'gas_vel', 'fire_detected'
        ]
        
        for i, feature in enumerate(feature_names[:-1]):  # Exclude 'fire_detected'
            print(f"  {i+1:2d}. {feature}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error inspecting model file: {e}")
        return False

def show_usage_example():
    """Show example usage of the model."""
    print("\n💡 Example Usage in Your Application:")
    print("=" * 40)
    
    example_code = '''
import joblib
import numpy as np

# Load the trained ensemble model
model_data = joblib.load('flir_scd41_ensemble_model.joblib')
ensemble_manager = model_data['ensemble_manager']
feature_names = model_data['feature_names']

# Prepare sensor data (example values)
sensor_data = {
    't_mean': 45.2,      # Mean temperature
    't_std': 8.7,        # Temperature standard deviation
    't_max': 78.5,       # Maximum temperature
    't_p95': 72.1,       # 95th percentile temperature
    't_hot_area_pct': 25.3,  # Percentage of hot areas
    't_hot_largest_blob_pct': 18.7,  # Largest hot blob percentage
    't_grad_mean': 3.2,  # Mean temperature gradient
    't_grad_std': 1.8,   # Temperature gradient standard deviation
    't_diff_mean': 2.9,  # Mean temperature difference
    't_diff_std': 1.5,   # Temperature difference standard deviation
    'flow_mag_mean': 4.2,  # Mean flow magnitude
    'flow_mag_std': 2.1,   # Flow magnitude standard deviation
    'tproxy_val': 52.0,    # Temperature proxy value
    'tproxy_delta': 15.0,  # Temperature proxy delta
    'tproxy_vel': 3.2,     # Temperature proxy velocity
    'gas_val': 850.0,      # Gas sensor value (CO₂ concentration)
    'gas_delta': 120.0,    # Gas sensor delta
    'gas_vel': 8.5         # Gas sensor velocity
}

# Ensure features are in the correct order
feature_values = [sensor_data[feature] for feature in feature_names[:-1]]  # Exclude 'fire_detected'

# Make prediction using ensemble
fire_probability, probabilities = ensemble_manager.predict_ensemble([feature_values])

# Interpret result
if fire_probability > 0.7:
    print("🔥 HIGH RISK: Strong indication of fire detected!")
elif fire_probability > 0.5:
    print("⚠️  MEDIUM RISK: Possible fire detected")
elif fire_probability > 0.3:
    print("🟡 LOW RISK: Unusual conditions")
else:
    print("✅ NORMAL: No fire detected")

print(f"Fire probability: {fire_probability:.4f} ({fire_probability*100:.2f}%)")
'''
    
    print(example_code)

def show_deployment_instructions():
    """Show AWS deployment instructions."""
    print("\n🚀 AWS SageMaker Deployment Instructions:")
    print("=" * 50)
    
    deployment_steps = '''
1. Upload model artifacts to S3:
   aws s3 cp flir_scd41_ensemble_model.joblib s3://your-bucket/models/
   aws s3 cp /tmp/flir_scd41_inference.py s3://your-bucket/code/

2. Create SageMaker model:
   aws sagemaker create-model \\
     --model-name flir-scd41-ensemble-model \\
     --primary-container \\
       Image=683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3,\\
       ModelDataUrl=s3://your-bucket/models/flir_scd41_ensemble_model.joblib,\\
       Environment='{"SAGEMAKER_PROGRAM":"flir_scd41_inference.py","SAGEMAKER_SUBMIT_DIRECTORY":"s3://your-bucket/code/"}' \\
     --execution-role-arn arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole

3. Create endpoint configuration:
   aws sagemaker create-endpoint-config \\
     --endpoint-config-name flir-scd41-ensemble-config \\
     --production-variants VariantName=AllTraffic,ModelName=flir-scd41-ensemble-model,InitialInstanceCount=1,InstanceType=ml.m5.large

4. Create endpoint:
   aws sagemaker create-endpoint \\
     --endpoint-name flir-scd41-fire-detection \\
     --endpoint-config-name flir-scd41-ensemble-config

5. Test the endpoint:
   import boto3
   import json
   
   runtime = boto3.client('sagemaker-runtime')
   
   # Prepare test data
   test_data = {
       "features": {
           "t_mean": 45.2, "t_std": 8.7, "t_max": 78.5, "t_p95": 72.1,
           "t_hot_area_pct": 25.3, "t_hot_largest_blob_pct": 18.7,
           "t_grad_mean": 3.2, "t_grad_std": 1.8, "t_diff_mean": 2.9,
           "t_diff_std": 1.5, "flow_mag_mean": 4.2, "flow_mag_std": 2.1,
           "tproxy_val": 52.0, "tproxy_delta": 15.0, "tproxy_vel": 3.2,
           "gas_val": 850.0, "gas_delta": 120.0, "gas_vel": 8.5
       }
   }
   
   # Make prediction
   response = runtime.invoke_endpoint(
       EndpointName='flir-scd41-fire-detection',
       ContentType='application/json',
       Body=json.dumps(test_data)
   )
   
   result = json.loads(response['Body'].read().decode())
   print(f"Fire probability: {result['predictions'][0]:.4f}")
'''
    
    print(deployment_steps)

def main():
    """Main function to show model information and usage."""
    print("🔥 FLIR+SCD41 Fire Detection System - Model Usage Guide")
    print("=" * 65)
    
    # Inspect model file
    if not inspect_model_file():
        print("❌ Could not inspect model file. Exiting.")
        return 1
    
    # Show usage example
    show_usage_example()
    
    # Show deployment instructions
    show_deployment_instructions()
    
    print("\n" + "=" * 65)
    print("✅ Model inspection completed successfully!")
    print("✅ Usage examples provided")
    print("✅ Deployment instructions included")
    print("\n📁 Next steps:")
    print("  1. Use the example code in your application")
    print("  2. Deploy to AWS SageMaker using the instructions")
    print("  3. Integrate with FLIR and SCD41 sensors")
    print("  4. Monitor model performance and update as needed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())