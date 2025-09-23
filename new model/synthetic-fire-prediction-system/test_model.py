#!/usr/bin/env python3
"""
Test the deployed FLIR+SCD41 fire detection model
"""

import boto3
import json
import datetime
import time

def test_model(endpoint_name):
    """Test the deployed model with sample data"""
    print("🧪 Testing FLIR+SCD41 fire detection model...")
    
    # Initialize SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
    
    # Sample test data (representing normal conditions)
    # Using all 18 features that were used in training
    normal_sample = {
        "t_mean": 25.5,
        "t_std": 3.2,
        "t_max": 35.0,
        "t_p95": 32.0,
        "t_hot_area_pct": 5.0,
        "t_hot_largest_blob_pct": 1.0,
        "t_grad_mean": 0.5,
        "t_grad_std": 0.8,
        "t_diff_mean": 1.0,
        "t_diff_std": 0.3,
        "flow_mag_mean": 2.0,
        "flow_mag_std": 0.5,
        "tproxy_val": 28.0,
        "tproxy_delta": -1.0,
        "tproxy_vel": -0.2,
        "gas_val": 400.0,
        "gas_delta": -20.0,
        "gas_vel": -5.0
    }
    
    # Sample test data (representing fire conditions)
    # Using all 18 features that were used in training
    fire_sample = {
        "t_mean": 45.5,
        "t_std": 15.2,
        "t_max": 75.0,
        "t_p95": 65.0,
        "t_hot_area_pct": 35.0,
        "t_hot_largest_blob_pct": 15.0,
        "t_grad_mean": 5.5,
        "t_grad_std": 3.8,
        "t_diff_mean": 8.0,
        "t_diff_std": 2.3,
        "flow_mag_mean": 6.0,
        "flow_mag_std": 2.5,
        "tproxy_val": 60.0,
        "tproxy_delta": -8.0,
        "tproxy_vel": -2.2,
        "gas_val": 800.0,
        "gas_delta": -150.0,
        "gas_vel": -35.0
    }
    
    try:
        print(f"📡 Sending test requests to endpoint: {endpoint_name}")
        
        # Test normal conditions
        print("\n🌤️  Testing normal conditions...")
        normal_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(normal_sample)
        )
        
        normal_result = json.loads(normal_response['Body'].read().decode())
        print(f"   Response: {json.dumps(normal_result, indent=2)}")
        
        # Test fire conditions
        print("\n🔥 Testing fire conditions...")
        fire_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(fire_sample)
        )
        
        fire_result = json.loads(fire_response['Body'].read().decode())
        print(f"   Response: {json.dumps(fire_result, indent=2)}")
        
        print("\n📊 Test Results Summary:")
        print(f"  Normal Conditions:")
        print(f"    Ensemble Prediction: {normal_result['ensemble_prediction']}")
        print(f"    Ensemble Probability: {normal_result['ensemble_probability']:.4f}")
        
        print(f"  Fire Conditions:")
        print(f"    Ensemble Prediction: {fire_result['ensemble_prediction']}")
        print(f"    Ensemble Probability: {fire_result['ensemble_probability']:.4f}")
        
        # Validate results
        if normal_result['ensemble_prediction'] == 0 and fire_result['ensemble_prediction'] == 1:
            print("\n✅ Model is working correctly!")
            print("   - Correctly identified normal conditions as non-fire")
            print("   - Correctly identified fire conditions as fire")
        else:
            print("\n⚠️  Model predictions may need review")
            print("   - Check if the model is properly trained")
        
        return normal_result, fire_result
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        raise

def wait_for_endpoint(endpoint_name, max_wait_time=600):
    """Wait for endpoint to be in service"""
    sagemaker = boto3.client('sagemaker', region_name='us-east-1')
    start_time = time.time()
    
    print(f"⏳ Waiting for endpoint {endpoint_name} to be in service...")
    
    while time.time() - start_time < max_wait_time:
        try:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            print(f"   Current status: {status}")
            
            if status == 'InService':
                print("✅ Endpoint is now in service!")
                return True
            elif status in ['Failed', 'OutOfService']:
                print(f"❌ Endpoint failed with status: {status}")
                return False
            else:
                print("   Waiting for endpoint to be ready...")
                time.sleep(30)  # Wait 30 seconds before checking again
        except Exception as e:
            print(f"❌ Error checking endpoint status: {e}")
            return False
    
    print("❌ Timeout waiting for endpoint to be ready")
    return False

def main():
    """Main function"""
    print("FLIR+SCD41 Fire Detection - Model Testing")
    print("=" * 45)
    
    # Use the endpoint name from our latest deployment
    endpoint_name = "flir-scd41-fire-detection-corrected-v3-20250901-121555"  # Updated with actual endpoint name
    
    try:
        # Wait for endpoint to be ready
        if not wait_for_endpoint(endpoint_name):
            print("💥 Endpoint is not ready for testing")
            return
            
        normal_result, fire_result = test_model(endpoint_name)
        print(f"\n🎉 Testing completed successfully!")
        
    except Exception as e:
        print(f"💥 Testing failed: {e}")
        print("💡 Make sure the endpoint is deployed and active before testing")
        raise

if __name__ == "__main__":
    main()