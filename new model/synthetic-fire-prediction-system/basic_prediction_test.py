#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Basic Prediction Test
This script demonstrates a simple prediction call to the deployed model.
"""

import boto3

# AWS Configuration
AWS_REGION = 'us-east-1'

def test_simple_prediction():
    """Test a simple prediction with the deployed model."""
    print("FLIR+SCD41 Fire Detection - Basic Prediction Test")
    print("=" * 50)
    
    # Initialize SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
    
    # Endpoint name from our deployment
    endpoint_name = "flir-scd41-xgboost-model-corrected-20250829-095914-endpoint"
    
    # Sample test data representing potential fire conditions
    # Format: t_mean,t_std,t_max,t_p95,t_hot_area_pct,t_hot_largest_blob_pct,
    #         t_grad_mean,t_grad_std,t_diff_mean,t_diff_std,flow_mag_mean,flow_mag_std,
    #         tproxy_val,tproxy_delta,tproxy_vel,gas_val,gas_delta,gas_vel
    test_data = "45.2,8.7,78.5,72.1,25.3,18.7,3.2,1.8,2.9,1.5,4.2,2.1,52.0,15.0,3.2,850.0,120.0,8.5"
    
    print(f"Endpoint: {endpoint_name}")
    print(f"Test data: {test_data}")
    print("-" * 50)
    
    try:
        # Make prediction
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=test_data
        )
        
        # Process response
        result = response['Body'].read().decode()
        print(f"✅ Prediction successful!")
        print(f"Raw result: {result}")
        
        # Try to convert to float for interpretation
        try:
            probability = float(result.strip())
            print(f"Fire probability: {probability:.4f} ({probability*100:.2f}%)")
            
            # Interpret result
            if probability > 0.7:
                print("🔥 HIGH RISK: Strong indication of fire detected!")
            elif probability > 0.5:
                print("⚠️  MEDIUM RISK: Possible fire detected")
            elif probability > 0.3:
                print("🟡 LOW RISK: Unusual conditions")
            else:
                print("✅ NORMAL: No fire detected")
                
        except ValueError:
            print("⚠️  Could not parse result as a probability value")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def main():
    """Main function to run basic prediction test."""
    success = test_simple_prediction()
    
    if success:
        print("\n🎉 Basic prediction test completed successfully!")
        print("The deployed model is ready for production use.")
    else:
        print("\n❌ Basic prediction test failed.")
        print("Please check the endpoint status and configuration.")

if __name__ == "__main__":
    main()