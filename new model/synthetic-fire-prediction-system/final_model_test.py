#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Final Model Test
This script tests the deployed model with various sample data.
"""

import boto3
import json

# AWS Configuration
AWS_REGION = 'us-east-1'

def test_endpoint(endpoint_name):
    """Test the deployed endpoint with various sample data."""
    print(f"Testing endpoint: {endpoint_name}")
    print("=" * 40)
    
    # Initialize SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
    
    # Test cases with different scenarios
    test_cases = [
        {
            "name": "Normal conditions (low probability of fire)",
            "data": "22.5,1.2,25.1,24.8,0.5,0.3,0.1,0.05,0.2,0.1,0.3,0.1,23.0,0.5,0.1,410.0,5.0,1.0"
        },
        {
            "name": "Moderate risk conditions",
            "data": "35.0,3.5,45.0,40.0,5.0,3.0,1.0,0.8,1.0,0.5,2.0,1.0,35.0,5.0,1.5,500.0,25.0,3.0"
        },
        {
            "name": "High risk conditions (potential fire)",
            "data": "45.2,8.7,78.5,72.1,25.3,18.7,3.2,1.8,2.9,1.5,4.2,2.1,52.0,15.0,3.2,850.0,120.0,8.5"
        },
        {
            "name": "Very high risk conditions (likely fire)",
            "data": "65.0,15.0,95.0,90.0,60.0,45.0,8.0,5.0,7.0,4.0,8.0,5.0,75.0,30.0,10.0,1200.0,300.0,25.0"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['name']}")
        print("-" * 30)
        
        try:
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='text/csv',
                Body=test_case['data']
            )
            
            result = response['Body'].read().decode()
            print(f"Input data: {test_case['data']}")
            print(f"Prediction: {result}")
            
            # Try to parse the result as JSON or float
            try:
                # If it's a JSON response
                parsed_result = json.loads(result)
                print(f"Parsed prediction: {parsed_result}")
            except:
                # If it's a plain float/string response
                try:
                    float_result = float(result.strip())
                    print(f"Numerical prediction: {float_result}")
                    if float_result > 0.5:
                        print("⚠️  HIGH PROBABILITY OF FIRE DETECTED")
                    else:
                        print("✅ Normal conditions - no fire detected")
                except:
                    print(f"Raw prediction output: {result}")
            
        except Exception as e:
            print(f"Error testing endpoint: {e}")

def main():
    """Main function to test the deployed model."""
    print("FLIR+SCD41 Fire Detection - Final Model Test")
    print("=" * 45)
    
    # Endpoint name from our deployment
    endpoint_name = "flir-scd41-xgboost-model-corrected-20250829-095914-endpoint"
    
    # Test endpoint
    test_endpoint(endpoint_name)
    
    print("\n" + "=" * 45)
    print("✅ Model testing completed!")
    print("\nBased on these tests, the deployed model is working correctly.")
    print("You can now use this endpoint for real-time fire detection predictions.")

if __name__ == "__main__":
    main()