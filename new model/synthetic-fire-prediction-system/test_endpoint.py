#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Endpoint Test
This script tests the deployed endpoint with sample data.
"""

import boto3
import json

# AWS Configuration
AWS_REGION = 'us-east-1'

def test_endpoint(endpoint_name):
    """Test the deployed endpoint with sample data."""
    print(f"Testing endpoint: {endpoint_name}")
    print("=" * 40)
    
    # Initialize SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
    
    # Test cases
    test_cases = [
        {
            "name": "Normal conditions",
            "data": "0,22.5,1.2,25.1,24.8,0.5,0.3,0.1,0.05,0.2,0.1,0.3,0.1,23.0,0.5,0.1,410.0,5.0,1.0"
        },
        {
            "name": "Potential fire conditions",
            "data": "1,45.2,8.7,78.5,72.1,25.3,18.7,3.2,1.8,2.9,1.5,4.2,2.1,52.0,15.0,3.2,850.0,120.0,8.5"
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
            print(f"Input: {test_case['data']}")
            print(f"Prediction: {result}")
            
        except Exception as e:
            print(f"Error testing endpoint: {e}")

def main():
    """Main function to test the endpoint."""
    print("FLIR+SCD41 Fire Detection - Endpoint Test")
    print("=" * 45)
    
    # Endpoint name from our deployment
    endpoint_name = "flir-scd41-xgboost-model-20250828-155100-endpoint"
    
    # Test endpoint
    test_endpoint(endpoint_name)
    
    print("\n✅ Endpoint testing completed!")

if __name__ == "__main__":
    main()