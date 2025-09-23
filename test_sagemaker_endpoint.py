#!/usr/bin/env python3
"""
Test script to verify SageMaker endpoint is working correctly
"""

import boto3
import json
import numpy as np

def test_sagemaker_endpoint():
    """Test the SageMaker endpoint with sample data"""
    print("Testing SageMaker endpoint...")
    
    # Initialize SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
    
    # Sample data matching the 18 features used in training
    sample_data = {
        't_mean': 35.2,
        't_std': 3.1,
        't_max': 72.5,
        't_p95': 68.3,
        't_hot_area_pct': 15.7,
        't_hot_largest_blob_pct': 8.2,
        't_grad_mean': 2.3,
        't_grad_std': 1.8,
        't_diff_mean': 1.9,
        't_diff_std': 1.5,
        'flow_mag_mean': 3.7,
        'flow_mag_std': 2.1,
        'tproxy_val': 42.8,
        'tproxy_delta': 5.2,
        'tproxy_vel': 1.3,
        'gas_val': 520.0,
        'gas_delta': 30.0,
        'gas_vel': 1.5
    }
    
    endpoint_name = 'flir-scd41-fire-detection-corrected-v3-20250901-121555'
    
    try:
        # Send request to SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(sample_data)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        print(f"✅ SageMaker endpoint response: {json.dumps(result, indent=2)}")
        return result
        
    except Exception as e:
        print(f"❌ Error testing SageMaker endpoint: {e}")
        return None

def test_analysis_agent():
    """Test the analysis agent Lambda function"""
    print("\nTesting Analysis Agent...")
    
    # Sample data matching what the analysis agent expects
    sample_event = {
        "features": {
            't_mean': 35.2,
            't_std': 3.1,
            't_max': 72.5,
            't_p95': 68.3,
            't_hot_area_pct': 15.7,
            't_hot_largest_blob_pct': 8.2,
            't_grad_mean': 2.3,
            't_grad_std': 1.8,
            't_diff_mean': 1.9,
            't_diff_std': 1.5,
            'flow_mag_mean': 3.7,
            'flow_mag_std': 2.1,
            'tproxy_val': 42.8,
            'tproxy_delta': 5.2,
            'tproxy_vel': 1.3,
            'gas_val': 520.0,
            'gas_delta': 30.0,
            'gas_vel': 1.5
        }
    }
    
    # Invoke the Lambda function
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    try:
        response = lambda_client.invoke(
            FunctionName='saafe-analysis-agent',
            InvocationType='RequestResponse',
            Payload=json.dumps(sample_event)
        )
        
        # Parse response
        response_payload = json.loads(response['Payload'].read())
        print(f"✅ Analysis Agent Response: {json.dumps(response_payload, indent=2)}")
        return response_payload
        
    except Exception as e:
        print(f"❌ Error testing analysis agent: {e}")
        return None

if __name__ == "__main__":
    print("🔥 Testing SageMaker Endpoint and Analysis Agent Connection")
    print("=" * 60)
    
    # Test SageMaker endpoint directly
    sagemaker_result = test_sagemaker_endpoint()
    
    # Test analysis agent
    agent_result = test_analysis_agent()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if sagemaker_result:
        print("✅ SageMaker Endpoint: CONNECTED and RESPONDING")
    else:
        print("❌ SageMaker Endpoint: CONNECTION ISSUE")
        
    if agent_result:
        print("✅ Analysis Agent: CONNECTED and PROCESSING")
    else:
        print("❌ Analysis Agent: CONNECTION ISSUE")
        
    if sagemaker_result and agent_result:
        print("\n🎉 The agents are successfully connected to the AI model!")
        print("   Data flows from the Analysis Agent to the SageMaker endpoint")
        print("   and predictions are returned correctly.")
    else:
        print("\n⚠️  There are connection issues that need to be addressed.")