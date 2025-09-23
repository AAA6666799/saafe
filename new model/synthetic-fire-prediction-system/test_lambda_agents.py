#!/usr/bin/env python3
"""
Test script for deployed Lambda agents
"""

import boto3
import json
from datetime import datetime

def test_monitoring_agent():
    """Test the monitoring agent Lambda function"""
    print("Testing Monitoring Agent...")
    
    # Create sample sensor data
    sample_event = {
        "sensor_data": {
            "timestamp": datetime.now().isoformat(),
            "flir": {
                "temperature": 25.5
            },
            "scd41": {
                "co2_concentration": 420.0
            }
        }
    }
    
    # Invoke the Lambda function
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    try:
        response = lambda_client.invoke(
            FunctionName='saafe-monitoring-agent',
            InvocationType='RequestResponse',
            Payload=json.dumps(sample_event)
        )
        
        # Parse response
        response_payload = json.loads(response['Payload'].read())
        print(f"Monitoring Agent Response: {response_payload}")
        return response_payload
        
    except Exception as e:
        print(f"Error testing monitoring agent: {e}")
        return None

def test_analysis_agent():
    """Test the analysis agent Lambda function"""
    print("\nTesting Analysis Agent...")
    
    # Create sample sensor data for fire detection
    sample_event = {
        "sensor_data": {
            "t_mean": 35.2,
            "t_std": 3.1,
            "t_max": 72.5,
            "t_min": 18.3,
            "t_gradient": 0.8,
            "t_skewness": 0.2,
            "t_kurtosis": 2.1,
            "t_entropy": 0.7,
            "t_variance": 9.6,
            "t_range": 54.2,
            "t_mad": 2.3,
            "t_rms": 36.1,
            "t_energy": 1200.5,
            "t_centroid": 35.0,
            "t_slope": 0.1,
            "gas_val": 520.0,
            "gas_delta": 30.0,
            "gas_vel": 1.5
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
        print(f"Analysis Agent Response: {response_payload}")
        return response_payload
        
    except Exception as e:
        print(f"Error testing analysis agent: {e}")
        return None

def test_response_agent():
    """Test the response agent Lambda function"""
    print("\nTesting Response Agent...")
    
    # Create sample fire detection result
    sample_event = {
        "detection_result": {
            "confidence": 0.85,
            "fire_detected": True,
            "threat_level": "HIGH",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Invoke the Lambda function
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    try:
        response = lambda_client.invoke(
            FunctionName='saafe-response-agent',
            InvocationType='RequestResponse',
            Payload=json.dumps(sample_event)
        )
        
        # Parse response
        response_payload = json.loads(response['Payload'].read())
        print(f"Response Agent Response: {response_payload}")
        return response_payload
        
    except Exception as e:
        print(f"Error testing response agent: {e}")
        return None

def main():
    """Main function to test all Lambda agents"""
    print("Saafe Fire Detection System - Lambda Agent Testing")
    print("=" * 50)
    
    # Test each agent
    monitoring_result = test_monitoring_agent()
    analysis_result = test_analysis_agent()
    response_result = test_response_agent()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Monitoring Agent: {'PASS' if monitoring_result else 'FAIL'}")
    print(f"Analysis Agent: {'PASS' if analysis_result else 'FAIL'}")
    print(f"Response Agent: {'PASS' if response_result else 'FAIL'}")
    
    if monitoring_result and analysis_result and response_result:
        print("\n🎉 All Lambda agents are working correctly!")
    else:
        print("\n⚠️  Some agents failed. Check the output above for details.")

if __name__ == "__main__":
    main()