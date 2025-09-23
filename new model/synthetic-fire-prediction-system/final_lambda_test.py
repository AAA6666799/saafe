#!/usr/bin/env python3
"""
Final test script for deployed Lambda agents
"""

import boto3
import json
import time
from datetime import datetime

def test_lambda_function(function_name, payload):
    """Test a Lambda function with a given payload"""
    print(f"Testing {function_name}...")
    
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    try:
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        # Parse response
        response_payload = json.loads(response['Payload'].read())
        print(f"  Status: {response['StatusCode']}")
        print(f"  Response: {json.dumps(response_payload, indent=2)}")
        return response_payload
        
    except Exception as e:
        print(f"  Error: {e}")
        return None

def main():
    """Main function to test all Lambda agents"""
    print("Saafe Fire Detection System - Final Lambda Agent Testing")
    print("=" * 55)
    
    # Test Monitoring Agent
    monitoring_payload = {
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
    
    monitoring_result = test_lambda_function("saafe-monitoring-agent", monitoring_payload)
    
    # Test Response Agent
    response_payload = {
        "detection_result": {
            "confidence": 0.85,
            "fire_detected": True,
            "threat_level": "HIGH",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    response_result = test_lambda_function("saafe-response-agent", response_payload)
    
    # Test Analysis Agent
    analysis_payload = {
        "features": {
            "t_mean": 35.2,
            "t_std": 3.1,
            "t_max": 72.5,
            "t_p95": 65.0,
            "t_hot_area_pct": 15.0,
            "t_hot_largest_blob_pct": 8.0,
            "t_grad_mean": 2.1,
            "t_grad_std": 1.2,
            "t_diff_mean": 1.5,
            "t_diff_std": 0.8,
            "flow_mag_mean": 3.2,
            "flow_mag_std": 1.1,
            "tproxy_val": 25.0,
            "tproxy_delta": 5.0,
            "tproxy_vel": 0.5,
            "gas_val": 520.0,
            "gas_delta": 30.0,
            "gas_vel": 1.5
        }
    }
    
    # Wait a bit for the analysis agent to be fully updated
    print("\nWaiting for analysis agent to be ready...")
    time.sleep(10)
    
    analysis_result = test_lambda_function("saafe-analysis-agent", analysis_payload)
    
    # Summary
    print("\n" + "=" * 55)
    print("FINAL TEST SUMMARY")
    print("=" * 55)
    print(f"Monitoring Agent: {'PASS' if monitoring_result and monitoring_result.get('statusCode') == 200 else 'FAIL'}")
    print(f"Analysis Agent: {'PASS' if analysis_result and analysis_result.get('statusCode') == 200 else 'FAIL'}")
    print(f"Response Agent: {'PASS' if response_result and response_result.get('statusCode') == 200 else 'FAIL'}")
    
    if (monitoring_result and monitoring_result.get('statusCode') == 200 and
        analysis_result and analysis_result.get('statusCode') == 200 and
        response_result and response_result.get('statusCode') == 200):
        print("\n🎉 All Lambda agents are working correctly!")
        print("\n✅ AWS Lambda deployment completed successfully!")
        print("\nNext steps:")
        print("1. Set up SNS topics for alerts")
        print("2. Configure CloudWatch alarms")
        print("3. Set up event triggers for the analysis agent")
        print("4. Monitor CloudWatch logs for any issues")
    else:
        print("\n⚠️  Some agents failed. Check the output above for details.")

if __name__ == "__main__":
    main()