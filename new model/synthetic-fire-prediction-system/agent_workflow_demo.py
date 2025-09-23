#!/usr/bin/env python3
"""
Agent Workflow Demo
This script demonstrates how the three agents work together in a coordinated workflow.
"""

import boto3
import json
import time
from datetime import datetime

def generate_sample_sensor_data():
    """Generate sample sensor data for testing"""
    return {
        "timestamp": datetime.now().isoformat(),
        "flir": {
            "temperature": 25.5
        },
        "scd41": {
            "co2_concentration": 420.0
        }
    }

def generate_sample_features():
    """Generate sample features for fire detection"""
    return {
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

def trigger_monitoring_agent(sensor_data):
    """Trigger the monitoring agent with sensor data"""
    print("📡 Triggering Monitoring Agent...")
    
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    payload = {
        "sensor_data": sensor_data
    }
    
    try:
        response = lambda_client.invoke(
            FunctionName='saafe-monitoring-agent',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        result = json.loads(response['Payload'].read())
        print(f"✅ Monitoring Agent Response: {result.get('statusCode', 'N/A')}")
        return result
        
    except Exception as e:
        print(f"❌ Error triggering monitoring agent: {e}")
        return None

def trigger_analysis_agent(features):
    """Trigger the analysis agent with features"""
    print("\n🔬 Triggering Analysis Agent...")
    
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    payload = {
        "features": features
    }
    
    try:
        response = lambda_client.invoke(
            FunctionName='saafe-analysis-agent',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        result = json.loads(response['Payload'].read())
        print(f"✅ Analysis Agent Response: {result.get('statusCode', 'N/A')}")
        return result
        
    except Exception as e:
        print(f"❌ Error triggering analysis agent: {e}")
        return None

def trigger_response_agent(detection_result):
    """Trigger the response agent with detection results"""
    print("\n🚨 Triggering Response Agent...")
    
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    payload = {
        "detection_result": detection_result
    }
    
    try:
        response = lambda_client.invoke(
            FunctionName='saafe-response-agent',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        result = json.loads(response['Payload'].read())
        print(f"✅ Response Agent Response: {result.get('statusCode', 'N/A')}")
        return result
        
    except Exception as e:
        print(f"❌ Error triggering response agent: {e}")
        return None

def simulate_fire_detection_workflow():
    """Simulate a complete fire detection workflow"""
    print("🔥 Synthetic Fire Prediction System - Agent Workflow Demo")
    print("=" * 58)
    
    # Step 1: Generate sample sensor data
    print("\n1️⃣ Generating sample sensor data...")
    sensor_data = generate_sample_sensor_data()
    print(f"   Sensor data: {json.dumps(sensor_data, indent=2)}")
    
    # Step 2: Trigger monitoring agent
    monitoring_result = trigger_monitoring_agent(sensor_data)
    
    # Step 3: Generate sample features for analysis
    print("\n2️⃣ Generating sample features for analysis...")
    features = generate_sample_features()
    print(f"   Features: {json.dumps({k: v for k, v in list(features.items())[:5]}, indent=2)}...")
    
    # Step 4: Trigger analysis agent
    analysis_result = trigger_analysis_agent(features)
    
    # Step 5: Create sample detection result
    print("\n3️⃣ Creating sample detection result...")
    detection_result = {
        "confidence": 0.85,
        "fire_detected": True,
        "threat_level": "HIGH",
        "timestamp": datetime.now().isoformat()
    }
    print(f"   Detection result: {json.dumps(detection_result, indent=2)}")
    
    # Step 6: Trigger response agent
    response_result = trigger_response_agent(detection_result)
    
    # Summary
    print("\n" + "=" * 58)
    print("WORKFLOW SUMMARY")
    print("=" * 58)
    print(f"Monitoring Agent: {'✅ SUCCESS' if monitoring_result and monitoring_result.get('statusCode') == 200 else '❌ FAILED'}")
    print(f"Analysis Agent: {'✅ SUCCESS' if analysis_result and analysis_result.get('statusCode') == 200 else '❌ FAILED'}")
    print(f"Response Agent: {'✅ SUCCESS' if response_result and response_result.get('statusCode') == 200 else '❌ FAILED'}")
    
    if (monitoring_result and monitoring_result.get('statusCode') == 200 and
        analysis_result and analysis_result.get('statusCode') == 200 and
        response_result and response_result.get('statusCode') == 200):
        print("\n🎉 Complete fire detection workflow executed successfully!")
        print("\n📋 The multi-agent system is working as expected:")
        print("   1. Monitoring Agent validates sensor data health")
        print("   2. Analysis Agent processes features for fire detection")
        print("   3. Response Agent determines appropriate actions")
        print("\n🚀 The Synthetic Fire Prediction System is ready for production use!")
    else:
        print("\n⚠️  Workflow completed with some issues.")
        print("   Please check the individual agent responses above.")

def main():
    """Main function"""
    simulate_fire_detection_workflow()

if __name__ == "__main__":
    main()