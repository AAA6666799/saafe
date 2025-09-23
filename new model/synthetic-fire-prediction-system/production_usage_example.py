#!/usr/bin/env python3
"""
Production Usage Example
This script demonstrates how to use the deployed synthetic fire prediction system in production.
"""

import boto3
import json
import time
from datetime import datetime

class FireDetectionSystem:
    """A class to interact with the deployed fire detection system."""
    
    def __init__(self):
        """Initialize the fire detection system client."""
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
        self.sns_client = boto3.client('sns', region_name='us-east-1')
        self.endpoint_name = 'fire-mvp-xgb-endpoint'
        self.alert_topic_arn = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
    
    def predict_fire_risk(self, sensor_data):
        """
        Predict fire risk based on sensor data.
        
        Args:
            sensor_data (dict): Dictionary containing sensor readings
            
        Returns:
            float: Fire risk probability (0-1)
        """
        # Convert dictionary to CSV format for XGBoost
        csv_data = f"{sensor_data['t_mean']},{sensor_data['t_std']},{sensor_data['t_max']},{sensor_data['t_p95']},{sensor_data['t_hot_area_pct']},{sensor_data['t_hot_largest_blob_pct']},{sensor_data['t_grad_mean']},{sensor_data['t_grad_std']},{sensor_data['t_diff_mean']},{sensor_data['t_diff_std']},{sensor_data['flow_mag_mean']},{sensor_data['flow_mag_std']},{sensor_data['tproxy_val']},{sensor_data['tproxy_delta']},{sensor_data['tproxy_vel']},{sensor_data['gas_val']},{sensor_data['gas_delta']},{sensor_data['gas_vel']}"
        
        try:
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='text/csv',
                Body=csv_data
            )
            
            result = float(response['Body'].read().decode())
            return result
            
        except Exception as e:
            print(f"Error predicting fire risk: {e}")
            return None
    
    def send_alert(self, risk_score, sensor_data, alert_level="INFO"):
        """
        Send an alert notification.
        
        Args:
            risk_score (float): Fire risk probability
            sensor_data (dict): Sensor readings that triggered the alert
            alert_level (str): Alert level (INFO, WARNING, ALERT, EMERGENCY)
        """
        alert_message = {
            "timestamp": datetime.now().isoformat(),
            "alert_level": alert_level,
            "risk_score": risk_score,
            "sensor_data": sensor_data,
            "message": f"Fire risk detected with probability {risk_score:.2f}"
        }
        
        try:
            self.sns_client.publish(
                TopicArn=self.alert_topic_arn,
                Message=json.dumps(alert_message, indent=2),
                Subject=f"Fire Detection Alert - {alert_level}"
            )
            print(f"Alert sent: {alert_level} - Risk score: {risk_score:.2f}")
            
        except Exception as e:
            print(f"Error sending alert: {e}")
    
    def process_sensor_data(self, sensor_data):
        """
        Process sensor data and take appropriate actions.
        
        Args:
            sensor_data (dict): Dictionary containing sensor readings
        """
        print(f"Processing sensor data: {sensor_data}")
        
        # Predict fire risk
        risk_score = self.predict_fire_risk(sensor_data)
        
        if risk_score is None:
            print("Failed to predict fire risk")
            return
        
        print(f"Fire risk probability: {risk_score:.2f}")
        
        # Take action based on risk score
        if risk_score >= 0.8:
            # Emergency - High risk
            self.send_alert(risk_score, sensor_data, "EMERGENCY")
        elif risk_score >= 0.6:
            # Alert - Medium-high risk
            self.send_alert(risk_score, sensor_data, "ALERT")
        elif risk_score >= 0.4:
            # Warning - Medium risk
            self.send_alert(risk_score, sensor_data, "WARNING")
        elif risk_score >= 0.2:
            # Informational - Low risk
            self.send_alert(risk_score, sensor_data, "INFO")

def example_usage():
    """Example usage of the fire detection system."""
    print("🔥 Synthetic Fire Prediction System - Production Usage Example")
    print("=" * 60)
    
    # Initialize the system
    fire_system = FireDetectionSystem()
    
    # Example 1: Normal conditions
    print("\n📝 Example 1: Normal conditions")
    normal_data = {
        "t_mean": 22.5,
        "t_std": 1.2,
        "t_max": 25.1,
        "t_p95": 24.8,
        "t_hot_area_pct": 0.5,
        "t_hot_largest_blob_pct": 0.3,
        "t_grad_mean": 0.1,
        "t_grad_std": 0.05,
        "t_diff_mean": 0.2,
        "t_diff_std": 0.1,
        "flow_mag_mean": 0.3,
        "flow_mag_std": 0.1,
        "tproxy_val": 23.0,
        "tproxy_delta": 0.5,
        "tproxy_vel": 0.1,
        "gas_val": 410.0,
        "gas_delta": 5.0,
        "gas_vel": 1.0
    }
    
    fire_system.process_sensor_data(normal_data)
    
    # Example 2: Potential fire conditions
    print("\n📝 Example 2: Potential fire conditions")
    fire_data = {
        "t_mean": 45.2,
        "t_std": 8.7,
        "t_max": 78.5,
        "t_p95": 72.1,
        "t_hot_area_pct": 25.3,
        "t_hot_largest_blob_pct": 18.7,
        "t_grad_mean": 3.2,
        "t_grad_std": 1.8,
        "t_diff_mean": 2.9,
        "t_diff_std": 1.5,
        "flow_mag_mean": 4.2,
        "flow_mag_std": 2.1,
        "tproxy_val": 52.0,
        "tproxy_delta": 15.0,
        "tproxy_vel": 3.2,
        "gas_val": 850.0,
        "gas_delta": 120.0,
        "gas_vel": 8.5
    }
    
    fire_system.process_sensor_data(fire_data)
    
    print("\n✅ Production usage example completed!")
    print("\n🚀 Next steps for production deployment:")
    print("  1. Deploy edge devices with sensors")
    print("  2. Configure data ingestion pipeline")
    print("  3. Monitor system through CloudWatch dashboard")
    print("  4. Set up alert notifications for stakeholders")

if __name__ == "__main__":
    example_usage()