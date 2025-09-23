#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Interactive Model Test
This script allows you to interactively test the deployed model with custom sensor data.
"""

import boto3
import json

# AWS Configuration
AWS_REGION = 'us-east-1'

class InteractiveFireDetector:
    def __init__(self, endpoint_name):
        """Initialize the interactive fire detector."""
        self.endpoint_name = endpoint_name
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
    
    def get_user_input(self):
        """Get sensor data input from the user."""
        print("\n" + "="*60)
        print("FLIR+SCD41 Fire Detection - Interactive Test")
        print("="*60)
        print("Enter sensor data for fire detection analysis")
        print("(Press Enter to use default values)")
        print("="*60)
        
        # Get temperature data
        print("\nTemperature Sensor Data (FLIR Lepton 3.5):")
        t_mean = input(f"Mean temperature (°C) [22.5]: ") or "22.5"
        t_std = input(f"Temperature standard deviation [1.2]: ") or "1.2"
        t_max = input(f"Maximum temperature (°C) [25.1]: ") or "25.1"
        t_p95 = input(f"95th percentile temperature (°C) [24.8]: ") or "24.8"
        t_hot_area_pct = input(f"Hot area percentage [%] [0.5]: ") or "0.5"
        t_hot_largest_blob_pct = input(f"Largest hot blob percentage [%] [0.3]: ") or "0.3"
        t_grad_mean = input(f"Mean temperature gradient [0.1]: ") or "0.1"
        t_grad_std = input(f"Temperature gradient std dev [0.05]: ") or "0.05"
        t_diff_mean = input(f"Mean temperature difference [0.2]: ") or "0.2"
        t_diff_std = input(f"Temperature difference std dev [0.1]: ") or "0.1"
        flow_mag_mean = input(f"Mean flow magnitude [0.3]: ") or "0.3"
        flow_mag_std = input(f"Flow magnitude std dev [0.1]: ") or "0.1"
        tproxy_val = input(f"Temperature proxy value (°C) [23.0]: ") or "23.0"
        tproxy_delta = input(f"Temperature proxy delta [0.5]: ") or "0.5"
        tproxy_vel = input(f"Temperature proxy velocity [0.1]: ") or "0.1"
        
        # Get gas data
        print("\nGas Sensor Data (Sensirion SCD41):")
        gas_val = input(f"CO2 concentration (ppm) [410.0]: ") or "410.0"
        gas_delta = input(f"CO2 delta [5.0]: ") or "5.0"
        gas_vel = input(f"Gas velocity [1.0]: ") or "1.0"
        
        # Convert to floats
        try:
            temperature_data = {
                "t_mean": float(t_mean),
                "t_std": float(t_std),
                "t_max": float(t_max),
                "t_p95": float(t_p95),
                "t_hot_area_pct": float(t_hot_area_pct),
                "t_hot_largest_blob_pct": float(t_hot_largest_blob_pct),
                "t_grad_mean": float(t_grad_mean),
                "t_grad_std": float(t_grad_std),
                "t_diff_mean": float(t_diff_mean),
                "t_diff_std": float(t_diff_std),
                "flow_mag_mean": float(flow_mag_mean),
                "flow_mag_std": float(flow_mag_std),
                "tproxy_val": float(tproxy_val),
                "tproxy_delta": float(tproxy_delta),
                "tproxy_vel": float(tproxy_vel)
            }
            
            gas_data = {
                "gas_val": float(gas_val),
                "gas_delta": float(gas_delta),
                "gas_vel": float(gas_vel)
            }
            
            return temperature_data, gas_data
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")
            return None, None
    
    def predict_fire_probability(self, temperature_data, gas_data):
        """Make a fire detection prediction."""
        # Create feature vector
        features = [
            temperature_data["t_mean"],
            temperature_data["t_std"],
            temperature_data["t_max"],
            temperature_data["t_p95"],
            temperature_data["t_hot_area_pct"],
            temperature_data["t_hot_largest_blob_pct"],
            temperature_data["t_grad_mean"],
            temperature_data["t_grad_std"],
            temperature_data["t_diff_mean"],
            temperature_data["t_diff_std"],
            temperature_data["flow_mag_mean"],
            temperature_data["flow_mag_std"],
            temperature_data["tproxy_val"],
            temperature_data["tproxy_delta"],
            temperature_data["tproxy_vel"],
            gas_data["gas_val"],
            gas_data["gas_delta"],
            gas_data["gas_vel"]
        ]
        
        # Convert to CSV string
        csv_data = ",".join([str(f) for f in features])
        
        try:
            # Make prediction
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='text/csv',
                Body=csv_data
            )
            
            # Process response
            result = response['Body'].read().decode()
            
            # Try to parse the result
            try:
                # If it's a JSON response
                parsed_result = json.loads(result)
                probability = parsed_result if isinstance(parsed_result, (int, float)) else parsed_result.get('predictions', [0])[0]
            except:
                # If it's a plain float/string response
                try:
                    probability = float(result.strip())
                except:
                    probability = 0.0
            
            return probability
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def interpret_result(self, probability):
        """Interpret the fire detection result."""
        print(f"\n{'='*40}")
        print("FIRE DETECTION RESULT")
        print(f"{'='*40}")
        print(f"Fire Probability: {probability:.4f} ({probability*100:.2f}%)")
        
        # Interpret result
        if probability > 0.7:
            print("🔥 HIGH RISK: Strong indication of fire detected!")
            print("   Immediate action recommended!")
        elif probability > 0.5:
            print("⚠️  MEDIUM RISK: Possible fire detected, requires attention")
            print("   Monitor closely and prepare response measures")
        elif probability > 0.3:
            print("🟡 LOW RISK: Unusual conditions, monitor closely")
            print("   Continue monitoring, no immediate action required")
        else:
            print("✅ NORMAL: No fire detected")
            print("   Conditions appear normal")
        
        print(f"{'='*40}")
    
    def run_interactive_session(self):
        """Run an interactive session for fire detection testing."""
        print("FLIR+SCD41 Fire Detection System - Interactive Testing")
        print("=" * 55)
        print(f"Endpoint: {self.endpoint_name}")
        print("=" * 55)
        
        while True:
            # Get user input
            temperature_data, gas_data = self.get_user_input()
            
            if temperature_data is None or gas_data is None:
                continue
            
            # Make prediction
            print("\n🔍 Analyzing sensor data...")
            probability = self.predict_fire_probability(temperature_data, gas_data)
            
            if probability is not None:
                # Interpret result
                self.interpret_result(probability)
            else:
                print("❌ Failed to get prediction result")
            
            # Ask if user wants to continue
            print("\n" + "-"*40)
            continue_test = input("Test another scenario? (y/n) [n]: ").lower().strip()
            if continue_test not in ['y', 'yes']:
                break
        
        print("\n👋 Thank you for using the FLIR+SCD41 Fire Detection System!")

def main():
    """Main function to run interactive model testing."""
    # Endpoint name from our deployment
    endpoint_name = "flir-scd41-xgboost-model-corrected-20250829-095914-endpoint"
    
    # Initialize interactive detector
    detector = InteractiveFireDetector(endpoint_name)
    
    # Run interactive session
    detector.run_interactive_session()

if __name__ == "__main__":
    main()