#!/usr/bin/env python3
"""
High-Frequency Data Processing Demo
This script demonstrates how to process high-frequency sensor data from S3 for real-time fire prediction.
"""

import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import io
import time

class HighFrequencyDataProcessor:
    """Process high-frequency sensor data for fire prediction."""
    
    def __init__(self):
        """Initialize the data processor with AWS clients."""
        self.s3_client = boto3.client('s3')
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
        self.sns_client = boto3.client('sns', region_name='us-east-1')
        self.bucket_name = 'data-collector-of-first-device'
        self.endpoint_name = 'fire-mvp-xgb-endpoint'
        self.alert_topic_arn = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
    
    def list_recent_files(self, minutes_back=5):
        """
        List files uploaded in the specified time period.
        
        Args:
            minutes_back (int): Number of minutes to look back
            
        Returns:
            list: List of recent file keys
        """
        try:
            # Calculate time threshold
            time_threshold = datetime.now() - timedelta(minutes=minutes_back)
            
            # List objects in the bucket
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            
            # Filter recent files
            recent_files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['LastModified'] > time_threshold:
                        recent_files.append(obj['Key'])
            
            print(f"Found {len(recent_files)} recent files")
            return recent_files
            
        except Exception as e:
            print(f"Error listing recent files: {e}")
            return []
    
    def download_file(self, file_key):
        """
        Download a file from S3.
        
        Args:
            file_key (str): S3 object key
            
        Returns:
            str: File content as string
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            content = response['Body'].read().decode('utf-8')
            return content
        except Exception as e:
            print(f"Error downloading {file_key}: {e}")
            return None
    
    def process_thermal_data(self, df):
        """
        Extract thermal features from raw data.
        
        Args:
            df (pandas.DataFrame): Raw thermal data
            
        Returns:
            list: Extracted features
        """
        try:
            # For MLX90640, we expect 32x24 = 768 pixels
            # Extract pixel data (assuming first 768 columns are pixel values)
            pixel_columns = min(768, len(df.columns))
            pixel_data = df.iloc[:, :pixel_columns].values.flatten()
            
            # Handle case where we have fewer pixels than expected
            if len(pixel_data) < 768:
                # Pad with mean value
                padded_data = np.full(768, np.mean(pixel_data))
                padded_data[:len(pixel_data)] = pixel_data
                pixel_data = padded_data
            
            # Calculate thermal features
            features = []
            
            # Basic temperature statistics
            features.append(float(np.mean(pixel_data)))  # t_mean
            features.append(float(np.std(pixel_data)))   # t_std
            features.append(float(np.max(pixel_data)))   # t_max
            features.append(float(np.percentile(pixel_data, 95)))  # t_p95
            
            # Hot area features (using 40°C as hot threshold)
            hot_mask = pixel_data > 40.0
            total_pixels = len(pixel_data)
            hot_pixels = np.sum(hot_mask)
            features.append(float(hot_pixels / total_pixels * 100.0))  # t_hot_area_pct
            
            # Largest hot blob percentage (simplified)
            features.append(float(hot_pixels / total_pixels * 50.0))  # t_hot_largest_blob_pct
            
            # Gradient features
            grad_x = np.gradient(pixel_data)
            gradient_magnitude = np.abs(grad_x)
            features.append(float(np.mean(gradient_magnitude)))  # t_grad_mean
            features.append(float(np.std(gradient_magnitude)))   # t_grad_std
            
            # Temporal difference features (using previous values or random for demo)
            features.append(float(np.random.normal(0.1, 0.05)))  # t_diff_mean
            features.append(float(np.random.normal(0.05, 0.02))) # t_diff_std
            
            # Optical flow features (simplified)
            features.append(float(np.random.normal(0.2, 0.1)))   # flow_mag_mean
            features.append(float(np.random.normal(0.1, 0.05)))  # flow_mag_std
            
            # Temperature proxy features
            features.append(features[2])  # tproxy_val = t_max
            features.append(float(np.random.normal(1.0, 0.5)))   # tproxy_delta
            features.append(float(np.random.normal(0.5, 0.2)))   # tproxy_vel
            
            return features
            
        except Exception as e:
            print(f"Error processing thermal data: {e}")
            # Return default values
            return [25.0, 1.0, 30.0, 28.0, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 30.0, 1.0, 0.5]
    
    def process_gas_data(self, df):
        """
        Extract gas features from raw data.
        
        Args:
            df (pandas.DataFrame): Raw gas data
            
        Returns:
            list: Extracted gas features
        """
        try:
            # Extract gas readings (assuming columns: CO, NO2, VOC)
            co_value = df['CO'].iloc[0] if 'CO' in df.columns else 400.0
            no2_value = df['NO2'].iloc[0] if 'NO2' in df.columns else 0.0
            voc_value = df['VOC'].iloc[0] if 'VOC' in df.columns else 0.0
            
            # Calculate gas features
            gas_features = []
            gas_features.append(float(co_value))  # gas_val
            gas_features.append(float(np.random.normal(5.0, 2.0)))  # gas_delta
            gas_features.append(gas_features[1])  # gas_vel = gas_delta
            
            return gas_features
            
        except Exception as e:
            print(f"Error processing gas data: {e}")
            # Return default values
            return [400.0, 5.0, 5.0]
    
    def combine_features(self, thermal_features, gas_features):
        """
        Combine thermal and gas features into a single feature vector.
        
        Args:
            thermal_features (list): Thermal features
            gas_features (list): Gas features
            
        Returns:
            list: Combined features (18 total)
        """
        # Combine features in the correct order
        # 15 thermal features + 3 gas features = 18 total
        combined = thermal_features + gas_features
        return combined
    
    def predict_fire_risk(self, features):
        """
        Send features to SageMaker endpoint for fire risk prediction.
        
        Args:
            features (list): Feature vector
            
        Returns:
            float: Fire risk probability (0-1)
        """
        try:
            # Convert features to CSV format
            csv_features = ','.join([str(f) for f in features])
            
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='text/csv',
                Body=csv_features
            )
            
            result = float(response['Body'].read().decode())
            return result
            
        except Exception as e:
            print(f"Error predicting fire risk: {e}")
            return 0.0
    
    def send_alert(self, risk_score, features, file_key):
        """
        Send alert via SNS if fire risk is detected.
        
        Args:
            risk_score (float): Fire risk probability
            features (list): Feature vector
            file_key (str): Source file name
        """
        # Determine alert level
        if risk_score >= 0.8:
            alert_level = "EMERGENCY"
        elif risk_score >= 0.6:
            alert_level = "ALERT"
        elif risk_score >= 0.4:
            alert_level = "WARNING"
        else:
            alert_level = "INFO"
        
        alert_message = {
            "timestamp": datetime.now().isoformat(),
            "alert_level": alert_level,
            "risk_score": risk_score,
            "source_file": file_key,
            "message": f"Fire risk detected with probability {risk_score:.2f}"
        }
        
        try:
            self.sns_client.publish(
                TopicArn=self.alert_topic_arn,
                Message=json.dumps(alert_message, indent=2),
                Subject=f"Fire Detection Alert - {alert_level} (Risk: {risk_score:.2f})"
            )
            print(f"🔔 Alert sent: {alert_level} - Risk score: {risk_score:.2f}")
            
        except Exception as e:
            print(f"Error sending alert: {e}")
    
    def process_file(self, file_key):
        """
        Process a single file from S3.
        
        Args:
            file_key (str): S3 object key
            
        Returns:
            dict: Processing result
        """
        print(f"Processing file: {file_key}")
        
        # Download file
        content = self.download_file(file_key)
        if not content:
            return None
        
        # Parse CSV content
        try:
            df = pd.read_csv(io.StringIO(content))
        except Exception as e:
            print(f"Error parsing CSV for {file_key}: {e}")
            return None
        
        # Process based on file type
        if 'thermal_data' in file_key:
            features = self.process_thermal_data(df)
            # For thermal-only data, we need to add default gas features
            gas_features = [400.0, 5.0, 5.0]  # Default gas features
            combined_features = self.combine_features(features, gas_features)
        elif 'gas_data' in file_key:
            gas_features = self.process_gas_data(df)
            # For gas-only data, we need to add default thermal features
            thermal_features = [25.0, 1.0, 30.0, 28.0, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 30.0, 1.0, 0.5]
            combined_features = self.combine_features(thermal_features, gas_features)
        else:
            print(f"Unknown file type: {file_key}")
            return None
        
        # Make prediction
        risk_score = self.predict_fire_risk(combined_features)
        
        # Send alert if needed
        if risk_score > 0.4:  # Send alerts for WARNING level and above
            self.send_alert(risk_score, combined_features, file_key)
        
        return {
            "file": file_key,
            "risk_score": risk_score,
            "features": combined_features,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_batch(self, minutes_back=5):
        """
        Process a batch of recent files.
        
        Args:
            minutes_back (int): Number of minutes to look back
            
        Returns:
            list: List of processing results
        """
        print(f"🚀 Starting batch processing for last {minutes_back} minutes")
        
        # Get recent files
        files = self.list_recent_files(minutes_back)
        
        if not files:
            print("No recent files found")
            return []
        
        # Process each file
        results = []
        for file_key in files:
            result = self.process_file(file_key)
            if result:
                results.append(result)
                print(f"✅ Processed {file_key} - Risk: {result['risk_score']:.2f}")
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.1)
        
        # Summary
        high_risk = len([r for r in results if r['risk_score'] > 0.6])
        print(f"📊 Batch processing complete: {len(results)} files processed, {high_risk} high-risk detections")
        
        return results

def demo_high_frequency_processing():
    """Demonstrate high-frequency data processing."""
    print("🔥 High-Frequency Data Processing Demo")
    print("=" * 50)
    
    # Initialize processor
    processor = HighFrequencyDataProcessor()
    
    # Process a batch of recent files
    print("\n🔄 Processing recent sensor data...")
    results = processor.process_batch(minutes_back=10)
    
    if results:
        # Show some results
        print(f"\n📈 Processing Results:")
        for i, result in enumerate(results[:5]):  # Show first 5 results
            print(f"  {i+1}. {result['file']}")
            print(f"     Risk Score: {result['risk_score']:.2f}")
            print(f"     Processed: {result['timestamp']}")
        
        if len(results) > 5:
            print(f"     ... and {len(results) - 5} more files")
        
        # Risk distribution
        risks = [r['risk_score'] for r in results]
        print(f"\n📊 Risk Statistics:")
        print(f"  Average Risk: {np.mean(risks):.2f}")
        print(f"  Max Risk: {np.max(risks):.2f}")
        print(f"  Min Risk: {np.min(risks):.2f}")
        
        # Alert summary
        high_risk = len([r for r in results if r['risk_score'] > 0.6])
        medium_risk = len([r for r in results if 0.4 <= r['risk_score'] <= 0.6])
        low_risk = len([r for r in results if r['risk_score'] < 0.4])
        
        print(f"\n🔔 Alert Summary:")
        print(f"  High Risk (>0.6): {high_risk}")
        print(f"  Medium Risk (0.4-0.6): {medium_risk}")
        print(f"  Low Risk (<0.4): {low_risk}")
    else:
        print("No data to process")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    demo_high_frequency_processing()