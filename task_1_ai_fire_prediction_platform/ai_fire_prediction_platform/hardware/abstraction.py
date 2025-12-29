"""
Hardware abstraction for synthetic fire prediction system
"""

import boto3
import csv
from io import StringIO
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime
import time

from ai_fire_prediction_platform.core.interfaces import SensorData, HardwareInterface


class S3HardwareInterface(HardwareInterface):
    """Hardware interface that reads data from S3 bucket"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.bucket_name = self.config.get('s3_bucket', 'data-collector-of-first-device')
        self.thermal_prefix = self.config.get('thermal_prefix', 'thermal-data/')
        self.gas_prefix = self.config.get('gas_prefix', 'gas-data/')
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client('s3', region_name='us-east-1')
            self.is_connected_flag = True
        except Exception as e:
            print(f"Failed to initialize S3 client: {e}")
            self.s3_client = None
            self.is_connected_flag = False
        
        self.last_thermal_key = None
        self.last_gas_key = None
        self.sensor_health = {
            "thermal": 1.0,
            "gas_co": 1.0,
            "gas_no2": 1.0,
            "gas_voc": 1.0
        }
    
    def get_sensor_data(self) -> Optional[SensorData]:
        """Get sensor data from S3 bucket"""
        if not self.is_connected():
            return None
        
        try:
            # Get the most recent thermal and gas data files
            thermal_data = self._get_latest_thermal_data()
            gas_data = self._get_latest_gas_data()
            
            if not thermal_data and not gas_data:
                return None
            
            # Create timestamp
            timestamp = time.time()
            
            # Parse thermal data
            thermal_frame = None
            if thermal_data:
                thermal_frame = self._parse_thermal_data(thermal_data)
            
            # Parse gas data
            gas_readings = None
            if gas_data:
                gas_readings = self._parse_gas_data(gas_data)
            
            # Create environmental data (using gas data for now)
            environmental_data = None
            if gas_data:
                environmental_data = self._create_environmental_data(gas_data)
            
            return SensorData(
                timestamp=timestamp,
                thermal_frame=thermal_frame,
                gas_readings=gas_readings,
                environmental_data=environmental_data,
                sensor_health=self.sensor_health.copy()
            )
            
        except Exception as e:
            print(f"Error getting sensor data from S3: {e}")
            return None
    
    def _get_latest_thermal_data(self) -> Optional[Dict]:
        """Get the most recent thermal data from S3"""
        try:
            # List objects with thermal prefix, sorted by last modified
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.thermal_prefix
            )
            
            if 'Contents' not in response:
                return None
            
            # Sort by last modified (most recent first)
            objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
            
            # Get the most recent file that's different from the last one we processed
            for obj in objects:
                key = obj['Key']
                if self.last_thermal_key is None or key != self.last_thermal_key:
                    self.last_thermal_key = key
                    # Download and parse the file
                    file_response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                    content = file_response['Body'].read().decode('utf-8')
                    csv_reader = csv.DictReader(StringIO(content))
                    rows = list(csv_reader)
                    if rows:
                        return rows[0]  # Return the first (and typically only) row
            
            return None
        except Exception as e:
            print(f"Error getting thermal data from S3: {e}")
            return None
    
    def _get_latest_gas_data(self) -> Optional[Dict]:
        """Get the most recent gas data from S3"""
        try:
            # List objects with gas prefix, sorted by last modified
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.gas_prefix
            )
            
            if 'Contents' not in response:
                return None
            
            # Sort by last modified (most recent first)
            objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
            
            # Get the most recent file that's different from the last one we processed
            for obj in objects:
                key = obj['Key']
                if self.last_gas_key is None or key != self.last_gas_key:
                    self.last_gas_key = key
                    # Download and parse the file
                    file_response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                    content = file_response['Body'].read().decode('utf-8')
                    csv_reader = csv.DictReader(StringIO(content))
                    rows = list(csv_reader)
                    if rows:
                        return rows[0]  # Return the first (and typically only) row
            
            return None
        except Exception as e:
            print(f"Error getting gas data from S3: {e}")
            return None
    
    def _parse_thermal_data(self, thermal_data: Dict) -> Optional[np.ndarray]:
        """Parse thermal data into a numpy array"""
        try:
            # Extract pixel values (assuming 32x24 resolution = 768 pixels)
            pixels = []
            for i in range(768):
                pixel_key = f'pixel_{i}'
                if pixel_key in thermal_data:
                    pixels.append(float(thermal_data[pixel_key]))
                else:
                    # If pixel data is missing, use a default value
                    pixels.append(25.0)
            
            # Reshape to 32x24 (or 24x32 depending on sensor orientation)
            thermal_frame = np.array(pixels).reshape(24, 32)
            return thermal_frame
        except Exception as e:
            print(f"Error parsing thermal data: {e}")
            # Return a default thermal frame
            return np.full((24, 32), 25.0, dtype=np.float32)
    
    def _parse_gas_data(self, gas_data: Dict) -> Optional[Dict[str, float]]:
        """Parse gas data into a dictionary"""
        try:
            gas_readings = {}
            if 'CO' in gas_data:
                gas_readings['co'] = float(gas_data['CO'])
            if 'NO2' in gas_data:
                gas_readings['no2'] = float(gas_data['NO2'])
            if 'VOC' in gas_data:
                gas_readings['voc'] = float(gas_data['VOC'])
            
            return gas_readings
        except Exception as e:
            print(f"Error parsing gas data: {e}")
            return None
    
    def _create_environmental_data(self, gas_data: Dict) -> Optional[Dict[str, float]]:
        """Create environmental data from gas data"""
        try:
            # For now, we'll use the timestamp from gas data to create some environmental data
            # In a real implementation, this would come from separate environmental sensors
            environmental_data = {
                'temperature': 22.0 + np.random.normal(0, 1),  # Simulate room temperature
                'humidity': 45.0 + np.random.normal(0, 5),     # Simulate humidity
                'pressure': 1013.0 + np.random.normal(0, 5),   # Simulate atmospheric pressure
                'voc': float(gas_data.get('VOC', 0))           # Use VOC from gas data
            }
            return environmental_data
        except Exception as e:
            print(f"Error creating environmental data: {e}")
            return None
    
    def calibrate_sensors(self, calibration_params: Dict[str, Any]) -> None:
        """Calibrate sensors (not implemented for S3 interface)"""
        print("Calibration not implemented for S3 hardware interface")
    
    def validate_sensor_health(self) -> Dict[str, float]:
        """Check sensor health status"""
        # For S3 interface, we assume sensors are healthy if we can connect
        # In a real implementation, this would check actual sensor health metrics
        return self.sensor_health.copy()
    
    def is_connected(self) -> bool:
        """Check if S3 connection is working"""
        return self.is_connected_flag and self.s3_client is not None