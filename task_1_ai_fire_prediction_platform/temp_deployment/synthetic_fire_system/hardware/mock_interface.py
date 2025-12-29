"""
Mock hardware interface for synthetic fire prediction system
"""

import numpy as np
import time
from typing import Dict, Any, Optional
from synthetic_fire_system.core.interfaces import SensorData, HardwareInterface


class MockHardwareInterface(HardwareInterface):
    """Mock hardware interface for testing and development"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_connected_flag = True
        self.sensor_health = {
            "thermal": 1.0,
            "gas_methane": 1.0,
            "gas_co": 1.0,
            "environmental": 1.0
        }
        self.start_time = time.time()
        self.call_count = 0
    
    def get_sensor_data(self) -> Optional[SensorData]:
        """Get mock sensor readings"""
        if not self.is_connected():
            return None
        
        self.call_count += 1
        current_time = time.time()
        timestamp = current_time - self.start_time
        
        # Generate mock thermal data (384x288 array)
        thermal_frame = self._generate_mock_thermal_data(timestamp)
        
        # Generate mock gas readings
        gas_readings = self._generate_mock_gas_data(timestamp)
        
        # Generate mock environmental data
        environmental_data = self._generate_mock_environmental_data(timestamp)
        
        return SensorData(
            timestamp=timestamp,
            thermal_frame=thermal_frame,
            gas_readings=gas_readings,
            environmental_data=environmental_data,
            sensor_health=self.sensor_health.copy()
        )
    
    def _generate_mock_thermal_data(self, timestamp: float) -> np.ndarray:
        """Generate mock thermal image data"""
        # Base ambient temperature with some variation
        ambient_temp = 25.0 + 2.0 * np.sin(timestamp * 0.1)
        
        # Create thermal frame with ambient temperature
        resolution = self.config.get('thermal_image_resolution', (384, 288))
        thermal_frame = np.full(resolution, ambient_temp, dtype=np.float32)
        
        # Add some noise
        noise = np.random.normal(0, 0.5, resolution)
        thermal_frame += noise
        
        # Occasionally add a hotspot to simulate fire
        if self.call_count % 100 == 0:  # Every 100th call
            hotspot_intensity = np.random.uniform(20, 50)
            center_x = np.random.randint(50, resolution[0] - 50)
            center_y = np.random.randint(50, resolution[1] - 50)
            size = np.random.randint(10, 30)
            
            for i in range(max(0, center_x - size), min(resolution[0], center_x + size)):
                for j in range(max(0, center_y - size), min(resolution[1], center_y + size)):
                    distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if distance <= size:
                        intensity = hotspot_intensity * np.exp(-distance**2 / (2 * (size/3)**2))
                        thermal_frame[i, j] += intensity
        
        return thermal_frame
    
    def _generate_mock_gas_data(self, timestamp: float) -> Dict[str, float]:
        """Generate mock gas sensor readings"""
        # Base concentrations with some variation
        base_concentrations = {
            "methane": 5.0 + 2.0 * np.sin(timestamp * 0.05),
            "propane": 2.0 + 1.0 * np.cos(timestamp * 0.07),
            "hydrogen": 1.0 + 0.5 * np.sin(timestamp * 0.03),
            "co": 3.0 + 1.5 * np.cos(timestamp * 0.04),
            "co2": 450.0 + 20.0 * np.sin(timestamp * 0.02)
        }
        
        # Add noise to each reading
        gas_readings = {}
        for gas_type, concentration in base_concentrations.items():
            noise = np.random.normal(0, concentration * 0.1)  # 10% noise
            gas_readings[gas_type] = max(0, concentration + noise)  # Ensure non-negative
        
        # Occasionally increase gas levels to simulate fire
        if self.call_count % 150 == 0:  # Every 150th call
            multiplier = np.random.uniform(1.5, 3.0)
            for gas_type in ["co", "co2"]:
                gas_readings[gas_type] *= multiplier
        
        return gas_readings
    
    def _generate_mock_environmental_data(self, timestamp: float) -> Dict[str, float]:
        """Generate mock environmental sensor readings"""
        # Base environmental conditions with variation
        environmental_data = {
            "temperature": 22.0 + 3.0 * np.sin(timestamp * 0.08),
            "humidity": 45.0 + 10.0 * np.cos(timestamp * 0.06),
            "pressure": 1013.0 + 5.0 * np.sin(timestamp * 0.04),
            "voc": 100.0 + 50.0 * np.cos(timestamp * 0.09)
        }
        
        # Add noise to each reading
        for param, value in environmental_data.items():
            noise = np.random.normal(0, value * 0.05)  # 5% noise
            environmental_data[param] = value + noise
        
        # Occasionally change conditions to simulate fire
        if self.call_count % 120 == 0:  # Every 120th call
            environmental_data["temperature"] += np.random.uniform(5, 15)
            environmental_data["humidity"] *= np.random.uniform(0.7, 0.9)
            environmental_data["voc"] *= np.random.uniform(1.5, 2.5)
        
        return environmental_data
    
    def calibrate_sensors(self, calibration_params: Dict[str, Any]) -> None:
        """Mock sensor calibration"""
        print("Calibrating sensors with parameters:", calibration_params)
        # In a real implementation, this would send calibration commands to hardware
        # For mock, we just simulate the process
        time.sleep(0.1)  # Simulate calibration time
        print("Sensor calibration completed")
    
    def validate_sensor_health(self) -> Dict[str, float]:
        """Check sensor health status"""
        # In a real implementation, this would check actual sensor health
        # For mock, we randomly adjust health scores
        for sensor in self.sensor_health:
            # Small random drift in health scores
            drift = np.random.normal(0, 0.01)
            self.sensor_health[sensor] = np.clip(self.sensor_health[sensor] + drift, 0.0, 1.0)
        
        return self.sensor_health.copy()
    
    def is_connected(self) -> bool:
        """Check if hardware is connected"""
        # In a real implementation, this would check actual connection status
        # For mock, we just return the flag
        return self.is_connected_flag
    
    def disconnect(self):
        """Disconnect from hardware"""
        self.is_connected_flag = False