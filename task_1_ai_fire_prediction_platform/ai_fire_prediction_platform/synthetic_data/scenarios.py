"""
Scenario generation for synthetic fire prediction system
"""

import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from ai_fire_prediction_platform.core.interfaces import SensorData
from ai_fire_prediction_platform.synthetic_data.thermal import ThermalDataGenerator
from ai_fire_prediction_platform.synthetic_data.gas import GasDataGenerator


class ScenarioGenerator:
    """Generate different fire scenarios for training and testing"""
    
    def __init__(self, synthetic_data_config: Dict[str, Any]):
        self.config = synthetic_data_config
        self.thermal_generator = ThermalDataGenerator(synthetic_data_config)
        self.gas_generator = GasDataGenerator(synthetic_data_config)
        
        # Environmental parameters
        self.temperature_range_env = synthetic_data_config.get('temperature_range_env', (10.0, 40.0))
        self.humidity_range = synthetic_data_config.get('humidity_range', (20.0, 80.0))
        self.pressure_range = synthetic_data_config.get('pressure_range', (980.0, 1020.0))
        self.voc_range = synthetic_data_config.get('voc_range', (0.0, 500.0))
    
    def generate_normal_scenario(self, duration_seconds: int = 3600) -> List[SensorData]:
        """Generate normal (non-fire) scenario data"""
        sensor_data_list = []
        num_samples = int(duration_seconds * 1)  # 1 sample per second
        
        # Base ambient conditions
        ambient_temp = np.random.uniform(*self.temperature_range_env)
        base_params = {
            'ambient_temperature': ambient_temp,
            'fire_present': False,
            'ambient_gas_concentrations': {
                'methane': np.random.uniform(0, 10),
                'propane': np.random.uniform(0, 5),
                'hydrogen': np.random.uniform(0, 2),
                'co': np.random.uniform(0, 5),
                'co2': np.random.uniform(400, 500)
            }
        }
        
        for i in range(num_samples):
            timestamp = i
            params = base_params.copy()
            
            # Generate sensor data
            thermal_frame = self.thermal_generator.generate(params, timestamp)
            gas_readings = self.gas_generator.generate(params, timestamp)
            
            # Environmental data with small variations
            environmental_data = {
                'temperature': ambient_temp + np.random.normal(0, 1),
                'humidity': np.random.uniform(*self.humidity_range),
                'pressure': np.random.uniform(*self.pressure_range),
                'voc': np.random.uniform(*self.voc_range)
            }
            
            sensor_data = SensorData(
                timestamp=float(timestamp),
                thermal_frame=thermal_frame,
                gas_readings=gas_readings,
                environmental_data=environmental_data
            )
            
            sensor_data_list.append(sensor_data)
        
        return sensor_data_list
    
    def generate_fire_scenario(self, duration_seconds: int = 600, fire_start_time: int = 120) -> List[SensorData]:
        """Generate fire scenario data"""
        sensor_data_list = []
        num_samples = int(duration_seconds * 1)  # 1 sample per second
        
        # Base ambient conditions
        ambient_temp = np.random.uniform(*self.temperature_range_env)
        base_params = {
            'ambient_temperature': ambient_temp,
            'fire_present': False,  # Will be set to True after fire_start_time
            'ambient_gas_concentrations': {
                'methane': np.random.uniform(0, 10),
                'propane': np.random.uniform(0, 5),
                'hydrogen': np.random.uniform(0, 2),
                'co': np.random.uniform(0, 5),
                'co2': np.random.uniform(400, 500)
            },
            'fire_gas_factors': {
                'methane': 50.0,
                'propane': 100.0,
                'hydrogen': 80.0,
                'co': 200.0,
                'co2': 300.0
            }
        }
        
        for i in range(num_samples):
            timestamp = i
            params = base_params.copy()
            
            # Set fire parameters after fire start time
            if i >= fire_start_time:
                params['fire_present'] = True
                params['time_since_fire_start'] = i - fire_start_time
                params['num_hotspots'] = np.random.randint(1, 4)
                params['hotspot_intensity'] = np.random.uniform(30, 80)
                # Increase ambient temperature due to fire
                params['ambient_temperature'] = ambient_temp + np.random.uniform(10, 30)
            
            # Generate sensor data
            thermal_frame = self.thermal_generator.generate(params, timestamp)
            gas_readings = self.gas_generator.generate(params, timestamp)
            
            # Environmental data - increase temperature and other parameters during fire
            env_temp = params['ambient_temperature']
            if i >= fire_start_time:
                humidity = np.random.uniform(self.humidity_range[0], self.humidity_range[1] * 0.7)  # Lower humidity
                voc = np.random.uniform(self.voc_range[0] * 2, self.voc_range[1] * 3)  # Higher VOC
            else:
                humidity = np.random.uniform(*self.humidity_range)
                voc = np.random.uniform(*self.voc_range)
            
            environmental_data = {
                'temperature': env_temp + np.random.normal(0, 1),
                'humidity': humidity,
                'pressure': np.random.uniform(*self.pressure_range),
                'voc': voc
            }
            
            sensor_data = SensorData(
                timestamp=float(timestamp),
                thermal_frame=thermal_frame,
                gas_readings=gas_readings,
                environmental_data=environmental_data
            )
            
            sensor_data_list.append(sensor_data)
        
        return sensor_data_list
    
    def generate_false_positive_scenario(self, duration_seconds: int = 600) -> List[SensorData]:
        """Generate false positive scenario (e.g., cooking, heating)"""
        sensor_data_list = []
        num_samples = int(duration_seconds * 1)  # 1 sample per second
        
        # Base ambient conditions
        ambient_temp = np.random.uniform(*self.temperature_range_env)
        base_params = {
            'ambient_temperature': ambient_temp,
            'fire_present': True,  # But with different characteristics
            'ambient_gas_concentrations': {
                'methane': np.random.uniform(0, 10),
                'propane': np.random.uniform(0, 5),
                'hydrogen': np.random.uniform(0, 2),
                'co': np.random.uniform(0, 5),
                'co2': np.random.uniform(400, 500)
            },
            'fire_gas_factors': {
                'methane': 10.0,  # Lower than actual fire
                'propane': 20.0,
                'hydrogen': 15.0,
                'co': 30.0,
                'co2': 50.0
            }
        }
        
        for i in range(num_samples):
            timestamp = i
            params = base_params.copy()
            
            # Different fire characteristics for false positive
            params['time_since_fire_start'] = i
            params['num_hotspots'] = np.random.randint(1, 2)  # Fewer hotspots
            params['hotspot_intensity'] = np.random.uniform(10, 30)  # Lower intensity
            # Moderate temperature increase
            params['ambient_temperature'] = ambient_temp + np.random.uniform(5, 15)
            
            # Generate sensor data
            thermal_frame = self.thermal_generator.generate(params, timestamp)
            gas_readings = self.gas_generator.generate(params, timestamp)
            
            # Environmental data with moderate changes
            env_temp = params['ambient_temperature']
            humidity = np.random.uniform(self.humidity_range[0], self.humidity_range[1] * 0.9)  # Slightly lower
            voc = np.random.uniform(self.voc_range[0] * 1.2, self.voc_range[1] * 1.5)  # Moderately higher
            
            environmental_data = {
                'temperature': env_temp + np.random.normal(0, 1),
                'humidity': humidity,
                'pressure': np.random.uniform(*self.pressure_range),
                'voc': voc
            }
            
            sensor_data = SensorData(
                timestamp=float(timestamp),
                thermal_frame=thermal_frame,
                gas_readings=gas_readings,
                environmental_data=environmental_data
            )
            
            sensor_data_list.append(sensor_data)
        
        return sensor_data_list