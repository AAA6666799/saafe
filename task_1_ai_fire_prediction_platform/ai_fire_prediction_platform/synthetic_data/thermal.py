"""
Thermal data generation for synthetic fire prediction system
"""

import numpy as np
from typing import Dict, Any
from ai_fire_prediction_platform.core.interfaces import DataGenerator, SensorData


class ThermalDataGenerator(DataGenerator):
    """Generate synthetic thermal image data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resolution = config.get('thermal_image_resolution', (384, 288))
        self.noise_level = config.get('thermal_noise_level', 0.1)
        self.temperature_range = config.get('temperature_range', (15.0, 100.0))
        self.hotspot_min_size = config.get('hotspot_min_size', 5)
        self.hotspot_max_size = config.get('hotspot_max_size', 50)
    
    def generate(self, scenario_params: Dict[str, Any], timestamp: float) -> np.ndarray:
        """Generate synthetic thermal frame based on scenario parameters"""
        # Base ambient temperature
        ambient_temp = scenario_params.get('ambient_temperature', 25.0)
        
        # Create base thermal frame with ambient temperature and noise
        thermal_frame = np.full(self.resolution, ambient_temp, dtype=np.float32)
        noise = np.random.normal(0, self.noise_level, self.resolution)
        thermal_frame += noise
        
        # Add hotspots if in fire scenario
        if scenario_params.get('fire_present', False):
            num_hotspots = scenario_params.get('num_hotspots', 1)
            hotspot_intensity = scenario_params.get('hotspot_intensity', 50.0)
            
            for _ in range(num_hotspots):
                # Random position for hotspot
                x_pos = np.random.randint(0, self.resolution[0])
                y_pos = np.random.randint(0, self.resolution[1])
                
                # Random size for hotspot
                size = np.random.randint(self.hotspot_min_size, self.hotspot_max_size)
                
                # Create hotspot with Gaussian distribution
                for i in range(max(0, x_pos-size), min(self.resolution[0], x_pos+size)):
                    for j in range(max(0, y_pos-size), min(self.resolution[1], y_pos+size)):
                        distance = np.sqrt((i - x_pos)**2 + (j - y_pos)**2)
                        if distance <= size:
                            # Gaussian intensity decay
                            intensity = hotspot_intensity * np.exp(-distance**2 / (2 * (size/3)**2))
                            thermal_frame[i, j] += intensity
        
        # Ensure values are within valid range
        thermal_frame = np.clip(thermal_frame, self.temperature_range[0], self.temperature_range[1])
        
        return thermal_frame
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate scenario parameters for thermal data generation"""
        required_params = ['ambient_temperature']
        for param in required_params:
            if param not in params:
                return False
        
        # Validate parameter ranges
        if not (self.temperature_range[0] <= params['ambient_temperature'] <= self.temperature_range[1]):
            return False
            
        if 'hotspot_intensity' in params:
            if not (0 <= params['hotspot_intensity'] <= (self.temperature_range[1] - self.temperature_range[0])):
                return False
                
        return True