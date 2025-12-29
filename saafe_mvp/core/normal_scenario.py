"""
Normal scenario generator for stable baseline readings.
"""

import numpy as np
from .scenario_generator import BaseScenarioGenerator
from .data_models import ScenarioConfig


class NormalScenarioGenerator(BaseScenarioGenerator):
    """Generator for normal environmental conditions with stable readings."""
    
    def __init__(self, config: ScenarioConfig):
        super().__init__(config)
        
        # Normal baseline values (center of normal ranges)
        self.base_temperature = 22.0  # °C
        self.base_pm25 = 12.0         # μg/m³
        self.base_co2 = 450.0         # ppm
        self.base_audio = 35.0        # dB
        
        # Small variation ranges for normal conditions
        self.temp_variation = 2.0     # ±2°C
        self.pm25_variation = 5.0     # ±5 μg/m³
        self.co2_variation = 50.0     # ±50 ppm
        self.audio_variation = 8.0    # ±8 dB
    
    def generate_baseline_values(self, sample_index: int) -> tuple:
        """Generate stable baseline values with minimal natural variation."""
        # Time-based variations (simulate daily patterns)
        time_factor = sample_index / self.config.total_samples
        
        # Very gentle sine wave variations to simulate natural fluctuations
        temp_cycle = 0.5 * np.sin(2 * np.pi * time_factor * 0.1)  # Slow temperature cycle
        co2_cycle = 0.3 * np.sin(2 * np.pi * time_factor * 0.15)  # Slight CO2 variation
        
        # Generate values with small random variations
        temperature = (
            self.base_temperature + 
            temp_cycle + 
            np.random.uniform(-self.temp_variation * 0.3, self.temp_variation * 0.3)
        )
        
        pm25 = (
            self.base_pm25 + 
            np.random.uniform(-self.pm25_variation * 0.2, self.pm25_variation * 0.2)
        )
        
        co2 = (
            self.base_co2 + 
            co2_cycle * 20 +  # Small CO2 fluctuation
            np.random.uniform(-self.co2_variation * 0.3, self.co2_variation * 0.3)
        )
        
        audio_level = (
            self.base_audio + 
            np.random.uniform(-self.audio_variation * 0.4, self.audio_variation * 0.4)
        )
        
        return (temperature, pm25, co2, audio_level)