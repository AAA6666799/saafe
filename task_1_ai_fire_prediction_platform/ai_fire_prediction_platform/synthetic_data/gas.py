"""
Gas sensor data generation for synthetic fire prediction system
"""

import numpy as np
from typing import Dict, Any, List
from ai_fire_prediction_platform.core.interfaces import DataGenerator


class GasDataGenerator(DataGenerator):
    """Generate synthetic gas sensor data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gas_types = config.get('gas_types', ["methane", "propane", "hydrogen", "co", "co2"])
        self.concentration_range = config.get('concentration_range', (0.0, 1000.0))
        self.diffusion_rate = config.get('diffusion_rate', 0.1)
        self.sensor_drift_rate = config.get('sensor_drift_rate', 0.01)
        self.noise_level = config.get('gas_noise_level', 0.05)
    
    def generate(self, scenario_params: Dict[str, Any], timestamp: float) -> Dict[str, float]:
        """Generate synthetic gas readings based on scenario parameters"""
        gas_readings = {}
        
        # Base ambient concentrations
        ambient_concentrations = scenario_params.get('ambient_gas_concentrations', {})
        
        # Add fire-related gas concentrations if fire is present
        fire_present = scenario_params.get('fire_present', False)
        fire_gas_factors = scenario_params.get('fire_gas_factors', {})
        
        for gas_type in self.gas_types:
            # Base concentration
            base_concentration = ambient_concentrations.get(gas_type, 0.0)
            
            # Increase if fire is present
            if fire_present and gas_type in fire_gas_factors:
                fire_factor = fire_gas_factors[gas_type]
                # Time-based increase with diffusion
                time_since_fire_start = scenario_params.get('time_since_fire_start', 0.0)
                concentration_increase = fire_factor * (1 - np.exp(-self.diffusion_rate * time_since_fire_start))
                base_concentration += concentration_increase
            
            # Add sensor drift over time
            drift = timestamp * self.sensor_drift_rate
            base_concentration += drift
            
            # Add noise
            noise = np.random.normal(0, self.noise_level * base_concentration)
            base_concentration += noise
            
            # Ensure within valid range
            base_concentration = np.clip(base_concentration, self.concentration_range[0], self.concentration_range[1])
            
            gas_readings[gas_type] = base_concentration
        
        return gas_readings
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate scenario parameters for gas data generation"""
        # Check if required parameters are present
        if 'ambient_gas_concentrations' in params:
            concentrations = params['ambient_gas_concentrations']
            for gas_type, concentration in concentrations.items():
                if not isinstance(concentration, (int, float)):
                    return False
                if not (self.concentration_range[0] <= concentration <= self.concentration_range[1]):
                    return False
        
        # Validate fire gas factors if present
        if 'fire_gas_factors' in params:
            factors = params['fire_gas_factors']
            for gas_type, factor in factors.items():
                if not isinstance(factor, (int, float)) or factor < 0:
                    return False
                    
        return True