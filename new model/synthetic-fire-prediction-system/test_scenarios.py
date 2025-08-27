#!/usr/bin/env python3
"""
Test scenario generation system implementation.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_scenario_generation():
    """Test scenario generation system"""
    print("🎭 Testing Scenario Generation System...")
    
    try:
        from data_generation.thermal.thermal_image_generator import ThermalImageGenerator
        from data_generation.gas.gas_concentration_generator import GasConcentrationGenerator
        from data_generation.environmental.environmental_data_generator import EnvironmentalDataGenerator
        from data_generation.scenarios.scenario_generator import ScenarioGenerator
        from data_generation.scenarios.false_positive_generator import FalsePositiveGenerator
        
        # Configure generators
        thermal_config = {
            'resolution': (288, 384),
            'min_temperature': 20.0,
            'max_temperature': 500.0,
            'output_formats': ['numpy'],
            'hotspot_config': {},
            'temporal_config': {},
            'noise_config': {
                'noise_types': ['gaussian'],
                'noise_params': {
                    'gaussian': {'mean': 0, 'std': 2.0}
                }
            }
        }
        
        gas_config = {
            'gas_types': ['methane', 'propane', 'hydrogen', 'carbon_monoxide'],
            'diffusion_config': {},
            'temporal_config': {},
            'sensor_configs': {}
        }
        
        env_config = {
            'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
            'parameter_ranges': {
                'temperature': {'min': 15.0, 'max': 35.0, 'unit': '°C'},
                'humidity': {'min': 20.0, 'max': 80.0, 'unit': '%'},
                'pressure': {'min': 990.0, 'max': 1030.0, 'unit': 'hPa'},
                'voc': {'min': 0.0, 'max': 2000.0, 'unit': 'ppb'}
            },
            'voc_config': {},
            'correlation_config': {},
            'variation_config': {}
        }
        
        # Create generators
        thermal_gen = ThermalImageGenerator(thermal_config)
        gas_gen = GasConcentrationGenerator(gas_config)
        env_gen = EnvironmentalDataGenerator(env_config)
        
        print("✓ Component generators initialized")
        
        # Create scenario generator
        scenario_config = {}
        scenario_gen = ScenarioGenerator(thermal_gen, gas_gen, env_gen, scenario_config)
        print("✓ Scenario generator initialized")
        
        # Test scenario definition validation
        valid_scenario = {
            "scenario_type": "electrical_fire",
            "duration": 300,
            "sample_rate": 0.1,
            "room_params": {
                "room_volume": 100.0,
                "ventilation_rate": 2.0,
                "initial_temperature": 25.0,
                "initial_humidity": 50.0,
                "fuel_load": 10.0
            },
            "fire_params": {
                "fire_location": [192, 144],
                "fire_size": 1.0,
                "growth_rate": 0.1,
                "max_size": 50.0
            },
            "gas_params": {
                "gas_types": ["carbon_monoxide", "methane"],
                "release_rates": {
                    "carbon_monoxide": 0.01,
                    "methane": 0.005
                }
            }
        }
        
        scenario_gen.validate_scenario_definition(valid_scenario)
        print("✓ Scenario definition validation passed")
        
        # Generate a fire scenario
        fire_scenario_params = {
            'room_params': {
                'room_volume': 100.0,
                'ventilation_rate': 2.0,
                'initial_temperature': 25.0,
                'initial_humidity': 50.0,
                'fuel_load': 10.0
            },
            'fire_params': {
                'fire_location': [192, 144],
                'fire_size': 1.0,
                'growth_rate': 0.1,
                'max_size': 50.0
            }
        }
        
        fire_scenario = scenario_gen.generate_scenario(
            start_time=datetime.now(),
            duration_seconds=120,
            sample_rate_hz=0.1,
            scenario_type='electrical_fire',
            scenario_params=fire_scenario_params,
            seed=42
        )
        
        print(f"✓ Generated electrical fire scenario:")
        print(f"  • Duration: {fire_scenario['metadata']['duration']} seconds")
        print(f"  • Thermal frames: {len(fire_scenario['thermal_data']['frames'])}")
        print(f"  • Gas types: {len(fire_scenario['gas_data']['gas_data'])}")
        print(f"  • Environmental parameters: {len(fire_scenario['environmental_data']['environmental_data'])}")
        
        # Test false positive generator
        fp_gen = FalsePositiveGenerator(thermal_gen, gas_gen, env_gen, {})
        print("✓ False positive generator initialized")
        
        # Generate a cooking false positive scenario
        cooking_scenario = fp_gen.generate_false_positive_scenario(
            start_time=datetime.now(),
            duration_seconds=60,
            sample_rate_hz=0.1,
            false_positive_type='cooking',
            seed=42
        )
        
        print(f"✓ Generated cooking false positive scenario:")
        print(f"  • Duration: {cooking_scenario['metadata']['duration']} seconds")
        print(f"  • Classification: {cooking_scenario['metadata']['scenario_params']['metadata']['classification']}")
        print(f"  • Type: {cooking_scenario['metadata']['scenario_params']['metadata']['false_positive_type']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Scenario generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run scenario generation test"""
    print("🚀 Scenario Generation System Test")
    print("=" * 50)
    
    success = test_scenario_generation()
    
    if success:
        print("\n✅ Scenario generation system is working correctly!")
    else:
        print("\n❌ Scenario generation system test failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)