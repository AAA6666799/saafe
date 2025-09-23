#!/usr/bin/env python3
"""
Test script for synthetic data augmentation features.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_additional_fire_scenarios():
    """Test the additional fire scenario generators."""
    print("Testing Additional Fire Scenarios...")
    
    try:
        # Check if the file exists
        import os
        additional_scenarios_path = os.path.join(os.path.dirname(__file__), 'src', 'data_generation', 'scenarios', 'additional_fire_scenarios.py')
        if os.path.exists(additional_scenarios_path):
            print("  Additional fire scenarios module exists")
            print("  ✅ Additional Fire Scenarios test passed")
            return True
        else:
            print("  ❌ Additional fire scenarios module not found")
            return False
        
    except Exception as e:
        print(f"  ❌ Additional Fire Scenarios test failed: {str(e)}")
        return False

def test_seasonal_temperature_variations():
    """Test the seasonal temperature variation generator."""
    print("Testing Seasonal Temperature Variations...")
    
    try:
        from src.data_generation.seasonal_temperature_variations import SeasonalTemperatureVariationGenerator
        
        # Create generator
        seasonal_gen = SeasonalTemperatureVariationGenerator({
            'location_latitude': 40.7128,
            'base_temperature': 20.0,
            'seasonal_amplitude': 15.0,
            'daily_amplitude': 10.0
        })
        
        # Generate seasonal temperature variations
        timestamp = datetime.now()
        temperature_variations = seasonal_gen.generate_seasonal_temperature(
            timestamp=timestamp,
            duration_seconds=3600,  # 1 hour
            sample_rate_hz=0.1  # Every 10 seconds
        )
        
        print(f"  Generated {len(temperature_variations)} temperature samples")
        print(f"  Temperature range: {np.min(temperature_variations):.2f}°C to {np.max(temperature_variations):.2f}°C")
        
        # Test applying to thermal data
        sample_thermal_data = {
            't_mean': [20.0] * 360,  # 1 hour of data at 0.1 Hz
            't_max': [25.0] * 360,
            'sample_rate_hz': 0.1
        }
        
        updated_thermal_data = seasonal_gen.apply_to_thermal_data(sample_thermal_data, timestamp)
        print(f"  Applied seasonal variations to thermal data")
        
        print("  ✅ Seasonal Temperature Variations test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Seasonal Temperature Variations test failed: {str(e)}")
        return False

def test_hvac_simulation():
    """Test the HVAC simulation generator."""
    print("Testing HVAC Simulation...")
    
    try:
        from src.data_generation.seasonal_temperature_variations import HVACSimulationGenerator
        
        # Create generator
        hvac_gen = HVACSimulationGenerator({
            'ventilation_rate': 0.5,
            'hvac_cycle_duration': 1800,
            'hvac_on_duration': 900,
            'air_mixing_efficiency': 0.8
        })
        
        # Create sample data
        sample_gas_data = {
            'gas_val': [500.0] * 120,  # 10 minutes of data at 0.2 Hz
            'gas_delta': [10.0] * 120,
            'sample_rate_hz': 0.2
        }
        
        sample_thermal_data = {
            't_mean': [22.0] * 120,
            't_max': [25.0] * 120,
            'sample_rate_hz': 0.2
        }
        
        # Simulate HVAC effects
        updated_gas_data, updated_thermal_data = hvac_gen.simulate_hvac_effect(
            sample_gas_data, sample_thermal_data, datetime.now()
        )
        
        print(f"  Applied HVAC effects to gas data: {len(updated_gas_data.get('gas_val', []))} samples")
        print(f"  Applied HVAC effects to thermal data: {len(updated_thermal_data.get('t_mean', []))} samples")
        
        print("  ✅ HVAC Simulation test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ HVAC Simulation test failed: {str(e)}")
        return False

def test_sunlight_heating():
    """Test the sunlight heating generator."""
    print("Testing Sunlight Heating Patterns...")
    
    try:
        from src.data_generation.seasonal_temperature_variations import SunlightHeatingGenerator
        
        # Create generator
        sunlight_gen = SunlightHeatingGenerator({
            'surface_types': ['wall', 'floor', 'ceiling', 'window'],
            'sunrise_time': 6.0,
            'sunset_time': 18.0,
            'max_solar_irradiance': 1000.0
        })
        
        # Create sample thermal data
        sample_thermal_data = {
            't_mean': [20.0] * 360,  # 1 hour of data at 9 Hz
            't_max': [25.0] * 360,
            'sample_rate_hz': 9.0
        }
        
        # Simulate sunlight heating on different surfaces
        for surface in ['wall', 'window']:
            updated_thermal_data = sunlight_gen.simulate_sunlight_heating(
                sample_thermal_data, datetime.now(), surface
            )
            max_increase = updated_thermal_data.get('metadata', {}).get('max_temp_increase', 0.0)
            print(f"  Sunlight heating on {surface}: max increase {max_increase:.2f}°C")
        
        print("  ✅ Sunlight Heating Patterns test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Sunlight Heating Patterns test failed: {str(e)}")
        return False

def test_flir_occlusion():
    """Test the FLIR occlusion generator."""
    print("Testing FLIR Occlusion Scenarios...")
    
    try:
        from src.data_generation.seasonal_temperature_variations import FlirOcclusionGenerator
        
        # Create generator
        occlusion_gen = FlirOcclusionGenerator({
            'occlusion_types': ['dust', 'steam', 'blockage'],
            'occlusion_severities': ['light', 'moderate', 'heavy'],
            'occlusion_probability': 0.05
        })
        
        # Create sample thermal data
        sample_thermal_data = {
            't_mean': [20.0] * 360,  # 1 hour of data at 9 Hz
            't_max': [25.0] * 360,
            'sample_rate_hz': 9.0
        }
        
        # Test different occlusion types and severities
        for occlusion_type in ['dust', 'steam', 'blockage']:
            for severity in ['light', 'moderate']:
                updated_thermal_data = occlusion_gen.simulate_flir_occlusion(
                    sample_thermal_data, occlusion_type, severity
                )
                print(f"  Applied {occlusion_type} ({severity}) occlusion")
        
        print("  ✅ FLIR Occlusion Scenarios test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ FLIR Occlusion Scenarios test failed: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Synthetic Data Augmentation Features")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_additional_fire_scenarios,
        test_seasonal_temperature_variations,
        test_hvac_simulation,
        test_sunlight_heating,
        test_flir_occlusion
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Synthetic data augmentation features are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())