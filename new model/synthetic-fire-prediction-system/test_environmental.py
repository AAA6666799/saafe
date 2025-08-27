#!/usr/bin/env python3
"""
Test environmental data generator implementation.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_environmental_generation():
    """Test environmental data generation"""
    print("🌍 Testing Environmental Data Generation...")
    
    try:
        from data_generation.environmental.environmental_data_generator import EnvironmentalDataGenerator
        
        # Configure environmental generator
        config = {
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
        
        generator = EnvironmentalDataGenerator(config)
        print("✓ Environmental generator initialized")
        
        # Generate environmental data
        env_data = generator.generate(
            timestamp=datetime.now(),
            duration_seconds=300,  # 5 minutes
            sample_rate_hz=0.1,    # Every 10 seconds
            seed=42
        )
        
        print(f"\n📊 Generated environmental data:")
        print(f"  • Duration: {env_data['metadata']['duration']} seconds")
        print(f"  • Sample rate: {env_data['metadata']['sample_rate']} Hz")
        print(f"  • Total samples: {env_data['metadata']['num_samples']}")
        
        for param, data in env_data['environmental_data'].items():
            values = np.array(data['values'])
            unit = data['unit']
            print(f"  • {param.title()}: {values.min():.2f} - {values.max():.2f} {unit}")
        
        # Test single reading generation
        reading = generator.generate_environmental_reading(
            timestamp=datetime.now(),
            parameter='temperature',
            baseline=25.0,
            daily_variation=0.1,
            noise_level=0.05
        )
        print(f"  • Single temperature reading: {reading:.2f}°C")
        
        return True
        
    except Exception as e:
        print(f"✗ Environmental generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run environmental data generator test"""
    print("🚀 Environmental Data Generator Test")
    print("=" * 50)
    
    success = test_environmental_generation()
    
    if success:
        print("\n✅ Environmental data generator is working correctly!")
    else:
        print("\n❌ Environmental data generator test failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)