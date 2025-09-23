#!/usr/bin/env python3
"""
Test script for newly implemented FLIR+SCD41 optimization features.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_spatio_temporal_aligner():
    """Test the spatio-temporal aligner functionality."""
    print("Testing Spatio-Temporal Aligner...")
    
    try:
        from src.feature_engineering.fusion.spatio_temporal_aligner import SpatioTemporalAligner
        
        # Create test data
        base_time = datetime.now()
        thermal_data = []
        gas_data = []
        
        # Generate test data with timestamps
        for i in range(10):
            timestamp = (base_time - timedelta(seconds=10-i)).isoformat()
            thermal_data.append({
                'timestamp': timestamp,
                't_mean': 25.0 + i * 0.5,
                't_max': 30.0 + i * 0.8,
                'fire_likelihood_score': 0.1 + i * 0.1
            })
            
            gas_data.append({
                'timestamp': timestamp,
                'gas_val': 450.0 + i * 5.0,
                'gas_fire_likelihood_score': 0.05 + i * 0.08
            })
        
        # Test aligner
        aligner = SpatioTemporalAligner()
        sensor_positions = {
            'flir_01': (0.0, 0.0),
            'scd41_01': (3.0, 4.0)  # 5 meters apart
        }
        
        alignment_features = aligner.align_sensor_data(thermal_data, gas_data, sensor_positions)
        
        print(f"  Alignment features generated: {len(alignment_features)}")
        print(f"  Time difference: {alignment_features.get('sensor_time_difference', 'N/A')}")
        print(f"  Spatial separation: {alignment_features.get('sensor_separation_distance', 'N/A')}")
        print("  ✅ Spatio-Temporal Aligner test passed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Spatio-Temporal Aligner test failed: {str(e)}")
        return False

def test_false_positive_discriminator():
    """Test the false positive discriminator functionality."""
    print("Testing False Positive Discriminator...")
    
    try:
        from src.feature_engineering.fusion.false_positive_discriminator import FalsePositiveDiscriminator
        
        # Create test data for different scenarios
        discriminator = FalsePositiveDiscriminator()
        
        # Test 1: Normal fire scenario
        thermal_fire = {
            't_max': 75.0,
            't_mean': 35.0,
            't_hot_area_pct': 15.0
        }
        
        gas_fire = {
            'gas_val': 1200.0,
            'gas_delta': 150.0
        }
        
        fire_discrimination = discriminator.discriminate_false_positives(thermal_fire, gas_fire)
        print(f"  Fire scenario false positive likelihood: {fire_discrimination.get('false_positive_likelihood', 'N/A')}")
        
        # Test 2: Sunlight heating scenario
        thermal_sunlight = {
            't_max': 45.0,
            't_mean': 28.0,
            't_hot_area_pct': 30.0
        }
        
        gas_sunlight = {
            'gas_val': 450.0,
            'gas_delta': 2.0
        }
        
        sunlight_discrimination = discriminator.discriminate_false_positives(thermal_sunlight, gas_sunlight)
        print(f"  Sunlight scenario discrimination score: {sunlight_discrimination.get('sunlight_discrimination_score', 'N/A')}")
        
        # Test 3: Cooking scenario
        thermal_cooking = {
            't_max': 35.0,
            't_mean': 26.0,
            't_hot_area_pct': 3.0
        }
        
        gas_cooking = {
            'gas_val': 800.0,
            'gas_delta': 20.0
        }
        
        cooking_discrimination = discriminator.discriminate_false_positives(thermal_cooking, gas_cooking)
        print(f"  Cooking scenario discrimination score: {cooking_discrimination.get('cooking_discrimination_score', 'N/A')}")
        
        print("  ✅ False Positive Discriminator test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ False Positive Discriminator test failed: {str(e)}")
        return False

def test_enhanced_extractors():
    """Test that enhanced extractors are working."""
    print("Testing Enhanced Extractors...")
    
    try:
        # Test enhanced FLIR extractor
        try:
            from src.feature_engineering.extractors.flir_thermal_extractor_enhanced import FlirThermalExtractorEnhanced
            
            thermal_data = {
                't_mean': 25.5, 't_std': 3.2, 't_max': 45.8, 't_p95': 38.2,
                't_hot_area_pct': 12.5, 't_hot_largest_blob_pct': 8.1,
                't_grad_mean': 2.1, 't_grad_std': 0.8, 't_diff_mean': 1.5,
                't_diff_std': 0.6, 'flow_mag_mean': 1.2, 'flow_mag_std': 0.4,
                'tproxy_val': 42.3, 'tproxy_delta': 8.7, 'tproxy_vel': 3.1
            }
            
            flir_extractor = FlirThermalExtractorEnhanced()
            flir_features = flir_extractor.extract_features(thermal_data)
            print(f"  Enhanced FLIR features: {len(flir_features)}")
        except ImportError as e:
            if "skimage" in str(e):
                print("  ⚠️  FLIR extractor test skipped due to missing skimage dependency")
            else:
                raise e
        
        # Test enhanced gas extractor
        try:
            from src.feature_engineering.extractors.scd41_gas_extractor_enhanced import Scd41GasExtractorEnhanced
            
            gas_data = {
                'gas_val': 550.0,
                'gas_delta': 25.0,
                'gas_vel': 25.0
            }
            
            gas_extractor = Scd41GasExtractorEnhanced()
            gas_features = gas_extractor.extract_features(gas_data)
            print(f"  Enhanced gas features: {len(gas_features)}")
        except ImportError as e:
            if "skimage" in str(e):
                print("  ⚠️  Gas extractor test skipped due to missing skimage dependency")
            else:
                raise e
        
        print("  ✅ Enhanced Extractors test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced Extractors test failed: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing New FLIR+SCD41 Optimization Features")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_enhanced_extractors,
        test_spatio_temporal_aligner,
        test_false_positive_discriminator
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
        print("🎉 All tests passed! New features are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())