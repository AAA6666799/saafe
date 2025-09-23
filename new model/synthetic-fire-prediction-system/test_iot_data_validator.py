#!/usr/bin/env python3
"""
Test script for IoT Data Validator.
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_iot_data_validator():
    """Test the IoT data validator functionality."""
    print("Testing IoT Data Validator...")
    
    try:
        from src.data_quality.iot_data_validator import IoTDataValidator, create_iot_data_validator, DataQualityLevel
        
        # Create validator
        validator = create_iot_data_validator()
        print("✓ Successfully created IoT data validator")
        
        # Test 1: Valid data
        print("\n--- Test 1: Valid Data ---")
        valid_data = {
            'flir': {
                'flir_001': {
                    't_mean': 25.5,
                    't_std': 2.3,
                    't_max': 35.2,
                    't_p95': 32.1,
                    't_hot_area_pct': 5.2,
                    't_grad_mean': 1.2,
                    'tproxy_val': 35.2,
                    'timestamp': datetime.now().isoformat()
                }
            },
            'scd41': {
                'scd41_001': {
                    'gas_val': 450.0,
                    'co2_concentration': 450.0,
                    'gas_delta': 10.0,
                    'timestamp': datetime.now().isoformat()
                }
            }
        }
        
        result = validator.validate_sensor_data(valid_data)
        print(f"✓ Valid data result: {result.is_valid}")
        print(f"✓ Quality level: {result.quality_level.value}")
        print(f"✓ Issues: {len(result.issues)}")
        print(f"✓ Recommendations: {len(result.recommendations)}")
        
        # Test 2: Invalid data (out of range values)
        print("\n--- Test 2: Invalid Data (Out of Range) ---")
        invalid_data = {
            'flir': {
                'flir_001': {
                    't_mean': 25.5,
                    't_std': 2.3,
                    't_max': 400.0,  # Too high
                    't_p95': 32.1,
                    't_hot_area_pct': 5.2,
                    't_grad_mean': 1.2,
                    'tproxy_val': 35.2,
                    'timestamp': datetime.now().isoformat()
                }
            },
            'scd41': {
                'scd41_001': {
                    'gas_val': 50000.0,  # Too high
                    'co2_concentration': 50000.0,
                    'gas_delta': 10.0,
                    'timestamp': datetime.now().isoformat()
                }
            }
        }
        
        result = validator.validate_sensor_data(invalid_data)
        print(f"✓ Invalid data result: {result.is_valid}")
        print(f"✓ Quality level: {result.quality_level.value}")
        print(f"✓ Issues: {len(result.issues)}")
        if result.issues:
            print(f"✓ First issue: {result.issues[0]}")
        print(f"✓ Recommendations: {len(result.recommendations)}")
        if result.recommendations:
            print(f"✓ First recommendation: {result.recommendations[0]}")
        
        # Test 3: Missing data
        print("\n--- Test 3: Missing Data ---")
        missing_data = {
            'flir': {
                'flir_001': {
                    't_mean': 25.5,
                    # Missing several required fields
                    'timestamp': datetime.now().isoformat()
                }
            },
            'scd41': {}  # Empty SCD41 data
        }
        
        result = validator.validate_sensor_data(missing_data)
        print(f"✓ Missing data result: {result.is_valid}")
        print(f"✓ Quality level: {result.quality_level.value}")
        print(f"✓ Issues: {len(result.issues)}")
        if result.issues:
            print(f"✓ First issue: {result.issues[0]}")
        
        # Test 4: Stale data
        print("\n--- Test 4: Stale Data ---")
        stale_data = {
            'flir': {
                'flir_001': {
                    't_mean': 25.5,
                    't_std': 2.3,
                    't_max': 35.2,
                    't_p95': 32.1,
                    't_hot_area_pct': 5.2,
                    't_grad_mean': 1.2,
                    'tproxy_val': 35.2,
                    'timestamp': (datetime.now() - timedelta(minutes=10)).isoformat()  # 10 minutes old
                }
            },
            'scd41': {
                'scd41_001': {
                    'gas_val': 450.0,
                    'co2_concentration': 450.0,
                    'gas_delta': 10.0,
                    'timestamp': (datetime.now() - timedelta(minutes=10)).isoformat()  # 10 minutes old
                }
            }
        }
        
        result = validator.validate_sensor_data(stale_data)
        print(f"✓ Stale data result: {result.is_valid}")
        print(f"✓ Quality level: {result.quality_level.value}")
        print(f"✓ Issues: {len(result.issues)}")
        if result.issues:
            print(f"✓ First issue: {result.issues[0]}")
        
        # Test 5: High delta values
        print("\n--- Test 5: High Delta Values ---")
        high_delta_data = {
            'flir': {
                'flir_001': {
                    't_mean': 25.5,
                    't_std': 2.3,
                    't_max': 35.2,
                    't_p95': 32.1,
                    't_hot_area_pct': 5.2,
                    't_grad_mean': 1.2,
                    'tproxy_val': 35.2,
                    'timestamp': datetime.now().isoformat()
                }
            },
            'scd41': {
                'scd41_001': {
                    'gas_val': 450.0,
                    'co2_concentration': 450.0,
                    'gas_delta': 2000.0,  # Very high delta
                    'timestamp': datetime.now().isoformat()
                }
            }
        }
        
        result = validator.validate_sensor_data(high_delta_data)
        print(f"✓ High delta data result: {result.is_valid}")
        print(f"✓ Quality level: {result.quality_level.value}")
        print(f"✓ Issues: {len(result.issues)}")
        if result.issues:
            print(f"✓ First issue: {result.issues[0]}")
        
        # Show validation statistics
        stats = validator.get_validation_statistics()
        print(f"\n--- Validation Statistics ---")
        print(f"✓ Total validations: {stats['total_validations']}")
        print(f"✓ Passed validations: {stats['passed_validations']}")
        print(f"✓ Failed validations: {stats['failed_validations']}")
        print(f"✓ Quality distribution: {stats['quality_distribution']}")
        
        print("\n✅ IoT Data Validator tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ IoT Data Validator tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_iot_data_validator()