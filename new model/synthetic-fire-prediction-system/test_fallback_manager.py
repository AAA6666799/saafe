#!/usr/bin/env python3
"""
Test script for Fallback Manager.
"""

import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_fallback_manager():
    """Test the fallback manager functionality."""
    print("Testing Fallback Manager...")
    
    try:
        from src.hardware.fallback_manager import FallbackManager, create_fallback_manager
        
        # Create fallback manager
        config = {
            'synthetic_data': {
                'synthetic_thermal': {
                    'image_width': 160,
                    'image_height': 120,
                    'base_temperature': 20.0
                },
                'synthetic_gas': {
                    'base_co2': 400.0,
                    'variation_factor': 0.1
                }
            }
        }
        
        manager = create_fallback_manager(config)
        print("✓ Successfully created fallback manager")
        
        # Test 1: Get fallback status
        print("\n--- Test 1: Fallback Status ---")
        status = manager.get_fallback_status()
        print(f"✓ Active strategies: {status['active_strategies']}")
        print(f"✓ Strategy names: {status['strategies']}")
        
        # Test 2: Update sensor data
        print("\n--- Test 2: Update Sensor Data ---")
        flir_data = {
            't_mean': 22.5,
            't_max': 35.2,
            't_hot_area_pct': 2.1,
            'timestamp': datetime.now().isoformat()
        }
        
        scd41_data = {
            'co2_concentration': 450.0,
            'gas_val': 450.0,
            'timestamp': datetime.now().isoformat()
        }
        
        manager.update_sensor_data("flir_001", "flir", flir_data)
        manager.update_sensor_data("scd41_001", "scd41", scd41_data)
        print("✓ Successfully updated sensor data")
        
        # Test 3: Handle FLIR sensor failure
        print("\n--- Test 3: Handle FLIR Sensor Failure ---")
        flir_failure_context = {
            'sensor_id': 'flir_002',
            'sensor_type': 'flir',
            'error': 'Connection timeout',
            'timestamp': datetime.now().isoformat(),
            'missing_sensors': {'flir': ['flir_002']}
        }
        
        flir_fallback = manager.handle_sensor_failure(flir_failure_context)
        print(f"✓ FLIR fallback generated: {bool(flir_fallback)}")
        if flir_fallback and 'flir' in flir_fallback:
            print(f"✓ FLIR fallback data keys: {list(flir_fallback['flir'].keys())}")
        
        # Test 4: Handle SCD41 sensor failure
        print("\n--- Test 4: Handle SCD41 Sensor Failure ---")
        scd41_failure_context = {
            'sensor_id': 'scd41_002',
            'sensor_type': 'scd41',
            'error': 'Sensor not responding',
            'timestamp': datetime.now().isoformat(),
            'missing_sensors': {'scd41': ['scd41_002']}
        }
        
        scd41_fallback = manager.handle_sensor_failure(scd41_failure_context)
        print(f"✓ SCD41 fallback generated: {bool(scd41_fallback)}")
        if scd41_fallback and 'scd41' in scd41_fallback:
            print(f"✓ SCD41 fallback data keys: {list(scd41_fallback['scd41'].keys())}")
        
        # Test 5: Handle system-wide failure
        print("\n--- Test 5: Handle System-wide Failure ---")
        system_failure_context = {
            'error': 'Complete system failure',
            'timestamp': datetime.now().isoformat(),
            'missing_sensors': {
                'flir': ['flir_001', 'flir_002'],
                'scd41': ['scd41_001', 'scd41_002']
            }
        }
        
        system_fallback = manager.handle_sensor_failure(system_failure_context)
        print(f"✓ System fallback generated: {bool(system_fallback)}")
        print(f"✓ System fallback metadata source: {system_fallback.get('metadata', {}).get('source', 'unknown')}")
        
        # Test 6: Test fallback status after updates
        print("\n--- Test 6: Fallback Status After Updates ---")
        updated_status = manager.get_fallback_status()
        print(f"✓ Last known sensors: {updated_status['last_known_sensors']}")
        print(f"✓ Cached sensor types: {updated_status['cached_sensor_types']}")
        
        # Test 7: Test multiple failures for same sensor (should use last known good)
        print("\n--- Test 7: Multiple Failures for Same Sensor ---")
        flir_fallback_2 = manager.handle_sensor_failure(flir_failure_context)
        print(f"✓ Second FLIR fallback generated: {bool(flir_fallback_2)}")
        if flir_fallback_2 and 'metadata' in flir_fallback_2:
            strategy = flir_fallback_2['metadata'].get('applied_strategy', 'unknown')
            print(f"✓ Applied strategy for second failure: {strategy}")
        
        print("\n✅ Fallback Manager tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Fallback Manager tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_fallback_manager()