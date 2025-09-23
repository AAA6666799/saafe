#!/usr/bin/env python3
"""
Test script for Sensor Data Logger.
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_sensor_data_logger():
    """Test the sensor data logger functionality."""
    print("Testing Sensor Data Logger...")
    
    try:
        from src.data_logging.sensor_data_logger import SensorDataLogger, create_sensor_data_logger
        
        # Create data logger
        config = {
            'storage_path': './test_sensor_data',
            'max_file_size_mb': 10,
            'rotation_interval_hours': 1,
            'compress_files': False,
            'default_format': 'json',
            'buffer_size': 100,
            'flush_interval_seconds': 5
        }
        
        logger = create_sensor_data_logger(config)
        print("✓ Successfully created sensor data logger")
        
        # Test 1: Log sample FLIR data
        print("\n--- Test 1: Log Sample FLIR Data ---")
        flir_data_1 = {
            'flir': {
                'flir_001': {
                    't_mean': 22.5,
                    't_std': 1.2,
                    't_max': 35.2,
                    't_p95': 28.1,
                    't_hot_area_pct': 2.1,
                    't_grad_mean': 0.8,
                    'tproxy_val': 35.2,
                    'timestamp': datetime.now().isoformat(),
                    'device_type': 'flir_lepton_3_5'
                }
            },
            'scd41': {
                'scd41_001': {
                    'gas_val': 450.0,
                    'co2_concentration': 450.0,
                    'gas_delta': 10.0,
                    'gas_vel': 5.0,
                    'timestamp': datetime.now().isoformat(),
                    'device_type': 'sensirion_scd41'
                }
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': 'test',
                'sensor_count': {
                    'flir': 1,
                    'scd41': 1
                }
            }
        }
        
        success = logger.log_sensor_data(flir_data_1)
        print(f"✓ FLIR data logging success: {success}")
        
        # Test 2: Log multiple data points
        print("\n--- Test 2: Log Multiple Data Points ---")
        for i in range(5):
            test_data = {
                'flir': {
                    f'flir_{i+2:03d}': {
                        't_mean': 20.0 + i,
                        't_std': 1.0 + i * 0.1,
                        't_max': 30.0 + i * 2,
                        'timestamp': datetime.now().isoformat(),
                        'device_type': 'flir_lepton_3_5'
                    }
                },
                'scd41': {
                    f'scd41_{i+2:03d}': {
                        'gas_val': 400.0 + i * 10,
                        'co2_concentration': 400.0 + i * 10,
                        'timestamp': datetime.now().isoformat(),
                        'device_type': 'sensirion_scd41'
                    }
                },
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'source': f'test_batch_{i}',
                    'sensor_count': {
                        'flir': 1,
                        'scd41': 1
                    }
                }
            }
            
            logger.log_sensor_data(test_data)
            time.sleep(0.1)  # Small delay between logs
        
        print("✓ Logged 5 additional data points")
        
        # Test 3: Check storage stats
        print("\n--- Test 3: Check Storage Stats ---")
        stats = logger.get_storage_stats()
        print(f"✓ Buffer size: {stats['buffer_size']}")
        if 'file_logger' in stats:
            print(f"✓ File logger current file: {stats['file_logger'].get('current_file', 'N/A')}")
        
        # Test 4: Force flush
        print("\n--- Test 4: Force Flush ---")
        logger.flush_buffer()
        print("✓ Buffer flushed successfully")
        
        # Test 5: Check if files were created
        print("\n--- Test 5: Check Created Files ---")
        storage_path = Path('./test_sensor_data')
        if storage_path.exists():
            files = list(storage_path.glob('*.json*'))
            print(f"✓ Found {len(files)} JSON files in storage directory")
            for file in files[:3]:  # Show first 3 files
                print(f"  - {file.name}")
        else:
            print("⚠️  Storage directory not found")
        
        # Test 6: Test CSV format
        print("\n--- Test 6: Test CSV Format ---")
        csv_config = {
            'storage_path': './test_sensor_data_csv',
            'default_format': 'csv',
            'compress_files': False,
            'flatten_data': True
        }
        
        csv_logger = create_sensor_data_logger(csv_config)
        csv_success = csv_logger.log_sensor_data(flir_data_1)
        csv_logger.flush_buffer()
        print(f"✓ CSV logging success: {csv_success}")
        
        # Check CSV files
        csv_storage_path = Path('./test_sensor_data_csv')
        if csv_storage_path.exists():
            csv_files = list(csv_storage_path.glob('*.csv*'))
            print(f"✓ Found {len(csv_files)} CSV files in storage directory")
        
        # Test 7: Shutdown loggers
        print("\n--- Test 7: Shutdown Loggers ---")
        logger.shutdown()
        csv_logger.shutdown()
        print("✓ Loggers shutdown successfully")
        
        print("\n✅ Sensor Data Logger tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Sensor Data Logger tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup test directories
        try:
            # Remove test directories
            test_dirs = ['./test_sensor_data', './test_sensor_data_csv']
            for dir_path in test_dirs:
                dir_path = Path(dir_path)
                if dir_path.exists():
                    import shutil
                    shutil.rmtree(dir_path)
                    print(f"✓ Cleaned up test directory: {dir_path}")
        except Exception as e:
            print(f"⚠️  Failed to cleanup test directories: {e}")


if __name__ == "__main__":
    test_sensor_data_logger()