#!/usr/bin/env python3
"""
Demonstration of Data Validation for FLIR+SCD41 Sensors.
This script shows how the data validation system works in the fire detection system.
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def demo_data_validation():
    """Demonstrate the complete data validation workflow."""
    print("🔥 Saafe Fire Detection System - Data Validation Demo 🔥")
    print("=" * 60)
    
    try:
        # Import required modules
        from src.hardware.sensor_manager import SensorManager, SensorMode
        from src.data_quality.iot_data_validator import create_iot_data_validator, DataQualityLevel
        
        print("✅ System modules loaded successfully")
        
        # Configuration for data validation
        config = {
            'mode': SensorMode.SYNTHETIC,
            'enable_data_validation': True,
            'data_validation_config': {
                'flir': {
                    'min_temperature': -40.0,
                    'max_temperature': 330.0,
                    'max_missing_features': 3,
                    'max_temperature_std': 50.0,
                },
                'scd41': {
                    'min_co2': 400.0,
                    'max_co2': 40000.0,
                    'max_delta': 1000.0,
                }
            },
            'mqtt': {
                'enabled': True,
                'broker': 'mqtt.saafe.ai',
                'port': 1883
            },
            'buffer_size': 1000
        }
        
        print("\n🔧 Data Validation Configuration:")
        print(f"   • Data validation enabled: {config['enable_data_validation']}")
        print(f"   • FLIR temperature range: {config['data_validation_config']['flir']['min_temperature']}°C to {config['data_validation_config']['flir']['max_temperature']}°C")
        print(f"   • SCD41 CO2 range: {config['data_validation_config']['scd41']['min_co2']} to {config['data_validation_config']['scd41']['max_co2']} ppm")
        print(f"   • Max CO2 delta: {config['data_validation_config']['scd41']['max_delta']} ppm/minute")
        
        # Create sensor manager
        print("\n🚀 Initializing Sensor Manager with Data Validation...")
        sensor_manager = SensorManager(config)
        
        # Initialize sensors
        print("🔌 Initializing sensors...")
        init_results = sensor_manager.initialize_sensors()
        print(f"✅ Sensor initialization completed")
        
        # Check if data validator is available
        if sensor_manager.data_validator:
            print("✅ Data validator successfully initialized")
        else:
            print("❌ Data validator not available")
            return False
        
        # Test scenarios
        test_scenarios = [
            {
                'name': 'Valid Normal Data',
                'data': {
                    'flir': {
                        'flir_001': {
                            't_mean': 22.5,
                            't_std': 1.8,
                            't_max': 28.3,
                            't_p95': 26.1,
                            't_hot_area_pct': 2.1,
                            't_grad_mean': 0.9,
                            'tproxy_val': 28.3,
                            'timestamp': datetime.now().isoformat()
                        }
                    },
                    'scd41': {
                        'scd41_001': {
                            'gas_val': 420.5,
                            'co2_concentration': 420.5,
                            'gas_delta': 5.2,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                },
                'expected_quality': DataQualityLevel.EXCELLENT
            },
            {
                'name': 'Valid Fire Condition Data',
                'data': {
                    'flir': {
                        'flir_001': {
                            't_mean': 45.2,
                            't_std': 8.3,
                            't_max': 85.7,
                            't_p95': 72.4,
                            't_hot_area_pct': 15.8,
                            't_grad_mean': 3.2,
                            'tproxy_val': 85.7,
                            'timestamp': datetime.now().isoformat()
                        }
                    },
                    'scd41': {
                        'scd41_001': {
                            'gas_val': 850.3,
                            'co2_concentration': 850.3,
                            'gas_delta': 120.5,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                },
                'expected_quality': DataQualityLevel.EXCELLENT
            },
            {
                'name': 'Data with Missing Features',
                'data': {
                    'flir': {
                        'flir_001': {
                            't_mean': 25.1,
                            't_max': 32.4,
                            # Missing several features
                            'timestamp': datetime.now().isoformat()
                        }
                    },
                    'scd41': {
                        'scd41_001': {
                            'gas_val': 430.2,
                            'co2_concentration': 430.2,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                },
                'expected_quality': DataQualityLevel.ACCEPTABLE
            },
            {
                'name': 'Data with Out of Range Values',
                'data': {
                    'flir': {
                        'flir_001': {
                            't_mean': 25.1,
                            't_std': 1.8,
                            't_max': 400.0,  # Too high
                            't_p95': 26.1,
                            't_hot_area_pct': 2.1,
                            't_grad_mean': 0.9,
                            'tproxy_val': 400.0,
                            'timestamp': datetime.now().isoformat()
                        }
                    },
                    'scd41': {
                        'scd41_001': {
                            'gas_val': 50000.0,  # Too high
                            'co2_concentration': 50000.0,
                            'gas_delta': 5.2,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                },
                'expected_quality': DataQualityLevel.UNUSABLE
            },
            {
                'name': 'Stale Data',
                'data': {
                    'flir': {
                        'flir_001': {
                            't_mean': 25.1,
                            't_std': 1.8,
                            't_max': 32.4,
                            't_p95': 26.1,
                            't_hot_area_pct': 2.1,
                            't_grad_mean': 0.9,
                            'tproxy_val': 32.4,
                            'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat()  # 15 minutes old
                        }
                    },
                    'scd41': {
                        'scd41_001': {
                            'gas_val': 430.2,
                            'co2_concentration': 430.2,
                            'gas_delta': 5.2,
                            'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat()  # 15 minutes old
                        }
                    }
                },
                'expected_quality': DataQualityLevel.POOR
            }
        ]
        
        print(f"\n🧪 Running Data Validation Tests:")
        print("-" * 50)
        
        initial_buffer_size = len(sensor_manager.data_buffer)
        valid_data_count = 0
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{i}. {scenario['name']}:")
            
            # Add metadata to the data
            scenario_data = scenario['data'].copy()
            scenario_data['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'source': 'mqtt',
                'test_scenario': scenario['name']
            }
            
            # Validate the data manually to show results
            validation_result = sensor_manager.data_validator.validate_sensor_data(scenario_data)
            
            print(f"   Quality Level: {validation_result.quality_level.value}")
            print(f"   Valid: {validation_result.is_valid}")
            print(f"   Issues: {len(validation_result.issues)}")
            if validation_result.issues:
                for issue in validation_result.issues[:2]:  # Show first 2 issues
                    print(f"     - {issue}")
                if len(validation_result.issues) > 2:
                    print(f"     - ... and {len(validation_result.issues) - 2} more issues")
            
            print(f"   Recommendations: {len(validation_result.recommendations)}")
            if validation_result.recommendations:
                for rec in validation_result.recommendations[:2]:  # Show first 2 recommendations
                    print(f"     - {rec}")
            
            # Simulate MQTT data reception
            sensor_manager._on_mqtt_data_received(scenario_data)
            
            # Check if data was added to buffer
            current_buffer_size = len(sensor_manager.data_buffer)
            data_added = current_buffer_size > initial_buffer_size + valid_data_count
            
            if data_added:
                valid_data_count += 1
                print(f"   ✅ Data added to buffer")
            else:
                print(f"   ❌ Data rejected (poor quality)")
        
        # Show final results
        print(f"\n📊 Final Results:")
        print(f"   • Initial buffer size: {initial_buffer_size}")
        print(f"   • Final buffer size: {len(sensor_manager.data_buffer)}")
        print(f"   • Valid data points added: {valid_data_count}")
        print(f"   • Data rejection rate: {((len(test_scenarios) - valid_data_count) / len(test_scenarios) * 100):.1f}%")
        
        # Show validation statistics
        stats = sensor_manager.data_validator.get_validation_statistics()
        print(f"\n📈 Validation Statistics:")
        print(f"   • Total validations: {stats['total_validations']}")
        print(f"   • Passed validations: {stats['passed_validations']}")
        print(f"   • Failed validations: {stats['failed_validations']}")
        print(f"   • Quality distribution:")
        for level, count in stats['quality_distribution'].items():
            if count > 0:
                print(f"     - {level}: {count}")
        
        print("\n" + "=" * 60)
        print("🎉 Data Validation Demo Completed Successfully!")
        print("💡 Key Benefits of Data Validation:")
        print("   • Ensures data quality before processing")
        print("   • Prevents invalid data from corrupting models")
        print("   • Provides actionable recommendations for issues")
        print("   • Maintains system reliability and accuracy")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    demo_data_validation()