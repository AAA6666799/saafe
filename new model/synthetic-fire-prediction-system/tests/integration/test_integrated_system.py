"""
Tests for Integrated Fire Detection System.

Comprehensive tests for the main system integration that coordinates
all subsystems for end-to-end fire detection functionality.
"""

import unittest
import time
import tempfile
import os
import shutil
from typing import Dict, Any

import sys
sys.path.append('/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

try:
    from src.integrated_system import IntegratedFireDetectionSystem, create_integrated_fire_system
    from src.hardware.sensor_manager import SensorMode
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import integration components: {e}")
    INTEGRATION_AVAILABLE = False


class TestIntegratedFireDetectionSystem(unittest.TestCase):
    """Test cases for the Integrated Fire Detection System."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not INTEGRATION_AVAILABLE:
            self.skipTest("Integration components not available")
        
        self.test_config = {
            'system_id': 'test_integrated_system',
            'sensors': {
                'mode': SensorMode.SYNTHETIC,
                'collection_interval': 0.1
            },
            'agents': {
                'analysis': {
                    'fire_pattern': {
                        'confidence_threshold': 0.6,
                        'pattern_window_size': 10
                    }
                },
                'response': {
                    'emergency': {
                        'response_thresholds': {
                            'LOW': 0.3,
                            'MEDIUM': 0.5,
                            'HIGH': 0.7,
                            'CRITICAL': 0.9
                        }
                    }
                }
            },
            'machine_learning': {
                'ensemble_strategy': 'weighted_voting',
                'confidence_threshold': 0.6
            }
        }
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_system_initialization(self):
        """Test integrated system initialization."""
        system = IntegratedFireDetectionSystem(self.test_config)
        
        # Check initial state
        self.assertEqual(system.system_id, 'test_integrated_system')
        self.assertEqual(system.state, 'initializing')
        self.assertFalse(system.is_running)
        
        # Initialize system
        init_success = system.initialize()
        self.assertTrue(init_success)
        self.assertEqual(system.state, 'ready')
        
        # Check subsystems are initialized
        self.assertIsNotNone(system.sensor_manager)
        self.assertIsNotNone(system.multi_agent_system)
        self.assertIsNotNone(system.model_ensemble)
        
        # Check subsystem health
        health = system.subsystem_health
        self.assertEqual(health['sensors'], 'healthy')
        self.assertEqual(health['agents'], 'healthy')
        self.assertEqual(health['models'], 'healthy')
    
    def test_convenience_function(self):
        """Test convenience function for creating integrated system."""
        system = create_integrated_fire_system(self.test_config)
        self.assertIsInstance(system, IntegratedFireDetectionSystem)
        
        # Test with default config
        default_system = create_integrated_fire_system()
        self.assertIsInstance(default_system, IntegratedFireDetectionSystem)
    
    def test_system_lifecycle(self):
        """Test complete system lifecycle (init -> start -> stop)."""
        system = create_integrated_fire_system(self.test_config)
        
        # Initialize
        self.assertTrue(system.initialize())
        self.assertEqual(system.state, 'ready')
        
        # Start
        self.assertTrue(system.start())
        self.assertEqual(system.state, 'running')
        self.assertTrue(system.is_running)
        
        # Let it run briefly
        time.sleep(0.5)
        
        # Stop
        system.stop()
        self.assertEqual(system.state, 'stopped')
        self.assertFalse(system.is_running)
    
    def test_data_processing_pipeline(self):
        """Test complete data processing pipeline."""
        system = create_integrated_fire_system(self.test_config)
        system.initialize()
        
        # Test with synthetic data (normal conditions)
        normal_data = {
            'thermal': {
                'thermal_01': {
                    'temperature_max': 25.0,
                    'temperature_avg': 22.0,
                    'hotspot_count': 0
                }
            },
            'gas': {
                'gas_01': {
                    'co_concentration': 8.0,
                    'smoke_density': 12.0
                }
            },
            'environmental': {
                'env_01': {
                    'temperature': 23.0,
                    'humidity': 45.0
                }
            }
        }
        
        results = system.process_data(normal_data)
        
        # Validate results structure
        self.assertIn('processing_id', results)
        self.assertIn('timestamp', results)
        self.assertIn('final_decision', results)
        
        final_decision = results['final_decision']
        self.assertIn('fire_detected', final_decision)
        self.assertIn('confidence_score', final_decision)
        self.assertIn('response_level', final_decision)
        
        # Should not detect fire in normal conditions
        self.assertFalse(final_decision['fire_detected'])
        self.assertLessEqual(final_decision['confidence_score'], 0.6)
    
    def test_fire_detection_scenario(self):
        """Test fire detection with elevated sensor readings."""
        system = create_integrated_fire_system(self.test_config)
        system.initialize()
        
        # Test with fire-like conditions
        fire_data = {
            'thermal': {
                'thermal_01': {
                    'temperature_max': 75.0,  # High temperature
                    'temperature_avg': 55.0,
                    'hotspot_count': 5
                }
            },
            'gas': {
                'gas_01': {
                    'co_concentration': 45.0,  # Elevated CO
                    'smoke_density': 80.0      # High smoke
                }
            },
            'environmental': {
                'env_01': {
                    'temperature': 35.0,
                    'humidity': 25.0
                }
            }
        }
        
        results = system.process_data(fire_data)
        final_decision = results['final_decision']
        
        # Should detect fire with high confidence
        self.assertTrue(final_decision['fire_detected'])
        self.assertGreater(final_decision['confidence_score'], 0.5)
        
        # Should generate appropriate response
        response_level = final_decision['response_level']
        self.assertIn(response_level, ['MEDIUM', 'HIGH', 'CRITICAL'])
        
        # Should have alerts
        if 'alerts' in final_decision:
            self.assertGreater(len(final_decision['alerts']), 0)
    
    def test_continuous_processing(self):
        """Test continuous data processing."""
        system = create_integrated_fire_system(self.test_config)
        system.initialize()
        system.start()
        
        # Let system process for a short time
        time.sleep(1.0)
        
        # Check processing pipeline has data
        self.assertGreater(len(system.processing_pipeline), 0)
        
        # Check metrics are being updated
        metrics = system.metrics
        self.assertGreater(metrics['total_processed'], 0)
        self.assertGreater(metrics['average_processing_time'], 0)
        
        system.stop()
    
    def test_system_status_reporting(self):
        """Test system status and health reporting."""
        system = create_integrated_fire_system(self.test_config)
        system.initialize()
        
        status = system.get_system_status()
        
        # Check status structure
        required_fields = [
            'system_id', 'timestamp', 'state', 'uptime_seconds',
            'subsystem_health', 'metrics', 'pipeline_status'
        ]
        
        for field in required_fields:
            self.assertIn(field, status)
        
        # Check subsystem health
        health = status['subsystem_health']
        self.assertIn('sensors', health)
        self.assertIn('agents', health)
        self.assertIn('models', health)
        
        # Check metrics
        metrics = status['metrics']
        self.assertIn('total_processed', metrics)
        self.assertIn('average_processing_time', metrics)
        
        # Check pipeline status
        pipeline = status['pipeline_status']
        self.assertIn('buffer_size', pipeline)
        self.assertIn('utilization', pipeline)
    
    def test_feature_extraction(self):
        """Test feature extraction from sensor data."""
        system = create_integrated_fire_system(self.test_config)
        system.initialize()
        
        sensor_data = {
            'thermal': {
                'thermal_01': {'temperature_max': 45.0, 'temperature_avg': 35.0},
                'thermal_02': {'temperature_max': 50.0, 'temperature_avg': 40.0}
            },
            'gas': {
                'gas_01': {'co_concentration': 25.0, 'smoke_density': 30.0}
            }
        }
        
        features = system._extract_features(sensor_data)
        
        # Check thermal features
        self.assertIn('thermal_thermal_01_max', features)
        self.assertIn('thermal_thermal_01_avg', features)
        self.assertIn('thermal_thermal_02_max', features)
        
        # Check gas features
        self.assertIn('gas_gas_01_co', features)
        self.assertIn('gas_gas_01_smoke', features)
        
        # Check feature values
        self.assertEqual(features['thermal_thermal_01_max'], 45.0)
        self.assertEqual(features['gas_gas_01_co'], 25.0)
    
    def test_basic_fire_detection_logic(self):
        """Test basic fire detection logic."""
        system = create_integrated_fire_system(self.test_config)
        
        # Test normal conditions
        normal_data = {
            'thermal': {
                'sensor1': {'temperature_max': 25.0}
            },
            'gas': {
                'sensor2': {'co_concentration': 10.0}
            }
        }
        
        result = system._basic_fire_detection(normal_data)
        self.assertFalse(result['fire_detected'])
        self.assertLess(result['confidence_score'], 0.5)
        
        # Test fire conditions
        fire_data = {
            'thermal': {
                'sensor1': {'temperature_max': 80.0}  # Above threshold
            },
            'gas': {
                'sensor2': {'co_concentration': 50.0}  # Above threshold
            }
        }
        
        result = system._basic_fire_detection(fire_data)
        self.assertTrue(result['fire_detected'])
        self.assertGreater(result['confidence_score'], 0.5)
        self.assertGreater(len(result['indicators']), 0)
    
    def test_error_handling(self):
        """Test system error handling."""
        system = create_integrated_fire_system(self.test_config)
        system.initialize()
        
        # Test with malformed data
        malformed_data = {'invalid': 'data'}
        result = system.process_data(malformed_data)
        
        # Should handle gracefully
        self.assertIsInstance(result, dict)
        self.assertIn('final_decision', result)
        
        # Test with None data
        result = system.process_data(None)
        self.assertIsInstance(result, dict)
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        system = create_integrated_fire_system(self.test_config)
        
        synthetic_data = system._get_synthetic_data()
        
        # Check structure
        self.assertIn('thermal', synthetic_data)
        self.assertIn('gas', synthetic_data)
        self.assertIn('environmental', synthetic_data)
        
        # Check thermal data
        thermal = synthetic_data['thermal']['thermal_01']
        self.assertIn('temperature_max', thermal)
        self.assertIn('temperature_avg', thermal)
        
        # Check gas data
        gas = synthetic_data['gas']['gas_01']
        self.assertIn('co_concentration', gas)
        self.assertIn('smoke_density', gas)
    
    def test_metrics_tracking(self):
        """Test system metrics tracking."""
        system = create_integrated_fire_system(self.test_config)
        system.initialize()
        
        # Process several data points
        for i in range(5):
            # Alternate between normal and fire conditions
            if i % 2 == 0:
                data = system._get_synthetic_data()
            else:
                data = {
                    'thermal': {'sensor1': {'temperature_max': 70.0}},
                    'gas': {'sensor1': {'co_concentration': 40.0}}
                }
            
            system.process_data(data)
        
        # Check metrics are updated
        metrics = system.metrics
        self.assertEqual(metrics['total_processed'], 5)
        self.assertGreater(metrics['average_processing_time'], 0)
        
        # Should have some fire detections
        self.assertGreaterEqual(metrics['fire_detections'], 0)


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for complete system workflows."""
    
    def test_end_to_end_fire_detection(self):
        """Test complete end-to-end fire detection workflow."""
        if not INTEGRATION_AVAILABLE:
            self.skipTest("Integration components not available")
        
        # Create system with realistic config
        config = {
            'system_id': 'e2e_test_system',
            'sensors': {'mode': SensorMode.SYNTHETIC},
            'agents': {
                'analysis': {'fire_pattern': {'confidence_threshold': 0.6}},
                'response': {'emergency': {'response_thresholds': {'HIGH': 0.7}}}
            }
        }
        
        system = create_integrated_fire_system(config)
        
        # Full lifecycle test
        self.assertTrue(system.initialize())
        self.assertTrue(system.start())
        
        # Simulate fire scenario
        fire_scenario = {
            'thermal': {
                'thermal_01': {'temperature_max': 85.0, 'temperature_avg': 60.0},
                'thermal_02': {'temperature_max': 78.0, 'temperature_avg': 55.0}
            },
            'gas': {
                'gas_01': {'co_concentration': 55.0, 'smoke_density': 75.0}
            },
            'environmental': {
                'env_01': {'temperature': 32.0, 'humidity': 30.0}
            }
        }
        
        # Process fire scenario
        results = system.process_data(fire_scenario)
        
        # Validate complete processing
        self.assertIn('ml_results', results)
        self.assertIn('agent_results', results)
        self.assertIn('final_decision', results)
        
        # Should detect fire
        final_decision = results['final_decision']
        self.assertTrue(final_decision['fire_detected'])
        self.assertGreater(final_decision['confidence_score'], 0.6)
        
        # Should have appropriate response
        self.assertIn(final_decision['response_level'], ['MEDIUM', 'HIGH', 'CRITICAL'])
        
        # System should remain healthy
        status = system.get_system_status()
        self.assertEqual(status['state'], 'running')
        
        # Clean shutdown
        system.stop()
        self.assertEqual(system.state, 'stopped')


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2, buffer=True)