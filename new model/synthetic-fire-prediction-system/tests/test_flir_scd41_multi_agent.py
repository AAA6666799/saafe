"""
Test file for FLIR+SCD41 multi-agent system processing.
"""

import sys
import os
import unittest
import logging
import json
from datetime import datetime
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

from src.agents.coordination.multi_agent_coordinator import MultiAgentFireDetectionSystem
from src.hardware.sensor_manager import SensorMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFlirScd41MultiAgentProcessing(unittest.TestCase):
    """Test FLIR+SCD41 multi-agent system processing."""
    
    def setUp(self):
        """Set up test environment."""
        # Configuration for FLIR+SCD41 system
        self.config = {
            'system_id': 'test_flir_scd41_system',
            'sensors': {'mode': SensorMode.SYNTHETIC},
            'agents': {
                'analysis': {
                    'fire_pattern': {
                        'confidence_threshold': 0.7,
                        'pattern_window_size': 50,
                        'fire_signatures': {}
                    }
                },
                'response': {
                    'emergency': {
                        'response_thresholds': {
                            'HIGH': 0.8,
                            'MEDIUM': 0.6,
                            'LOW': 0.4
                        }
                    }
                },
                'learning': {
                    'adaptive': {
                        'learning_rate': 0.01,
                        'performance_window': 100
                    }
                }
            }
        }
        
        # Create multi-agent system
        self.system = MultiAgentFireDetectionSystem(self.config)
        
        # Initialize the system
        self.assertTrue(self.system.initialize(), "System initialization should succeed")
        
        # Start the system
        self.assertTrue(self.system.start(), "System start should succeed")
    
    def tearDown(self):
        """Clean up after tests."""
        self.system.stop()
    
    def test_normal_conditions_processing(self):
        """Test processing of normal conditions with FLIR+SCD41 data."""
        # Normal conditions data in FLIR+SCD41 format
        normal_data = {
            'flir': {
                'flir_lepton35': {
                    't_mean': 22.0, 't_std': 2.0, 't_max': 25.0, 't_p95': 24.0,
                    't_hot_area_pct': 0.5, 't_hot_largest_blob_pct': 0.1,
                    't_grad_mean': 1.0, 't_grad_std': 0.5, 't_diff_mean': 0.1,
                    't_diff_std': 0.05, 'flow_mag_mean': 0.2, 'flow_mag_std': 0.1,
                    'tproxy_val': 25.0, 'tproxy_delta': 0.1, 'tproxy_vel': 0.05
                }
            },
            'scd41': {
                'scd41_co2': {
                    'gas_val': 450.0,  # Normal indoor CO₂
                    'gas_delta': 0.0,   # No change
                    'gas_vel': 0.0      # No velocity
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Process the data
        results = self.system.process_sensor_data(normal_data)
        
        # Verify results structure
        self.assertIn('fire_detection', results)
        self.assertIn('confidence_score', results['fire_detection'])
        self.assertIn('fire_detected', results['fire_detection'])
        
        # For normal conditions, should not detect fire
        self.assertFalse(results['fire_detection']['fire_detected'])
        self.assertLess(results['fire_detection']['confidence_score'], 0.5)
        
        logger.info(f"Normal conditions test passed. Confidence: {results['fire_detection']['confidence_score']}")
    
    def test_fire_conditions_processing(self):
        """Test processing of fire conditions with FLIR+SCD41 data."""
        # Fire conditions data in FLIR+SCD41 format
        fire_data = {
            'flir': {
                'flir_lepton35': {
                    't_mean': 55.0, 't_std': 15.0, 't_max': 85.0, 't_p95': 75.0,
                    't_hot_area_pct': 25.0, 't_hot_largest_blob_pct': 15.0,
                    't_grad_mean': 8.0, 't_grad_std': 3.0, 't_diff_mean': 5.0,
                    't_diff_std': 2.0, 'flow_mag_mean': 4.0, 'flow_mag_std': 1.5,
                    'tproxy_val': 80.0, 'tproxy_delta': 25.0, 'tproxy_vel': 12.0
                }
            },
            'scd41': {
                'scd41_co2': {
                    'gas_val': 1500.0,  # Elevated CO₂
                    'gas_delta': 300.0,  # Rapid increase
                    'gas_vel': 150.0    # High velocity
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Process the data
        results = self.system.process_sensor_data(fire_data)
        
        # Verify results structure
        self.assertIn('fire_detection', results)
        self.assertIn('confidence_score', results['fire_detection'])
        self.assertIn('fire_detected', results['fire_detection'])
        
        # For fire conditions, should detect fire with high confidence
        self.assertTrue(results['fire_detection']['fire_detected'])
        self.assertGreater(results['fire_detection']['confidence_score'], 0.7)
        
        logger.info(f"Fire conditions test passed. Confidence: {results['fire_detection']['confidence_score']}")
    
    def test_sensor_data_summarization(self):
        """Test that sensor data is properly summarized."""
        # Test data
        test_data = {
            'flir': {
                'flir_lepton35': {
                    't_mean': 30.0, 't_max': 60.0, 't_hot_area_pct': 10.0, 'tproxy_val': 55.0
                }
            },
            'scd41': {
                'scd41_co2': {
                    'gas_val': 800.0, 'gas_delta': 50.0, 'gas_vel': 25.0
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Get system status which includes sensor data summary
        results = self.system.process_sensor_data(test_data)
        summary = results.get('sensor_data_summary', {})
        
        # Verify FLIR data is summarized
        self.assertIn('flir_lepton35', summary)
        flir_summary = summary['flir_lepton35']
        self.assertEqual(flir_summary['max_temperature'], 60.0)
        self.assertEqual(flir_summary['avg_temperature'], 30.0)
        self.assertEqual(flir_summary['hot_area_percentage'], 10.0)
        
        # Verify SCD41 data is summarized
        self.assertIn('scd41_co2', summary)
        gas_summary = summary['scd41_co2']
        self.assertEqual(gas_summary['co2_concentration'], 800.0)
        self.assertEqual(gas_summary['co2_change_rate'], 50.0)
        self.assertEqual(gas_summary['co2_velocity'], 25.0)
        
        logger.info("Sensor data summarization test passed")


if __name__ == '__main__':
    unittest.main(verbosity=2)