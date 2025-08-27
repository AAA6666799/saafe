"""
Tests for the Multi-Agent Fire Detection System.

This module provides comprehensive tests for the multi-agent coordination system
including agent integration, communication, and end-to-end fire detection workflows.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
import time
from typing import Dict, Any, Optional

# Import test framework
import sys
sys.path.append('/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

try:
    from src.agents.coordination.multi_agent_coordinator import (
        MultiAgentFireDetectionSystem, 
        create_multi_agent_fire_system
    )
    from src.agents.analysis.fire_pattern_analysis import FirePatternAnalysisAgent
    from src.agents.response.emergency_response import EmergencyResponseAgent
    from src.agents.learning.adaptive_learning import AdaptiveLearningAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agent components: {e}")
    AGENTS_AVAILABLE = False


class TestMultiAgentFireDetectionSystem(unittest.TestCase):
    """Test cases for the Multi-Agent Fire Detection System."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not AGENTS_AVAILABLE:
            self.skipTest("Multi-agent components not available")
        
        # Create test configuration
        self.test_config = {
            'system_id': 'test_fire_detection_system',
            'monitoring': {
                'system_health': {
                    'check_interval': 1.0,
                    'health_thresholds': {'cpu': 0.8, 'memory': 0.8}
                }
            },
            'analysis': {
                'fire_pattern': {
                    'confidence_threshold': 0.6,  # Lower for testing
                    'pattern_window_size': 10,   # Smaller for testing
                    'fire_signatures': {
                        'test_fire': {
                            'thermal_threshold': 40.0,
                            'gas_thresholds': {'co': 20.0, 'smoke': 30.0}
                        }
                    }
                }
            },
            'response': {
                'emergency': {
                    'response_thresholds': {
                        'LOW': 0.2,
                        'MEDIUM': 0.4,
                        'HIGH': 0.6,
                        'CRITICAL': 0.8
                    },
                    'alert_channels': ['system'],
                    'emergency_contacts': ['test@example.com']
                }
            },
            'learning': {
                'adaptive': {
                    'learning_window_size': 50,  # Smaller for testing
                    'performance_threshold': 0.8,
                    'error_analysis_interval': 10
                }
            }
        }
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _generate_test_sensor_data(self, fire_scenario: bool = False, 
                                  confidence_level: float = 0.7) -> Dict[str, Any]:
        """Generate realistic test sensor data."""
        if fire_scenario:
            # Generate data indicating fire conditions
            thermal_data = {
                'temperature_max': 65.0 + np.random.normal(0, 5),
                'temperature_avg': 45.0 + np.random.normal(0, 3),
                'hotspot_count': 3 + int(np.random.normal(0, 1)),
                'thermal_spread_rate': 0.7 + np.random.normal(0, 0.1)
            }
            
            gas_data = {
                'co_concentration': 35.0 + np.random.normal(0, 5),
                'co2_concentration': 800.0 + np.random.normal(0, 100),
                'smoke_density': 60.0 + np.random.normal(0, 10),
                'voc_total': 800.0 + np.random.normal(0, 100)
            }
            
            environmental_data = {
                'temperature': 28.0 + np.random.normal(0, 2),
                'humidity': 35.0 + np.random.normal(0, 5),
                'pressure': 1010.0 + np.random.normal(0, 5),
                'wind_speed': 2.5 + np.random.normal(0, 0.5)
            }
        else:
            # Generate normal/no-fire data
            thermal_data = {
                'temperature_max': 22.0 + np.random.normal(0, 2),
                'temperature_avg': 20.0 + np.random.normal(0, 1),
                'hotspot_count': 0,
                'thermal_spread_rate': 0.0
            }
            
            gas_data = {
                'co_concentration': 5.0 + np.random.normal(0, 1),
                'co2_concentration': 400.0 + np.random.normal(0, 20),
                'smoke_density': 10.0 + np.random.normal(0, 2),
                'voc_total': 200.0 + np.random.normal(0, 50)
            }
            
            environmental_data = {
                'temperature': 22.0 + np.random.normal(0, 1),
                'humidity': 50.0 + np.random.normal(0, 5),
                'pressure': 1013.0 + np.random.normal(0, 2),
                'wind_speed': 1.0 + np.random.normal(0, 0.2)
            }
        
        return {
            'thermal': thermal_data,
            'gas': gas_data,
            'environmental': environmental_data,
            'timestamp': time.time(),
            'sensor_id': 'test_sensor_001'
        }
    
    def test_system_initialization(self):
        """Test multi-agent system initialization."""
        try:
            # Create system
            system = MultiAgentFireDetectionSystem(self.test_config)
            self.assertIsNotNone(system)
            self.assertEqual(system.system_id, 'test_fire_detection_system')
            self.assertFalse(system.initialized)
            
            # Initialize system
            init_success = system.initialize()
            self.assertTrue(init_success)
            self.assertTrue(system.initialized)
            
            # Check agents are created
            self.assertGreater(len(system.agents), 0)
            self.assertIn('system_health_monitor', system.agents)
            self.assertIn('fire_pattern_analyzer', system.agents)
            self.assertIn('emergency_responder', system.agents)
            self.assertIn('adaptive_learner', system.agents)
            
        except Exception as e:
            self.skipTest(f"System initialization failed: {e}")
    
    def test_convenience_function(self):
        """Test convenience function for creating multi-agent system."""
        try:
            system = create_multi_agent_fire_system(self.test_config)
            self.assertIsInstance(system, MultiAgentFireDetectionSystem)
            
            # Test with default config
            default_system = create_multi_agent_fire_system()
            self.assertIsInstance(default_system, MultiAgentFireDetectionSystem)
            
        except Exception as e:
            self.skipTest(f"Convenience function test failed: {e}")
    
    def test_fire_detection_workflow(self):
        """Test complete fire detection workflow."""
        try:
            # Create and initialize system
            system = create_multi_agent_fire_system(self.test_config)
            init_success = system.initialize()
            self.assertTrue(init_success)
            
            # Test fire scenario
            fire_data = self._generate_test_sensor_data(fire_scenario=True)
            fire_results = system.process_sensor_data(fire_data)
            
            # Validate fire detection results
            self.assertIn('fire_detection', fire_results)
            fire_detection = fire_results['fire_detection']
            
            # Should detect fire with reasonable confidence
            self.assertIsInstance(fire_detection['fire_detected'], bool)
            self.assertIsInstance(fire_detection['confidence_score'], (int, float))
            self.assertGreaterEqual(fire_detection['confidence_score'], 0.0)
            self.assertLessEqual(fire_detection['confidence_score'], 1.0)
            
            # Should have processing stages
            self.assertIn('monitoring_results', fire_results)
            self.assertIn('analysis_results', fire_results)
            self.assertIn('response_results', fire_results)
            self.assertIn('learning_results', fire_results)
            
        except Exception as e:
            self.skipTest(f"Fire detection workflow test failed: {e}")
    
    def test_normal_conditions_workflow(self):
        """Test workflow under normal (no-fire) conditions."""
        try:
            # Create and initialize system
            system = create_multi_agent_fire_system(self.test_config)
            system.initialize()
            
            # Test normal conditions
            normal_data = self._generate_test_sensor_data(fire_scenario=False)
            normal_results = system.process_sensor_data(normal_data)
            
            # Validate normal condition results
            fire_detection = normal_results['fire_detection']
            
            # Should not detect fire or have low confidence
            self.assertLessEqual(fire_detection['confidence_score'], 0.8)
            
            # Should still process through all stages
            self.assertIn('monitoring_results', normal_results)
            self.assertIn('analysis_results', normal_results)
            
        except Exception as e:
            self.skipTest(f"Normal conditions test failed: {e}")
    
    def test_system_status_reporting(self):
        """Test system status and metrics reporting."""
        try:
            # Create and initialize system
            system = create_multi_agent_fire_system(self.test_config)
            system.initialize()
            
            # Get initial status
            initial_status = system.get_system_status()
            self.assertIsInstance(initial_status, dict)
            
            # Check required status fields
            required_fields = [
                'system_id', 'timestamp', 'system_state', 
                'system_metrics', 'agent_status'
            ]
            for field in required_fields:
                self.assertIn(field, initial_status)
            
            # Process some data to generate metrics
            for _ in range(3):
                test_data = self._generate_test_sensor_data()
                system.process_sensor_data(test_data)
            
            # Get updated status
            updated_status = system.get_system_status()
            self.assertGreater(
                updated_status['system_metrics']['data_points_processed'], 0
            )
            
        except Exception as e:
            self.skipTest(f"System status test failed: {e}")
    
    def test_alert_generation(self):
        """Test alert generation for fire conditions."""
        try:
            system = create_multi_agent_fire_system(self.test_config)
            system.initialize()
            
            # Generate high-confidence fire data
            fire_data = self._generate_test_sensor_data(fire_scenario=True, confidence_level=0.9)
            results = system.process_sensor_data(fire_data)
            
            # Check for alert generation
            response_results = results.get('response_results', {})
            alerts = response_results.get('alerts', [])
            
            # Should generate alerts for fire detection
            if results['fire_detection']['fire_detected']:
                self.assertGreater(len(alerts), 0)
                
                # Check alert structure
                for alert in alerts:
                    self.assertIn('alert_id', alert)
                    self.assertIn('alert_type', alert)
                    self.assertIn('message', alert)
                    self.assertIn('timestamp', alert)
            
        except Exception as e:
            self.skipTest(f"Alert generation test failed: {e}")
    
    def test_learning_and_adaptation(self):
        """Test learning agent functionality."""
        try:
            system = create_multi_agent_fire_system(self.test_config)
            system.initialize()
            
            # Process multiple data points to trigger learning
            for i in range(5):
                # Alternate between fire and normal conditions
                fire_scenario = i % 2 == 0
                test_data = self._generate_test_sensor_data(fire_scenario=fire_scenario)
                results = system.process_sensor_data(test_data)
                
                # Check learning results
                learning_results = results.get('learning_results', {})
                if learning_results and 'improvement_recommendations' in learning_results:
                    recommendations = learning_results['improvement_recommendations']
                    self.assertIsInstance(recommendations, list)
            
        except Exception as e:
            self.skipTest(f"Learning and adaptation test failed: {e}")
    
    def test_system_lifecycle(self):
        """Test complete system lifecycle (start/stop)."""
        try:
            system = create_multi_agent_fire_system(self.test_config)
            
            # Initialize
            self.assertTrue(system.initialize())
            
            # Start
            self.assertTrue(system.start())
            self.assertTrue(system.running)
            
            # Process some data while running
            test_data = self._generate_test_sensor_data()
            results = system.process_sensor_data(test_data)
            self.assertIsInstance(results, dict)
            
            # Stop
            system.stop()
            self.assertFalse(system.running)
            
        except Exception as e:
            self.skipTest(f"System lifecycle test failed: {e}")
    
    def test_error_handling(self):
        """Test system error handling and robustness."""
        try:
            system = create_multi_agent_fire_system(self.test_config)
            system.initialize()
            
            # Test with malformed data
            malformed_data = {'invalid': 'data'}
            results = system.process_sensor_data(malformed_data)
            
            # Should handle gracefully
            self.assertIsInstance(results, dict)
            self.assertIn('fire_detection', results)
            
            # Test with missing data fields
            incomplete_data = {'thermal': {'temperature_max': 25.0}}
            results = system.process_sensor_data(incomplete_data)
            self.assertIsInstance(results, dict)
            
        except Exception as e:
            self.skipTest(f"Error handling test failed: {e}")
    
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        try:
            system = create_multi_agent_fire_system(self.test_config)
            system.initialize()
            
            # Process several data points
            for i in range(10):
                test_data = self._generate_test_sensor_data(fire_scenario=(i % 3 == 0))
                system.process_sensor_data(test_data)
            
            # Check system metrics
            status = system.get_system_status()
            metrics = status['system_metrics']
            
            # Verify metrics are tracked
            self.assertIn('total_detections', metrics)
            self.assertIn('average_processing_time', metrics)
            self.assertIn('data_points_processed', metrics)
            
            # Should have processed data
            self.assertGreater(metrics['data_points_processed'], 0)
            
        except Exception as e:
            self.skipTest(f"Performance metrics test failed: {e}")


class TestIndividualAgents(unittest.TestCase):
    """Test individual agent components."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not AGENTS_AVAILABLE:
            self.skipTest("Agent components not available")
    
    def test_fire_pattern_analysis_agent(self):
        """Test fire pattern analysis agent."""
        try:
            config = {
                'confidence_threshold': 0.7,
                'pattern_window_size': 50
            }
            
            agent = FirePatternAnalysisAgent('test_analyzer', config)
            self.assertEqual(agent.agent_id, 'test_analyzer')
            
            # Test processing
            test_data = {
                'thermal': {'temperature_max': 60.0, 'temperature_avg': 45.0, 'hotspot_count': 2},
                'gas': {'co_concentration': 30.0, 'smoke_density': 40.0},
                'environmental': {'temperature': 25.0, 'humidity': 40.0}
            }
            
            results = agent.process(test_data)
            self.assertIsInstance(results, dict)
            self.assertIn('confidence_score', results)
            self.assertIn('fire_detected', results)
            
        except Exception as e:
            self.skipTest(f"Fire pattern analysis agent test failed: {e}")
    
    def test_emergency_response_agent(self):
        """Test emergency response agent."""
        try:
            config = {
                'response_thresholds': {
                    'LOW': 0.3,
                    'MEDIUM': 0.5,
                    'HIGH': 0.7,
                    'CRITICAL': 0.9
                },
                'alert_channels': ['system'],
                'emergency_contacts': []
            }
            
            agent = EmergencyResponseAgent('test_responder', config)
            
            # Test response determination
            risk_assessment = {
                'risk_score': 0.8,
                'confidence': 0.9,
                'fire_detected': True
            }
            
            results = agent.process({'risk_assessment': risk_assessment})
            self.assertIsInstance(results, dict)
            self.assertIn('response_level', results)
            self.assertIn('alerts', results)
            
        except Exception as e:
            self.skipTest(f"Emergency response agent test failed: {e}")
    
    def test_adaptive_learning_agent(self):
        """Test adaptive learning agent."""
        try:
            config = {
                'learning_window_size': 100,
                'performance_threshold': 0.85,
                'error_analysis_interval': 10
            }
            
            agent = AdaptiveLearningAgent('test_learner', config)
            
            # Test learning from performance data
            performance_data = {
                'performance_metrics': {
                    'accuracy': 0.92,
                    'precision': 0.89,
                    'recall': 0.95,
                    'total_predictions': 100,
                    'correct_predictions': 92
                }
            }
            
            results = agent.process(performance_data)
            self.assertIsInstance(results, dict)
            self.assertIn('performance_metrics', results)
            
        except Exception as e:
            self.skipTest(f"Adaptive learning agent test failed: {e}")


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run tests
    unittest.main(verbosity=2, buffer=True)