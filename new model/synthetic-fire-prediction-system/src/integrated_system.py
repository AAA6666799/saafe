"""
Integrated Fire Detection System - Main System Coordination.

This module provides the main integration layer that coordinates all components:
hardware, agents, ML models, and feature processing for comprehensive fire detection.
"""

import logging
import numpy as np
import time
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
import json

# Import system components
from .hardware.sensor_manager import create_sensor_manager, SensorMode
from .agents.coordination.multi_agent_coordinator import create_multi_agent_fire_system
from .ml.ensemble.model_ensemble_manager import create_fire_prediction_ensemble


class IntegratedFireDetectionSystem:
    """
    Comprehensive integrated fire detection system.
    
    Coordinates all subsystems for complete fire detection and response.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the integrated system."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.system_id = config.get('system_id', 'saafe_fire_detection')
        self.state = 'initializing'
        
        # Core subsystems
        self.sensor_manager = None
        self.multi_agent_system = None
        self.model_ensemble = None
        
        # Processing
        self.processing_pipeline = deque(maxlen=1000)
        self.is_running = False
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Metrics
        self.metrics = {
            'total_processed': 0,
            'fire_detections': 0,
            'average_processing_time': 0.0,
            'system_uptime': 0.0
        }
        
        self.subsystem_health = {
            'sensors': 'unknown',
            'agents': 'unknown', 
            'models': 'unknown'
        }
        
        self.initialization_time = None
    
    def initialize(self) -> bool:
        """Initialize all subsystems."""
        try:
            self.initialization_time = datetime.now()
            self.logger.info("Initializing integrated fire detection system...")
            
            # Initialize sensor management
            sensor_config = self.config.get('sensors', {'mode': SensorMode.SYNTHETIC})
            self.sensor_manager = create_sensor_manager(sensor_config)
            self.sensor_manager.initialize_sensors()
            self.subsystem_health['sensors'] = 'healthy'
            
            # Initialize multi-agent system
            agent_config = self.config.get('agents', {})
            self.multi_agent_system = create_multi_agent_fire_system(agent_config)
            if self.multi_agent_system.initialize():
                self.subsystem_health['agents'] = 'healthy'
            
            # Initialize ML ensemble
            ml_config = self.config.get('machine_learning', {})
            self.model_ensemble = create_fire_prediction_ensemble(ml_config)
            self.model_ensemble.create_default_ensemble()
            self.subsystem_health['models'] = 'healthy'
            
            self.state = 'ready'
            self.logger.info("System initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            self.state = 'error'
            return False
    
    def start(self) -> bool:
        """Start the integrated system."""
        if self.state != 'ready':
            return False
        
        try:
            # Start sensor collection
            if self.sensor_manager:
                self.sensor_manager.start_data_collection()
            
            # Start agent system
            if self.multi_agent_system:
                self.multi_agent_system.start()
            
            # Start processing loop
            self.is_running = True
            self.stop_event.clear()
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.state = 'running'
            self.logger.info("System started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start: {str(e)}")
            return False
    
    def stop(self) -> None:
        """Stop the system."""
        try:
            self.is_running = False
            self.stop_event.set()
            
            if self.processing_thread:
                self.processing_thread.join(timeout=5.0)
            
            if self.sensor_manager:
                self.sensor_manager.stop_data_collection()
            
            if self.multi_agent_system:
                self.multi_agent_system.stop()
            
            self.state = 'stopped'
            self.logger.info("System stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping: {str(e)}")
    
    def process_data(self, sensor_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process sensor data through complete pipeline."""
        processing_start = datetime.now()
        processing_id = f"proc_{len(self.processing_pipeline)}_{processing_start.timestamp()}"
        
        try:
            # Step 1: Get sensor data
            if sensor_data is None:
                if self.sensor_manager:
                    sensor_data = self.sensor_manager.read_all_sensors()
                else:
                    sensor_data = self._get_synthetic_data()
            
            # Step 2: Extract features
            features = self._extract_features(sensor_data)
            
            # Step 3: ML prediction
            if self.model_ensemble and features:
                try:
                    ml_results = self.model_ensemble.predict(features)
                except:
                    ml_results = self._basic_fire_detection(sensor_data)
            else:
                ml_results = self._basic_fire_detection(sensor_data)
            
            # Step 4: Agent analysis
            if self.multi_agent_system:
                try:
                    agent_results = self.multi_agent_system.process_sensor_data(sensor_data)
                except:
                    agent_results = self._basic_response(ml_results)
            else:
                agent_results = self._basic_response(ml_results)
            
            # Step 5: Integration
            final_results = {
                'processing_id': processing_id,
                'timestamp': processing_start.isoformat(),
                'sensor_summary': self._summarize_sensors(sensor_data),
                'ml_results': ml_results,
                'agent_results': agent_results,
                'final_decision': {
                    'fire_detected': ml_results.get('fire_detected', False),
                    'confidence_score': ml_results.get('confidence_score', 0.0),
                    'response_level': agent_results.get('response_level', 'NONE'),
                    'alerts': agent_results.get('alerts', [])
                },
                'processing_time_ms': (datetime.now() - processing_start).total_seconds() * 1000
            }
            
            # Update metrics
            self._update_metrics(final_results, processing_start)
            
            # Store results
            self.processing_pipeline.append(final_results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return {
                'processing_id': processing_id,
                'error': str(e),
                'timestamp': processing_start.isoformat(),
                'final_decision': {'fire_detected': False, 'confidence_score': 0.0}
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_time = datetime.now()
        
        if self.initialization_time:
            uptime = (current_time - self.initialization_time).total_seconds()
        else:
            uptime = 0.0
        
        return {
            'system_id': self.system_id,
            'timestamp': current_time.isoformat(),
            'state': self.state,
            'uptime_seconds': uptime,
            'subsystem_health': self.subsystem_health.copy(),
            'metrics': self.metrics.copy(),
            'recent_activity': self._get_recent_activity(),
            'pipeline_status': {
                'buffer_size': len(self.processing_pipeline),
                'max_size': self.processing_pipeline.maxlen,
                'utilization': len(self.processing_pipeline) / self.processing_pipeline.maxlen
            }
        }
    
    def _processing_loop(self) -> None:
        """Main processing loop."""
        while self.is_running and not self.stop_event.is_set():
            try:
                self.process_data()
                self.stop_event.wait(1.0)  # Process every second
            except Exception as e:
                self.logger.error(f"Processing loop error: {str(e)}")
                self.stop_event.wait(5.0)
    
    def _extract_features(self, sensor_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract basic features from sensor data."""
        features = {}
        
        # Thermal features
        if 'thermal' in sensor_data:
            thermal = sensor_data['thermal']
            for sensor_id, data in thermal.items():
                if isinstance(data, dict):
                    features[f'thermal_{sensor_id}_max'] = data.get('temperature_max', 0.0)
                    features[f'thermal_{sensor_id}_avg'] = data.get('temperature_avg', 0.0)
        
        # Gas features  
        if 'gas' in sensor_data:
            gas = sensor_data['gas']
            for sensor_id, data in gas.items():
                if isinstance(data, dict):
                    features[f'gas_{sensor_id}_co'] = data.get('co_concentration', 0.0)
                    features[f'gas_{sensor_id}_smoke'] = data.get('smoke_density', 0.0)
        
        return features
    
    def _basic_fire_detection(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic threshold-based fire detection."""
        fire_score = 0.0
        indicators = []
        
        # Check thermal data
        if 'thermal' in sensor_data:
            for sensor_id, data in sensor_data['thermal'].items():
                if isinstance(data, dict):
                    max_temp = data.get('temperature_max', 0.0)
                    if max_temp > 40.0:
                        temp_score = min(1.0, (max_temp - 40.0) / 60.0)
                        fire_score = max(fire_score, temp_score)
                        indicators.append(f'thermal_{sensor_id}_high_temp')
        
        # Check gas data
        if 'gas' in sensor_data:
            for sensor_id, data in sensor_data['gas'].items():
                if isinstance(data, dict):
                    co_level = data.get('co_concentration', 0.0)
                    if co_level > 25.0:
                        co_score = min(1.0, (co_level - 25.0) / 75.0)
                        fire_score = max(fire_score, co_score)
                        indicators.append(f'gas_{sensor_id}_high_co')
        
        return {
            'fire_detected': fire_score > 0.5,
            'confidence_score': fire_score,
            'indicators': indicators,
            'method': 'basic_threshold'
        }
    
    def _basic_response(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Basic response logic."""
        fire_detected = ml_results.get('fire_detected', False)
        confidence = ml_results.get('confidence_score', 0.0)
        
        if fire_detected:
            if confidence > 0.8:
                response_level = 'HIGH'
                alert_type = 'CRITICAL'
            else:
                response_level = 'MEDIUM'
                alert_type = 'WARNING'
            
            alerts = [{
                'type': alert_type,
                'message': f'Fire detected (confidence: {confidence:.1%})',
                'timestamp': datetime.now().isoformat()
            }]
        else:
            response_level = 'NONE'
            alerts = []
        
        return {
            'response_level': response_level,
            'alerts': alerts,
            'method': 'basic_response'
        }
    
    def _get_synthetic_data(self) -> Dict[str, Any]:
        """Generate synthetic sensor data."""
        return {
            'thermal': {
                'thermal_01': {
                    'temperature_max': 22.0 + np.random.normal(0, 2),
                    'temperature_avg': 20.0 + np.random.normal(0, 1),
                    'timestamp': datetime.now().isoformat()
                }
            },
            'gas': {
                'gas_01': {
                    'co_concentration': 5.0 + np.random.normal(0, 1),
                    'smoke_density': 10.0 + np.random.normal(0, 2),
                    'timestamp': datetime.now().isoformat()
                }
            },
            'environmental': {
                'env_01': {
                    'temperature': 22.0 + np.random.normal(0, 1),
                    'humidity': 50.0 + np.random.normal(0, 5),
                    'timestamp': datetime.now().isoformat()
                }
            }
        }
    
    def _summarize_sensors(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize sensor data."""
        return {
            'sources': list(sensor_data.keys()),
            'sensor_count': sum(len(v) if isinstance(v, dict) else 1 for v in sensor_data.values()),
            'timestamp': datetime.now().isoformat()
        }
    
    def _update_metrics(self, results: Dict[str, Any], start_time: datetime) -> None:
        """Update system metrics."""
        self.metrics['total_processed'] += 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        if self.metrics['average_processing_time'] == 0:
            self.metrics['average_processing_time'] = processing_time
        else:
            self.metrics['average_processing_time'] = \
                0.9 * self.metrics['average_processing_time'] + 0.1 * processing_time
        
        if results.get('final_decision', {}).get('fire_detected'):
            self.metrics['fire_detections'] += 1
    
    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent system activity."""
        recent = list(self.processing_pipeline)[-5:]  # Last 5 results
        return [{
            'timestamp': r.get('timestamp'),
            'fire_detected': r.get('final_decision', {}).get('fire_detected', False),
            'confidence': r.get('final_decision', {}).get('confidence_score', 0.0)
        } for r in recent]


# Convenience function for creating integrated system
def create_integrated_fire_system(config: Optional[Dict[str, Any]] = None) -> IntegratedFireDetectionSystem:
    """Create integrated fire detection system with default config."""
    default_config = {
        'system_id': 'saafe_integrated_fire_detection',
        'sensors': {'mode': SensorMode.SYNTHETIC},
        'agents': {
            'analysis': {'fire_pattern': {'confidence_threshold': 0.7}},
            'response': {'emergency': {'response_thresholds': {'HIGH': 0.7}}},
            'learning': {'adaptive': {'learning_window_size': 100}}
        },
        'machine_learning': {'ensemble_strategy': 'weighted_voting'}
    }
    
    if config:
        default_config.update(config)
    
    return IntegratedFireDetectionSystem(default_config)