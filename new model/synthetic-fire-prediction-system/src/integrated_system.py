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
import pandas as pd

# Import system components
from .hardware.sensor_manager import create_sensor_manager, SensorMode
from .agents.coordination.multi_agent_coordinator import create_multi_agent_fire_system
from .ml.ensemble.model_ensemble_manager import create_fire_prediction_ensemble


class IntegratedFireDetectionSystem:
    """
    Integrated fire detection system for FLIR Lepton 3.5 + SCD41 sensors.
    
    This system coordinates FLIR thermal imaging and SCD41 CO₂ sensors
    with 18 total features (15 thermal + 3 gas) for fire detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the FLIR + SCD41 integrated system."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.system_id = config.get('system_id', 'flir_scd41_fire_detection')
        self.state = 'initializing'
        
        # Sensor specifications
        self.sensor_specs = {
            'flir_lepton35': {
                'type': 'thermal',
                'resolution': (120, 160),
                'features': [
                    't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
                    't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
                    't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
                    'tproxy_val', 'tproxy_delta', 'tproxy_vel'
                ]
            },
            'scd41_co2': {
                'type': 'gas',
                'measurement_range': (400, 40000),
                'features': ['gas_val', 'gas_delta', 'gas_vel']
            }
        }
        
        # Feature extractors
        self.thermal_extractor = None
        self.gas_extractor = None
        
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
            'false_positives': 0,
            'average_processing_time': 0.0,
            'system_uptime': 0.0,
            'flir_frames_processed': 0,
            'scd41_readings_processed': 0
        }
        
        self.subsystem_health = {
            'flir_sensor': 'unknown',
            'scd41_sensor': 'unknown',
            'thermal_extractor': 'unknown',
            'gas_extractor': 'unknown',
            'agents': 'unknown', 
            'models': 'unknown'
        }
        
        self.initialization_time = None
    
    def initialize(self) -> bool:
        """Initialize FLIR + SCD41 subsystems."""
        try:
            self.initialization_time = datetime.now()
            self.logger.info("Initializing FLIR Lepton 3.5 + SCD41 fire detection system...")
            
            # Initialize FLIR Lepton 3.5 thermal feature extractor
            try:
                from .feature_engineering.extractors.flir_thermal_extractor import FlirThermalExtractor
                thermal_config = self.config.get('thermal_extractor', {
                    'hot_temperature_threshold': 50.0,
                    'gradient_kernel_size': 3,
                    'percentile_threshold': 95,
                    'flow_history_length': 3,
                    'temperature_proxy_alpha': 0.8
                })
                self.thermal_extractor = FlirThermalExtractor(thermal_config)
                self.subsystem_health['thermal_extractor'] = 'healthy'
                self.logger.info("FLIR Lepton 3.5 thermal extractor initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize thermal extractor: {e}")
                self.subsystem_health['thermal_extractor'] = 'error'
            
            # Initialize SCD41 CO₂ gas feature extractor
            try:
                from .feature_engineering.extractors.scd41_gas_extractor import Scd41GasExtractor
                gas_config = self.config.get('gas_extractor', {
                    'co2_smoothing_alpha': 0.8,
                    'co2_normal_range': [400, 1000],
                    'velocity_history_length': 5,
                    'delta_threshold': 50.0
                })
                self.gas_extractor = Scd41GasExtractor(gas_config)
                self.subsystem_health['gas_extractor'] = 'healthy'
                self.logger.info("SCD41 CO₂ gas extractor initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize gas extractor: {e}")
                self.subsystem_health['gas_extractor'] = 'error'
            
            # Initialize sensor management for FLIR + SCD41
            sensor_config = self.config.get('sensors', {
                'mode': SensorMode.SYNTHETIC,
                'flir_config': {
                    'resolution': (120, 160),
                    'frame_rate': 9.0,
                    'temperature_range': (-10, 400)
                },
                'scd41_config': {
                    'measurement_range': (400, 40000),
                    'sampling_rate': 0.2  # Every 5 seconds
                }
            })
            self.sensor_manager = create_sensor_manager(sensor_config)
            self.sensor_manager.initialize_sensors()
            self.subsystem_health['flir_sensor'] = 'healthy'
            self.subsystem_health['scd41_sensor'] = 'healthy'
            
            # Initialize multi-agent system for FLIR + SCD41
            agent_config = self.config.get('agents', {
                'thermal_analysis_enabled': True,
                'gas_analysis_enabled': True,
                'feature_count': 18,  # 15 thermal + 3 gas
                'thermal_thresholds': {
                    't_max': 60.0,
                    't_hot_area_pct': 5.0,
                    'tproxy_delta': 10.0
                },
                'gas_thresholds': {
                    'gas_val': 2000.0,  # ppm CO₂
                    'gas_delta': 100.0,  # ppm change
                    'gas_vel': 50.0     # ppm/s rate
                },
                'analysis': {
                    'fire_pattern': {
                        'confidence_threshold': 0.7,
                        'pattern_window_size': 50,
                        'fire_signatures': {}
                    }
                },
                'monitoring': {
                    'system_health': {}
                },
                'response': {
                    'emergency': {}
                },
                'learning': {
                    'adaptive': {}
                }
            })
            self.multi_agent_system = create_multi_agent_fire_system(agent_config)
            if self.multi_agent_system.initialize():
                self.subsystem_health['agents'] = 'healthy'
            
            # Initialize ML ensemble for 18-feature input
            ml_config = self.config.get('machine_learning', {
                'input_features': 18,
                'thermal_feature_count': 15,
                'gas_feature_count': 3,
                'model_types': ['random_forest', 'xgboost', 'neural_network'],
                'ensemble_method': 'voting'
            })
            self.model_ensemble = create_fire_prediction_ensemble(ml_config)
            self.model_ensemble.create_default_ensemble()
            self.subsystem_health['models'] = 'healthy'
            
            self.state = 'ready'
            self.logger.info("FLIR + SCD41 system initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"FLIR + SCD41 initialization failed: {str(e)}")
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
                    # Convert features to DataFrames for thermal and gas separately
                    thermal_feature_names = self.sensor_specs['flir_lepton35']['features']
                    gas_feature_names = self.sensor_specs['scd41_co2']['features']
                    
                    # Extract thermal features (15 features)
                    thermal_features_dict = {name: [features.get(name, 0.0)] for name in thermal_feature_names}
                    thermal_df = pd.DataFrame(thermal_features_dict)
                    
                    # Extract gas features (3 features)
                    gas_features_dict = {name: [features.get(name, 0.0)] for name in gas_feature_names}
                    gas_df = pd.DataFrame(gas_features_dict)
                    
                    # Use the specialized FLIR+SCD41 prediction method
                    ml_prediction = self.model_ensemble.predict_flir_scd41(thermal_df, gas_df)
                    ml_results = ml_prediction
                    
                except Exception as e:
                    self.logger.warning(f"ML prediction failed: {e}")
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
                'max_size': self.processing_pipeline.maxlen or 1000,
                'utilization': len(self.processing_pipeline) / (self.processing_pipeline.maxlen or 1000)
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
        """Extract 18 features from FLIR + SCD41 sensor data."""
        features = {}
        
        try:
            # Extract 15 FLIR Lepton 3.5 thermal features
            if 'flir' in sensor_data or 'thermal' in sensor_data:
                thermal_data = sensor_data.get('flir') or sensor_data.get('thermal')
                
                if self.thermal_extractor and thermal_data:
                    try:
                        thermal_features = self.thermal_extractor.extract_features(thermal_data)
                        
                        # Add all 15 FLIR features
                        for feature_name in self.sensor_specs['flir_lepton35']['features']:
                            features[feature_name] = thermal_features.get(feature_name, 0.0)
                        
                        self.metrics['flir_frames_processed'] += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Thermal feature extraction failed: {e}")
                        # Fill with default values if extraction fails
                        for feature_name in self.sensor_specs['flir_lepton35']['features']:
                            features[feature_name] = 0.0
                else:
                    # Extract basic thermal features if extractor not available
                    if isinstance(thermal_data, dict):
                        # Handle dictionary format
                        for sensor_id, data in thermal_data.items():
                            if isinstance(data, dict):
                                features['t_max'] = data.get('temperature_max', 0.0)
                                features['t_mean'] = data.get('temperature_avg', 0.0)
                                features['t_std'] = data.get('temperature_std', 0.0)
                            elif hasattr(data, 'shape'):  # numpy array (thermal image)
                                features['t_max'] = float(np.max(data))
                                features['t_mean'] = float(np.mean(data))
                                features['t_std'] = float(np.std(data))
                    
                    # Fill remaining thermal features with defaults
                    thermal_defaults = {
                        't_p95': features.get('t_max', 0.0) * 0.9,
                        't_hot_area_pct': 0.0,
                        't_hot_largest_blob_pct': 0.0,
                        't_grad_mean': 0.0,
                        't_grad_std': 0.0,
                        't_diff_mean': 0.0,
                        't_diff_std': 0.0,
                        'flow_mag_mean': 0.0,
                        'flow_mag_std': 0.0,
                        'tproxy_val': features.get('t_mean', 0.0),
                        'tproxy_delta': 0.0,
                        'tproxy_vel': 0.0
                    }
                    
                    for name, default_val in thermal_defaults.items():
                        if name not in features:
                            features[name] = default_val
            
            # Extract 3 SCD41 CO₂ gas features
            if 'scd41' in sensor_data or 'gas' in sensor_data:
                gas_data = sensor_data.get('scd41') or sensor_data.get('gas')
                
                if self.gas_extractor and gas_data:
                    try:
                        gas_features = self.gas_extractor.extract_features(gas_data)
                        
                        # Add all 3 SCD41 features
                        for feature_name in self.sensor_specs['scd41_co2']['features']:
                            features[feature_name] = gas_features.get(feature_name, 0.0)
                        
                        self.metrics['scd41_readings_processed'] += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Gas feature extraction failed: {e}")
                        # Fill with default values if extraction fails
                        features['gas_val'] = 400.0  # Normal CO₂ baseline
                        features['gas_delta'] = 0.0
                        features['gas_vel'] = 0.0
                else:
                    # Extract basic gas features if extractor not available
                    if isinstance(gas_data, dict):
                        for sensor_id, data in gas_data.items():
                            if isinstance(data, dict):
                                co2_concentration = data.get('co2_concentration', 400.0)
                                features['gas_val'] = co2_concentration
                                features['gas_delta'] = data.get('co2_delta', 0.0)
                                features['gas_vel'] = data.get('co2_velocity', 0.0)
                    else:
                        # Default gas features
                        features['gas_val'] = 400.0
                        features['gas_delta'] = 0.0
                        features['gas_vel'] = 0.0
            
            # Ensure we have exactly 18 features
            expected_features = (self.sensor_specs['flir_lepton35']['features'] + 
                               self.sensor_specs['scd41_co2']['features'])
            
            for feature_name in expected_features:
                if feature_name not in features:
                    features[feature_name] = 0.0
            
            self.logger.debug(f"Extracted {len(features)} features: {list(features.keys())}")
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            # Return default feature set if extraction completely fails
            features = {name: 0.0 for name in 
                       (self.sensor_specs['flir_lepton35']['features'] + 
                        self.sensor_specs['scd41_co2']['features'])}
        
        return features
    
    def _basic_fire_detection(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """FLIR + SCD41 threshold-based fire detection."""
        fire_score = 0.0
        indicators = []
        feature_scores = {}
        
        # Extract features for analysis
        features = self._extract_features(sensor_data)
        
        # FLIR Lepton 3.5 thermal analysis
        thermal_score = 0.0
        
        # Temperature thresholds
        if features.get('t_max', 0) > 60.0:  # High temperature threshold
            temp_score = min(1.0, (features['t_max'] - 60.0) / 100.0)
            thermal_score = max(thermal_score, temp_score)
            indicators.append('flir_high_temperature')
            feature_scores['thermal_max'] = temp_score
        
        if features.get('t_mean', 0) > 35.0:  # Elevated mean temperature
            mean_score = min(1.0, (features['t_mean'] - 35.0) / 50.0)
            thermal_score = max(thermal_score, mean_score * 0.7)
            indicators.append('flir_elevated_mean_temp')
            feature_scores['thermal_mean'] = mean_score
        
        # Hot area analysis
        if features.get('t_hot_area_pct', 0) > 5.0:  # >5% hot area
            hot_area_score = min(1.0, features['t_hot_area_pct'] / 20.0)
            thermal_score = max(thermal_score, hot_area_score)
            indicators.append('flir_significant_hot_area')
            feature_scores['hot_area'] = hot_area_score
        
        # Temperature gradient (indicates rapid heating)
        if features.get('t_grad_mean', 0) > 5.0:
            grad_score = min(1.0, features['t_grad_mean'] / 20.0)
            thermal_score = max(thermal_score, grad_score * 0.6)
            indicators.append('flir_high_gradient')
            feature_scores['gradient'] = grad_score
        
        # Temporal changes
        if features.get('tproxy_delta', 0) > 10.0:  # Rapid temperature rise
            delta_score = min(1.0, features['tproxy_delta'] / 30.0)
            thermal_score = max(thermal_score, delta_score)
            indicators.append('flir_rapid_temperature_rise')
            feature_scores['temperature_delta'] = delta_score
        
        # SCD41 CO₂ analysis
        gas_score = 0.0
        
        # CO₂ concentration thresholds
        co2_level = features.get('gas_val', 400)
        if co2_level > 2000:  # Elevated CO₂ (normal indoor ~400-1000 ppm)
            co2_score = min(1.0, (co2_level - 2000) / 8000.0)  # Scale to 10000 ppm max
            gas_score = max(gas_score, co2_score)
            indicators.append('scd41_elevated_co2')
            feature_scores['co2_level'] = co2_score
        
        # CO₂ rate of change
        co2_delta = features.get('gas_delta', 0)
        if co2_delta > 100:  # Rapid CO₂ increase
            delta_score = min(1.0, co2_delta / 500.0)
            gas_score = max(gas_score, delta_score)
            indicators.append('scd41_rapid_co2_increase')
            feature_scores['co2_delta'] = delta_score
        
        # CO₂ velocity (acceleration)
        co2_vel = features.get('gas_vel', 0)
        if co2_vel > 50:  # Accelerating CO₂ increase
            vel_score = min(1.0, co2_vel / 200.0)
            gas_score = max(gas_score, vel_score * 0.8)
            indicators.append('scd41_accelerating_co2')
            feature_scores['co2_velocity'] = vel_score
        
        # Combined fire score with sensor fusion
        if thermal_score > 0 and gas_score > 0:
            # Both sensors indicate fire - high confidence
            fire_score = min(1.0, (thermal_score + gas_score) * 0.8)
            indicators.append('multi_sensor_correlation')
        elif thermal_score > 0.7:
            # Strong thermal signal alone
            fire_score = thermal_score * 0.9
        elif gas_score > 0.7:
            # Strong gas signal alone (could be smoldering)
            fire_score = gas_score * 0.7
        else:
            # Weak signals - take maximum
            fire_score = max(thermal_score, gas_score)
        
        # Fire classification based on sensor patterns
        fire_type = 'none'
        if fire_score > 0.5:
            if thermal_score > gas_score:
                if features.get('t_grad_mean', 0) > 10:
                    fire_type = 'rapid_combustion'
                else:
                    fire_type = 'thermal_fire'
            elif gas_score > thermal_score:
                fire_type = 'smoldering_fire'
            else:
                fire_type = 'developing_fire'
        
        return {
            'fire_detected': fire_score > 0.5,
            'confidence_score': fire_score,
            'fire_type': fire_type,
            'indicators': indicators,
            'feature_scores': feature_scores,
            'sensor_scores': {
                'thermal_score': thermal_score,
                'gas_score': gas_score
            },
            'sensor_data_summary': {
                'flir_max_temp': features.get('t_max', 0),
                'flir_mean_temp': features.get('t_mean', 0),
                'flir_hot_area_pct': features.get('t_hot_area_pct', 0),
                'scd41_co2_ppm': features.get('gas_val', 400),
                'scd41_co2_delta': features.get('gas_delta', 0)
            },
            'method': 'flir_scd41_threshold'
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
        """Generate synthetic FLIR + SCD41 sensor data."""
        current_time = datetime.now()
        
        # Generate FLIR Lepton 3.5 thermal data
        flir_resolution = self.sensor_specs['flir_lepton35']['resolution']
        thermal_frame = np.random.normal(22.0, 2.0, flir_resolution)  # Base room temperature
        
        # Add random hotspot for testing
        if np.random.random() < 0.1:  # 10% chance of hotspot
            center_y = np.random.randint(20, flir_resolution[0] - 20)
            center_x = np.random.randint(20, flir_resolution[1] - 20)
            radius = np.random.randint(5, 15)
            hotspot_temp = np.random.uniform(40, 80)
            
            y, x = np.ogrid[:flir_resolution[0], :flir_resolution[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            thermal_frame[mask] = hotspot_temp
        
        # Generate SCD41 CO₂ data
        base_co2 = np.random.normal(600, 100)  # Typical indoor CO₂
        co2_variation = np.random.normal(0, 20)
        
        # Occasionally simulate elevated CO₂
        if np.random.random() < 0.05:  # 5% chance of elevated CO₂
            co2_variation += np.random.uniform(500, 2000)
        
        current_co2 = max(400, base_co2 + co2_variation)  # Minimum outdoor level
        
        return {
            'flir': {
                'flir_lepton35': {
                    'thermal_frame': thermal_frame,
                    'timestamp': current_time.isoformat(),
                    'frame_rate': 9.0,
                    'resolution': flir_resolution,
                    'temperature_stats': {
                        'min': float(np.min(thermal_frame)),
                        'max': float(np.max(thermal_frame)),
                        'mean': float(np.mean(thermal_frame)),
                        'std': float(np.std(thermal_frame))
                    }
                }
            },
            'scd41': {
                'scd41_co2': {
                    'co2_concentration': current_co2,
                    'gas_reading': current_co2,
                    'value': current_co2,
                    'timestamp': current_time.isoformat(),
                    'sensor_temp': 20.0 + np.random.normal(0, 2),
                    'sensor_humidity': 45.0 + np.random.normal(0, 5),
                    'measurement_range': self.sensor_specs['scd41_co2']['measurement_range']
                }
            },
            'timestamp': current_time.isoformat(),
            'system_id': self.system_id
        }
    
    def _summarize_sensors(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize FLIR + SCD41 sensor data."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'sensor_types': [],
            'sensor_count': 0,
            'data_quality': 'good'
        }
        
        # FLIR Lepton 3.5 summary
        if 'flir' in sensor_data:
            flir_data = sensor_data['flir']
            summary['sensor_types'].append('flir_lepton35')
            summary['sensor_count'] += len(flir_data)
            
            for sensor_id, data in flir_data.items():
                if 'thermal_frame' in data:
                    frame = data['thermal_frame']
                    summary[f'{sensor_id}_status'] = {
                        'type': 'thermal_imaging',
                        'resolution': frame.shape if hasattr(frame, 'shape') else 'unknown',
                        'temperature_range': {
                            'min': float(np.min(frame)) if hasattr(frame, 'shape') else 0,
                            'max': float(np.max(frame)) if hasattr(frame, 'shape') else 0
                        },
                        'data_quality': 'good' if hasattr(frame, 'shape') else 'poor'
                    }
        
        # SCD41 CO₂ summary
        if 'scd41' in sensor_data:
            scd41_data = sensor_data['scd41']
            summary['sensor_types'].append('scd41_co2')
            summary['sensor_count'] += len(scd41_data)
            
            for sensor_id, data in scd41_data.items():
                co2_reading = data.get('co2_concentration', 0)
                min_range, max_range = self.sensor_specs['scd41_co2']['measurement_range']
                
                summary[f'{sensor_id}_status'] = {
                    'type': 'co2_sensor',
                    'co2_ppm': co2_reading,
                    'in_range': min_range <= co2_reading <= max_range,
                    'sensor_temp': data.get('sensor_temp', 20.0),
                    'data_quality': 'good' if min_range <= co2_reading <= max_range else 'questionable'
                }
        
        # Handle legacy format for backward compatibility
        if 'thermal' in sensor_data:
            summary['sensor_types'].append('thermal_legacy')
            summary['sensor_count'] += len(sensor_data['thermal'])
        
        if 'gas' in sensor_data:
            summary['sensor_types'].append('gas_legacy')
            summary['sensor_count'] += len(sensor_data['gas'])
        
        # Overall data quality assessment
        if summary['sensor_count'] == 0:
            summary['data_quality'] = 'no_data'
        elif len(summary['sensor_types']) < 2:
            summary['data_quality'] = 'partial'
        
        return summary
    
    def _update_metrics(self, results: Dict[str, Any], start_time: datetime) -> None:
        """Update FLIR + SCD41 system metrics."""
        self.metrics['total_processed'] += 1
        
        # Processing time metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        if self.metrics['average_processing_time'] == 0:
            self.metrics['average_processing_time'] = processing_time
        else:
            self.metrics['average_processing_time'] = \
                0.9 * self.metrics['average_processing_time'] + 0.1 * processing_time
        
        # Fire detection metrics
        final_decision = results.get('final_decision', {})
        if final_decision.get('fire_detected'):
            self.metrics['fire_detections'] += 1
        
        # Sensor-specific metrics
        ml_results = results.get('ml_results', {})
        sensor_scores = ml_results.get('sensor_scores', {})
        
        # Track thermal sensor performance
        thermal_score = sensor_scores.get('thermal_score', 0)
        if not hasattr(self, '_thermal_score_history'):
            self._thermal_score_history = []
        self._thermal_score_history.append(thermal_score)
        if len(self._thermal_score_history) > 100:
            self._thermal_score_history = self._thermal_score_history[-100:]
        
        # Track gas sensor performance
        gas_score = sensor_scores.get('gas_score', 0)
        if not hasattr(self, '_gas_score_history'):
            self._gas_score_history = []
        self._gas_score_history.append(gas_score)
        if len(self._gas_score_history) > 100:
            self._gas_score_history = self._gas_score_history[-100:]
        
        # Calculate sensor correlation
        if hasattr(self, '_thermal_score_history') and hasattr(self, '_gas_score_history'):
            if len(self._thermal_score_history) >= 10:
                thermal_array = np.array(self._thermal_score_history[-10:])
                gas_array = np.array(self._gas_score_history[-10:])
                correlation = np.corrcoef(thermal_array, gas_array)[0, 1]
                self.metrics['sensor_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
        
        # Feature extraction success rates
        if 'flir_frames_processed' in self.metrics:
            flir_success_rate = self.metrics['flir_frames_processed'] / max(1, self.metrics['total_processed'])
            self.metrics['flir_processing_success_rate'] = flir_success_rate
        
        if 'scd41_readings_processed' in self.metrics:
            scd41_success_rate = self.metrics['scd41_readings_processed'] / max(1, self.metrics['total_processed'])
            self.metrics['scd41_processing_success_rate'] = scd41_success_rate
        
        # False positive tracking (basic heuristic)
        confidence = final_decision.get('confidence_score', 0)
        fire_type = ml_results.get('fire_type', 'none')
        
        # Heuristic: very low confidence detections might be false positives
        if final_decision.get('fire_detected') and confidence < 0.6:
            self.metrics['false_positives'] = self.metrics.get('false_positives', 0) + 1
        
        # Update system uptime
        if self.initialization_time:
            self.metrics['system_uptime'] = (datetime.now() - self.initialization_time).total_seconds()
    
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