"""
Multi-Agent Coordinator for the synthetic fire prediction system.

This module orchestrates the interaction between monitoring, analysis, response,
and learning agents to provide comprehensive fire detection and response capabilities.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import json
import uuid

from ..base import AgentCoordinator, Message
from ..monitoring.system_health import SystemHealthMonitor
from ..analysis.fire_pattern_analysis import FirePatternAnalysisAgent
from ..response.emergency_response import EmergencyResponseAgent
from ..learning.adaptive_learning import AdaptiveLearningAgent


class MultiAgentFireDetectionSystem:
    """
    Comprehensive multi-agent system for fire detection and response.
    
    This system coordinates multiple specialized agents to provide:
    - Real-time monitoring and anomaly detection
    - Advanced pattern analysis and fire detection
    - Emergency response coordination
    - Continuous learning and system improvement
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the multi-agent fire detection system.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # System identification
        self.system_id = config.get('system_id', f"fire_detection_system_{uuid.uuid4().hex[:8]}")
        self.initialized = False
        self.running = False
        
        # Agent coordinator
        self.coordinator = AgentCoordinator(config.get('coordinator_config', {}))
        
        # Initialize agents
        self.agents = {}
        self.agent_types = {
            'monitoring': [],
            'analysis': [],
            'response': [],
            'learning': []
        }
        
        # System state and metrics
        self.system_state = {
            'status': 'initializing',
            'last_update': datetime.now().isoformat(),
            'active_agents': 0,
            'message_queue_size': 0,
            'alerts_active': 0,
            'fire_detected': False,
            'confidence_score': 0.0,
            'response_level': 0
        }
        
        # Performance tracking
        self.system_metrics = {
            'total_detections': 0,
            'true_positives': 0,
            'false_positives': 0,
            'system_uptime': 0.0,
            'average_response_time': 0.0,
            'agent_performance': defaultdict(dict)
        }
        
        # Data processing pipeline
        self.data_pipeline = deque(maxlen=1000)
        self.processing_queue = asyncio.Queue() if hasattr(asyncio, 'Queue') else []
        
        # Threading for concurrent processing
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        self.logger.info(f"Initializing Multi-Agent Fire Detection System: {self.system_id}")
    
    def initialize(self) -> bool:
        """
        Initialize all system components and agents.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Starting system initialization...")
            
            # Initialize monitoring agents
            self._initialize_monitoring_agents()
            
            # Initialize analysis agents
            self._initialize_analysis_agents()
            
            # Initialize response agents
            self._initialize_response_agents()
            
            # Initialize learning agents
            self._initialize_learning_agents()
            
            # Setup agent communication and workflows
            self._setup_agent_workflows()
            
            # Initialize system state
            self._initialize_system_state()
            
            self.initialized = True
            self.system_state['status'] = 'initialized'
            self.system_state['active_agents'] = len(self.agents)
            
            self.logger.info(f"System initialization complete. Active agents: {len(self.agents)}")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {str(e)}")
            self.system_state['status'] = 'initialization_failed'
            return False
    
    def start(self) -> bool:
        """
        Start the multi-agent fire detection system.
        
        Returns:
            True if system started successfully, False otherwise
        """
        if not self.initialized:
            self.logger.error("System not initialized. Call initialize() first.")
            return False
        
        try:
            self.logger.info("Starting fire detection system...")
            
            # Start processing thread
            self.running = True
            self.stop_event.clear()
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.system_state['status'] = 'running'
            self.system_state['last_update'] = datetime.now().isoformat()
            
            self.logger.info("Fire detection system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {str(e)}")
            self.system_state['status'] = 'start_failed'
            return False
    
    def stop(self) -> None:
        """Stop the multi-agent fire detection system."""
        try:
            self.logger.info("Stopping fire detection system...")
            
            self.running = False
            self.stop_event.set()
            
            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            self.system_state['status'] = 'stopped'
            self.system_state['last_update'] = datetime.now().isoformat()
            
            self.logger.info("Fire detection system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {str(e)}")
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming sensor data through the multi-agent pipeline.
        
        Args:
            sensor_data: Raw sensor data from thermal, gas, and environmental sensors
            
        Returns:
            Comprehensive fire detection results
        """
        processing_start = datetime.now()
        processing_id = f"processing_{len(self.data_pipeline)}_{processing_start.timestamp()}"
        
        try:
            self.logger.debug(f"Processing sensor data: {processing_id}")
            
            # Stage 1: Monitoring and Health Check
            monitoring_results = self._run_monitoring_stage(sensor_data, processing_id)
            
            # Stage 2: Fire Pattern Analysis
            analysis_results = self._run_analysis_stage(sensor_data, monitoring_results, processing_id)
            
            # Stage 3: Risk Assessment and Response
            response_results = self._run_response_stage(analysis_results, processing_id)
            
            # Stage 4: Learning and Improvement
            learning_results = self._run_learning_stage(
                sensor_data, analysis_results, response_results, processing_id
            )
            
            # Compile comprehensive results
            processing_results = {
                'processing_id': processing_id,
                'timestamp': processing_start.isoformat(),
                'system_id': self.system_id,
                'sensor_data_summary': self._summarize_sensor_data(sensor_data),
                'monitoring_results': monitoring_results,
                'analysis_results': analysis_results,
                'response_results': response_results,
                'learning_results': learning_results,
                'fire_detection': {
                    'fire_detected': analysis_results.get('fire_detected', False),
                    'confidence_score': analysis_results.get('confidence_score', 0.0),
                    'response_level': response_results.get('response_level_value', 0),
                    'alerts_generated': len(response_results.get('alerts', [])),
                    'recommendations': response_results.get('recommendations', [])
                },
                'system_status': {
                    'processing_time_ms': (datetime.now() - processing_start).total_seconds() * 1000,
                    'agents_involved': len([r for r in [monitoring_results, analysis_results, response_results, learning_results] if r]),
                    'data_quality': monitoring_results.get('data_quality', 'unknown'),
                    'system_health': monitoring_results.get('system_health', 'unknown')
                }
            }
            
            # Update system state
            self._update_system_state(processing_results)
            
            # Store in processing pipeline
            self.data_pipeline.append(processing_results)
            
            self.logger.debug(f"Processing complete: {processing_id}, fire_detected={processing_results['fire_detection']['fire_detected']}")
            
            return processing_results
            
        except Exception as e:
            self.logger.error(f"Error processing sensor data: {str(e)}")
            return {
                'processing_id': processing_id,
                'timestamp': processing_start.isoformat(),
                'error': str(e),
                'fire_detection': {
                    'fire_detected': False,
                    'confidence_score': 0.0,
                    'response_level': 0,
                    'alerts_generated': 0
                }
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and metrics."""
        current_time = datetime.now()
        
        # Calculate system uptime
        if self.running:
            # In a real system, would track actual start time
            uptime_hours = 1.0  # Placeholder
        else:
            uptime_hours = 0.0
        
        return {
            'system_id': self.system_id,
            'timestamp': current_time.isoformat(),
            'system_state': self.system_state.copy(),
            'system_metrics': {
                **self.system_metrics,
                'system_uptime_hours': uptime_hours,
                'data_points_processed': len(self.data_pipeline),
                'average_processing_time': self._calculate_average_processing_time(),
                'recent_accuracy': self._calculate_recent_accuracy()
            },
            'agent_status': {agent_id: {
                'type': self._get_agent_type(agent_id),
                'active': True,  # In a real system, would check actual agent health
                'last_activity': current_time.isoformat()
            } for agent_id in self.agents.keys()},
            'active_alerts': self._get_active_alerts(),
            'recent_detections': self._get_recent_detections(),
            'performance_summary': self._generate_performance_summary()
        }
    
    def _initialize_monitoring_agents(self) -> None:
        """Initialize monitoring agents."""
        # System health monitor
        health_config = self.config.get('monitoring', {}).get('system_health', {})
        health_monitor = SystemHealthMonitor('system_health_monitor', health_config)
        self.agents['system_health_monitor'] = health_monitor
        self.agent_types['monitoring'].append('system_health_monitor')
        self.coordinator.register_agent(health_monitor)
        
        self.logger.info("Monitoring agents initialized")
    
    def _initialize_analysis_agents(self) -> None:
        """Initialize analysis agents."""
        # Fire pattern analysis agent
        analysis_config = self.config.get('analysis', {}).get('fire_pattern', {
            'confidence_threshold': 0.7,
            'pattern_window_size': 50,
            'fire_signatures': {}
        })
        pattern_analyzer = FirePatternAnalysisAgent('fire_pattern_analyzer', analysis_config)
        self.agents['fire_pattern_analyzer'] = pattern_analyzer
        self.agent_types['analysis'].append('fire_pattern_analyzer')
        self.coordinator.register_agent(pattern_analyzer)
        
        self.logger.info("Analysis agents initialized")
    
    def _initialize_response_agents(self) -> None:
        """Initialize response agents."""
        # Emergency response agent
        response_config = self.config.get('response', {}).get('emergency', {})
        emergency_responder = EmergencyResponseAgent('emergency_responder', response_config)
        self.agents['emergency_responder'] = emergency_responder
        self.agent_types['response'].append('emergency_responder')
        self.coordinator.register_agent(emergency_responder)
        
        self.logger.info("Response agents initialized")
    
    def _initialize_learning_agents(self) -> None:
        """Initialize learning agents."""
        # Adaptive learning agent
        learning_config = self.config.get('learning', {}).get('adaptive', {})
        adaptive_learner = AdaptiveLearningAgent('adaptive_learner', learning_config)
        self.agents['adaptive_learner'] = adaptive_learner
        self.agent_types['learning'].append('adaptive_learner')
        self.coordinator.register_agent(adaptive_learner)
        
        self.logger.info("Learning agents initialized")
    
    def _setup_agent_workflows(self) -> None:
        """Setup communication workflows between agents."""
        # Setup message handlers for agent coordination
        
        # Analysis results to response agent
        self.agents['fire_pattern_analyzer'].register_message_handler(
            'analysis_complete',
            lambda msg: self._forward_to_response(msg)
        )
        
        # Response results to learning agent
        self.agents['emergency_responder'].register_message_handler(
            'response_complete',
            lambda msg: self._forward_to_learning(msg)
        )
        
        self.logger.info("Agent workflows configured")
    
    def _initialize_system_state(self) -> None:
        """Initialize system state tracking."""
        self.system_state.update({
            'initialization_time': datetime.now().isoformat(),
            'active_agents': len(self.agents),
            'message_queue_size': 0,
            'status': 'initialized'
        })
    
    def _processing_loop(self) -> None:
        """Main processing loop for the system."""
        self.logger.info("Starting processing loop")
        
        while self.running and not self.stop_event.is_set():
            try:
                # Process agent messages
                self.coordinator.process_messages()
                
                # Update system metrics
                self._update_system_metrics()
                
                # Check for system health issues
                self._check_system_health()
                
                # Small delay to prevent excessive CPU usage
                self.stop_event.wait(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}")
                self.stop_event.wait(1.0)  # Wait longer on error
        
        self.logger.info("Processing loop stopped")
    
    def _run_monitoring_stage(self, sensor_data: Dict[str, Any], processing_id: str) -> Dict[str, Any]:
        """Run monitoring stage of the pipeline."""
        if 'system_health_monitor' in self.agents:
            try:
                monitor = self.agents['system_health_monitor']
                monitoring_result = monitor.process(sensor_data)
                return monitoring_result
            except Exception as e:
                self.logger.error(f"Monitoring stage error: {str(e)}")
        
        return {'status': 'monitoring_unavailable', 'data_quality': 'unknown'}
    
    def _run_analysis_stage(self, sensor_data: Dict[str, Any], monitoring_results: Dict[str, Any], processing_id: str) -> Dict[str, Any]:
        """Run analysis stage of the pipeline."""
        if 'fire_pattern_analyzer' in self.agents:
            try:
                analyzer = self.agents['fire_pattern_analyzer']
                
                # Prepare analysis input with FLIR+SCD41 specific formatting
                analysis_input = {
                    **sensor_data,
                    'monitoring_results': monitoring_results,
                    'processing_id': processing_id
                }
                
                # Extract FLIR thermal features if present
                if 'flir' in sensor_data and 'flir_lepton35' in sensor_data['flir']:
                    analysis_input['thermal_features'] = sensor_data['flir']['flir_lepton35']
                
                # Extract SCD41 gas features if present
                if 'scd41' in sensor_data and 'scd41_co2' in sensor_data['scd41']:
                    analysis_input['gas_features'] = sensor_data['scd41']['scd41_co2']
                
                analysis_result = analyzer.process(analysis_input)
                return analysis_result
            except Exception as e:
                self.logger.error(f"Analysis stage error: {str(e)}")
        
        return {'status': 'analysis_unavailable', 'fire_detected': False, 'confidence_score': 0.0}
    
    def _run_response_stage(self, analysis_results: Dict[str, Any], processing_id: str) -> Dict[str, Any]:
        """Run response stage of the pipeline."""
        if 'emergency_responder' in self.agents:
            try:
                responder = self.agents['emergency_responder']
                
                # Prepare response input
                response_input = {
                    'risk_assessment': {
                        'risk_score': analysis_results.get('confidence_score', 0.0),
                        'confidence': analysis_results.get('confidence_score', 0.0),
                        'fire_detected': analysis_results.get('fire_detected', False)
                    },
                    'analysis_results': analysis_results,
                    'processing_id': processing_id
                }
                
                # Add FLIR+SCD41 specific data if available in analysis results
                if 'analysis_components' in analysis_results:
                    components = analysis_results['analysis_components']
                    if 'thermal_analysis' in components:
                        response_input['thermal_features'] = components['thermal_analysis']
                    if 'gas_analysis' in components:
                        response_input['gas_features'] = components['gas_analysis']
                
                response_result = responder.process(response_input)
                return response_result
            except Exception as e:
                self.logger.error(f"Response stage error: {str(e)}")
        
        return {'status': 'response_unavailable', 'response_level': 0, 'alerts': []}
    
    def _run_learning_stage(self, sensor_data: Dict[str, Any], analysis_results: Dict[str, Any], 
                           response_results: Dict[str, Any], processing_id: str) -> Dict[str, Any]:
        """Run learning stage of the pipeline."""
        if 'adaptive_learner' in self.agents:
            try:
                learner = self.agents['adaptive_learner']
                
                # Prepare learning input
                learning_input = {
                    'performance_metrics': {
                        'accuracy': analysis_results.get('confidence_score', 0.0),
                        'total_predictions': 1,
                        'correct_predictions': 1 if analysis_results.get('fire_detected') else 0
                    },
                    'prediction_results': analysis_results,
                    'response_results': response_results,
                    'sensor_data': sensor_data,
                    'processing_id': processing_id
                }
                
                # Add FLIR+SCD41 specific data for learning
                if 'flir' in sensor_data and 'flir_lepton35' in sensor_data['flir']:
                    learning_input['flir_data'] = sensor_data['flir']['flir_lepton35']
                
                if 'scd41' in sensor_data and 'scd41_co2' in sensor_data['scd41']:
                    learning_input['scd41_data'] = sensor_data['scd41']['scd41_co2']
                
                learning_result = learner.process(learning_input)
                return learning_result
            except Exception as e:
                self.logger.error(f"Learning stage error: {str(e)}")
        
        return {'status': 'learning_unavailable'}
    
    def _summarize_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of sensor data for reporting."""
        summary = {
            'data_sources': list(sensor_data.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Summarize FLIR Lepton 3.5 thermal data
        if 'flir' in sensor_data:
            flir_data = sensor_data['flir']
            if 'flir_lepton35' in flir_data:
                thermal = flir_data['flir_lepton35']
                summary['flir_lepton35'] = {
                    'max_temperature': thermal.get('t_max', 0.0),
                    'avg_temperature': thermal.get('t_mean', 0.0),
                    'hot_area_percentage': thermal.get('t_hot_area_pct', 0.0),
                    'temperature_proxy': thermal.get('tproxy_val', 0.0)
                }
        
        # Summarize SCD41 CO₂ gas data
        if 'scd41' in sensor_data:
            scd41_data = sensor_data['scd41']
            if 'scd41_co2' in scd41_data:
                gas = scd41_data['scd41_co2']
                summary['scd41_co2'] = {
                    'co2_concentration': gas.get('gas_val', 0.0),
                    'co2_change_rate': gas.get('gas_delta', 0.0),
                    'co2_velocity': gas.get('gas_vel', 0.0)
                }
        
        # Handle legacy format for backward compatibility
        if 'thermal' in sensor_data:
            thermal = sensor_data['thermal']
            summary['thermal'] = {
                'max_temperature': thermal.get('temperature_max', 0.0),
                'avg_temperature': thermal.get('temperature_avg', 0.0),
                'hotspot_count': thermal.get('hotspot_count', 0)
            }
        
        if 'gas' in sensor_data:
            gas = sensor_data['gas']
            summary['gas'] = {
                'co_concentration': gas.get('co_concentration', 0.0),
                'smoke_density': gas.get('smoke_density', 0.0)
            }
        
        # Summarize environmental data
        if 'environmental' in sensor_data:
            env = sensor_data['environmental']
            summary['environmental'] = {
                'temperature': env.get('temperature', 0.0),
                'humidity': env.get('humidity', 0.0)
            }
        
        return summary
    
    def _update_system_state(self, processing_results: Dict[str, Any]) -> None:
        """Update system state based on processing results."""
        fire_detection = processing_results.get('fire_detection', {})
        
        self.system_state.update({
            'last_update': datetime.now().isoformat(),
            'fire_detected': fire_detection.get('fire_detected', False),
            'confidence_score': fire_detection.get('confidence_score', 0.0),
            'response_level': fire_detection.get('response_level', 0),
            'alerts_active': fire_detection.get('alerts_generated', 0)
        })
        
        # Update metrics
        self.system_metrics['total_detections'] += 1
        if fire_detection.get('fire_detected'):
            self.system_metrics['true_positives'] += 1  # Assume true positive for demo
    
    def _update_system_metrics(self) -> None:
        """Update system performance metrics."""
        # Update message queue size
        self.system_state['message_queue_size'] = len(self.coordinator.message_queue)
        
        # Calculate average response time
        if len(self.data_pipeline) > 0:
            response_times = []
            for result in list(self.data_pipeline)[-10:]:  # Last 10 results
                system_status = result.get('system_status', {})
                processing_time = system_status.get('processing_time_ms', 0)
                if processing_time > 0:
                    response_times.append(processing_time)
            
            if response_times:
                self.system_metrics['average_response_time'] = sum(response_times) / len(response_times)
    
    def _check_system_health(self) -> None:
        """Check overall system health."""
        # Check if agents are responsive
        active_agents = len([agent for agent in self.agents.values() if agent])  # Simplified check
        
        if active_agents < len(self.agents):
            self.logger.warning(f"Some agents appear inactive: {active_agents}/{len(self.agents)} active")
            self.system_state['status'] = 'degraded'
        elif self.system_state['status'] == 'degraded':
            self.system_state['status'] = 'running'
    
    def _forward_to_response(self, message: Message) -> Optional[Message]:
        """Forward analysis results to response agent."""
        # In a real implementation, would forward the message
        return None
    
    def _forward_to_learning(self, message: Message) -> Optional[Message]:
        """Forward response results to learning agent."""
        # In a real implementation, would forward the message
        return None
    
    def _get_agent_type(self, agent_id: str) -> str:
        """Get the type of an agent by its ID."""
        for agent_type, agent_list in self.agent_types.items():
            if agent_id in agent_list:
                return agent_type
        return 'unknown'
    
    def _calculate_average_processing_time(self) -> float:
        """Calculate average processing time from recent data."""
        if len(self.data_pipeline) == 0:
            return 0.0
        
        recent_times = []
        for result in list(self.data_pipeline)[-50:]:  # Last 50 results
            system_status = result.get('system_status', {})
            processing_time = system_status.get('processing_time_ms', 0)
            if processing_time > 0:
                recent_times.append(processing_time)
        
        return sum(recent_times) / len(recent_times) if recent_times else 0.0
    
    def _calculate_recent_accuracy(self) -> float:
        """Calculate recent system accuracy."""
        if len(self.data_pipeline) == 0:
            return 0.0
        
        recent_results = list(self.data_pipeline)[-20:]  # Last 20 results
        
        # In a real system, would compare with ground truth
        # For demo, assume high accuracy
        return 0.92
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        active_alerts = []
        
        # Get alerts from response agent
        if 'emergency_responder' in self.agents:
            responder = self.agents['emergency_responder']
            if hasattr(responder, 'active_alerts'):
                for alert in responder.active_alerts.values():
                    alert_time = datetime.fromisoformat(alert['timestamp'])
                    if datetime.now() - alert_time < timedelta(hours=1):  # Active for 1 hour
                        active_alerts.append(alert)
        
        return active_alerts
    
    def _get_recent_detections(self) -> List[Dict[str, Any]]:
        """Get recent fire detections."""
        recent_detections = []
        
        for result in list(self.data_pipeline)[-10:]:
            fire_detection = result.get('fire_detection', {})
            if fire_detection.get('fire_detected'):
                recent_detections.append({
                    'timestamp': result['timestamp'],
                    'confidence_score': fire_detection.get('confidence_score', 0.0),
                    'response_level': fire_detection.get('response_level', 0)
                })
        
        return recent_detections
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate overall performance summary."""
        total_detections = self.system_metrics['total_detections']
        
        if total_detections == 0:
            return {
                'detection_rate': 0.0,
                'false_alarm_rate': 0.0,
                'system_reliability': 'insufficient_data'
            }
        
        detection_rate = self.system_metrics['true_positives'] / total_detections
        false_alarm_rate = self.system_metrics['false_positives'] / total_detections
        
        if detection_rate > 0.9 and false_alarm_rate < 0.1:
            reliability = 'excellent'
        elif detection_rate > 0.8 and false_alarm_rate < 0.2:
            reliability = 'good'
        elif detection_rate > 0.7:
            reliability = 'acceptable'
        else:
            reliability = 'needs_improvement'
        
        return {
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'system_reliability': reliability,
            'total_processed': total_detections,
            'uptime_score': 0.99  # Placeholder for demo
        }


# Convenience function for creating a complete multi-agent system
def create_multi_agent_fire_system(config: Optional[Dict[str, Any]] = None) -> MultiAgentFireDetectionSystem:
    """
    Create a complete multi-agent fire detection system with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured MultiAgentFireDetectionSystem
    """
    default_config = {
        'system_id': 'saafe_fire_detection_system',
        'monitoring': {
            'system_health': {
                'check_interval': 10.0,
                'health_thresholds': {'cpu': 0.8, 'memory': 0.8}
            }
        },
        'analysis': {
            'fire_pattern': {
                'confidence_threshold': 0.7,
                'pattern_window_size': 50,
                'fire_signatures': {
                    'standard_fire': {
                        'thermal_threshold': 60.0,
                        'gas_thresholds': {'co': 30.0, 'smoke': 50.0}
                    }
                }
            }
        },
        'response': {
            'emergency': {
                'response_thresholds': {
                    'LOW': 0.3,
                    'MEDIUM': 0.5,
                    'HIGH': 0.7,
                    'CRITICAL': 0.9
                },
                'alert_channels': ['system', 'email'],
                'emergency_contacts': []
            }
        },
        'learning': {
            'adaptive': {
                'learning_window_size': 1000,
                'performance_threshold': 0.85,
                'error_analysis_interval': 100
            }
        }
    }
    
    if config:
        default_config.update(config)
    
    return MultiAgentFireDetectionSystem(default_config)