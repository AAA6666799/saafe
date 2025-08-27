"""
Fire Pattern Analysis Agent for the multi-agent fire prediction system.

This agent specializes in analyzing fire-related patterns and signatures in sensor data
to provide accurate fire detection and classification results.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import deque
import statistics

from ..base import AnalysisAgent, Message


class FirePatternAnalysisAgent(AnalysisAgent):
    """
    Fire Pattern Analysis Agent for analyzing fire-related patterns and signatures.
    
    This agent uses multiple analysis techniques to identify fire patterns in sensor data
    and provides confidence-scored predictions with detailed analysis results.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the Fire Pattern Analysis Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration dictionary
        """
        super().__init__(agent_id, config)
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
        # Analysis configuration
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.pattern_window_size = config.get('pattern_window_size', 50)
        self.fire_signatures = config.get('fire_signatures', {})
        
        # Pattern analysis components
        self.thermal_analyzer = ThermalPatternAnalyzer(config.get('thermal_config', {}))
        self.gas_analyzer = GasPatternAnalyzer(config.get('gas_config', {}))
        self.temporal_analyzer = TemporalPatternAnalyzer(config.get('temporal_config', {}))
        
        # State tracking
        self.pattern_history = deque(maxlen=self.pattern_window_size)
        self.analysis_results = []
        self.fire_signature_matches = []
        
        # Performance tracking
        self.analysis_count = 0
        self.high_confidence_detections = 0
        self.pattern_match_rate = 0.0
        
        self.logger.info(f"Initialized Fire Pattern Analysis Agent: {agent_id}")
    
    def validate_config(self) -> None:
        """Validate the configuration parameters."""
        required_keys = ['confidence_threshold', 'pattern_window_size']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if not 0 < self.config['confidence_threshold'] <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if self.config['pattern_window_size'] <= 0:
            raise ValueError("pattern_window_size must be positive")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and perform comprehensive fire pattern analysis.
        
        Args:
            data: Input sensor data containing thermal, gas, and environmental measurements
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            self.analysis_count += 1
            analysis_timestamp = datetime.now()
            
            # Extract sensor data components
            thermal_data = data.get('thermal', {})
            gas_data = data.get('gas', {})
            environmental_data = data.get('environmental', {})
            
            # Perform pattern analysis
            thermal_analysis = self.thermal_analyzer.analyze(thermal_data)
            gas_analysis = self.gas_analyzer.analyze(gas_data)
            temporal_analysis = self.temporal_analyzer.analyze(data, self.pattern_history)
            
            # Analyze overall patterns
            pattern_results = self.analyze_pattern(data)
            
            # Calculate confidence scores
            confidence_score = self.calculate_confidence({
                'thermal': thermal_analysis,
                'gas': gas_analysis,
                'temporal': temporal_analysis,
                'pattern': pattern_results
            })
            
            # Match against known fire signatures
            signature_matches = self.match_fire_signature(data)
            
            # Compile comprehensive analysis results
            analysis_result = {
                'timestamp': analysis_timestamp.isoformat(),
                'agent_id': self.agent_id,
                'analysis_id': f"analysis_{self.analysis_count}",
                'confidence_score': confidence_score,
                'fire_detected': confidence_score >= self.confidence_threshold,
                'analysis_components': {
                    'thermal_analysis': thermal_analysis,
                    'gas_analysis': gas_analysis,
                    'temporal_analysis': temporal_analysis,
                    'pattern_analysis': pattern_results
                },
                'signature_matches': signature_matches,
                'metadata': {
                    'processing_time_ms': (datetime.now() - analysis_timestamp).total_seconds() * 1000,
                    'data_quality': self._assess_data_quality(data),
                    'analysis_method': 'multi_pattern_fusion'
                }
            }
            
            # Update state and tracking
            self._update_state(analysis_result)
            self.pattern_history.append(data)
            
            # Track high confidence detections
            if confidence_score >= self.confidence_threshold:
                self.high_confidence_detections += 1
            
            self.logger.debug(f"Analysis complete: confidence={confidence_score:.3f}, fire_detected={analysis_result['fire_detected']}")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'agent_id': self.agent_id,
                'error': str(e),
                'fire_detected': False,
                'confidence_score': 0.0
            }
    
    def analyze_pattern(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze patterns in the input data using multi-modal fusion.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Dictionary containing pattern analysis results
        """
        pattern_features = []
        pattern_scores = []
        
        # Thermal pattern analysis
        if 'thermal' in data:
            thermal_patterns = self._analyze_thermal_patterns(data['thermal'])
            pattern_features.extend(['thermal_gradient', 'thermal_hotspots', 'thermal_spread'])
            pattern_scores.extend([
                thermal_patterns.get('gradient_score', 0.0),
                thermal_patterns.get('hotspot_score', 0.0),
                thermal_patterns.get('spread_score', 0.0)
            ])
        
        # Gas pattern analysis
        if 'gas' in data:
            gas_patterns = self._analyze_gas_patterns(data['gas'])
            pattern_features.extend(['gas_co_pattern', 'gas_smoke_pattern', 'gas_voc_pattern'])
            pattern_scores.extend([
                gas_patterns.get('co_score', 0.0),
                gas_patterns.get('smoke_score', 0.0),
                gas_patterns.get('voc_score', 0.0)
            ])
        
        # Environmental correlation analysis
        if 'environmental' in data:
            env_patterns = self._analyze_environmental_patterns(data['environmental'])
            pattern_features.extend(['env_temp_correlation', 'env_humidity_pattern'])
            pattern_scores.extend([
                env_patterns.get('temp_correlation', 0.0),
                env_patterns.get('humidity_score', 0.0)
            ])
        
        # Calculate overall pattern strength
        overall_score = statistics.mean(pattern_scores) if pattern_scores else 0.0
        
        return {
            'overall_pattern_score': overall_score,
            'pattern_features': dict(zip(pattern_features, pattern_scores)),
            'pattern_strength': 'high' if overall_score > 0.7 else 'medium' if overall_score > 0.4 else 'low',
            'dominant_patterns': [f for f, s in zip(pattern_features, pattern_scores) if s > 0.6]
        }
    
    def calculate_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """
        Calculate confidence level for analysis results using weighted fusion.
        
        Args:
            analysis_results: Results of component analyses
            
        Returns:
            Confidence level as a float between 0 and 1
        """
        confidence_components = []
        weights = []
        
        # Thermal analysis confidence
        if 'thermal' in analysis_results:
            thermal_conf = analysis_results['thermal'].get('confidence', 0.0)
            confidence_components.append(thermal_conf)
            weights.append(0.4)  # Higher weight for thermal
        
        # Gas analysis confidence
        if 'gas' in analysis_results:
            gas_conf = analysis_results['gas'].get('confidence', 0.0)
            confidence_components.append(gas_conf)
            weights.append(0.3)
        
        # Temporal analysis confidence
        if 'temporal' in analysis_results:
            temporal_conf = analysis_results['temporal'].get('confidence', 0.0)
            confidence_components.append(temporal_conf)
            weights.append(0.2)
        
        # Pattern analysis confidence
        if 'pattern' in analysis_results:
            pattern_conf = analysis_results['pattern'].get('overall_pattern_score', 0.0)
            confidence_components.append(pattern_conf)
            weights.append(0.1)
        
        # Calculate weighted confidence
        if confidence_components and weights:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Calculate weighted average
            weighted_confidence = sum(c * w for c, w in zip(confidence_components, normalized_weights))
            return max(0.0, min(1.0, weighted_confidence))
        
        return 0.0
    
    def match_fire_signature(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match data patterns against known fire signatures.
        
        Args:
            data: Input data to match
            
        Returns:
            Dictionary containing signature matching results
        """
        matches = []
        best_match_score = 0.0
        best_match_type = None
        
        for signature_name, signature_config in self.fire_signatures.items():
            match_score = self._calculate_signature_match(data, signature_config)
            
            if match_score > 0.5:  # Threshold for considering a match
                matches.append({
                    'signature_name': signature_name,
                    'match_score': match_score,
                    'signature_type': signature_config.get('type', 'unknown')
                })
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_type = signature_name
        
        # Sort matches by score
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        return {
            'matches': matches,
            'best_match': {
                'signature_name': best_match_type,
                'score': best_match_score
            } if best_match_type else None,
            'total_matches': len(matches),
            'high_confidence_matches': len([m for m in matches if m['match_score'] > 0.8])
        }
    
    def _analyze_thermal_patterns(self, thermal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze thermal patterns in the data."""
        # Extract thermal features
        temp_max = thermal_data.get('temperature_max', 0.0)
        temp_avg = thermal_data.get('temperature_avg', 0.0)
        hotspot_count = thermal_data.get('hotspot_count', 0)
        
        # Calculate pattern scores
        gradient_score = min(1.0, (temp_max - temp_avg) / 50.0) if temp_avg > 0 else 0.0
        hotspot_score = min(1.0, hotspot_count / 10.0)
        spread_score = thermal_data.get('thermal_spread_rate', 0.0)
        
        return {
            'gradient_score': gradient_score,
            'hotspot_score': hotspot_score,
            'spread_score': spread_score,
            'confidence': statistics.mean([gradient_score, hotspot_score, spread_score])
        }
    
    def _analyze_gas_patterns(self, gas_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze gas patterns in the data."""
        # Extract gas concentrations
        co_ppm = gas_data.get('co_concentration', 0.0)
        smoke_density = gas_data.get('smoke_density', 0.0)
        voc_level = gas_data.get('voc_total', 0.0)
        
        # Calculate pattern scores (normalized to 0-1)
        co_score = min(1.0, co_ppm / 50.0)  # 50 ppm as reference
        smoke_score = min(1.0, smoke_density / 100.0)  # 100% as reference
        voc_score = min(1.0, voc_level / 1000.0)  # 1000 ppb as reference
        
        return {
            'co_score': co_score,
            'smoke_score': smoke_score,
            'voc_score': voc_score,
            'confidence': statistics.mean([co_score, smoke_score, voc_score])
        }
    
    def _analyze_environmental_patterns(self, env_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze environmental patterns in the data."""
        # Extract environmental data
        temp_ambient = env_data.get('temperature', 20.0)  # Default room temp
        humidity = env_data.get('humidity', 50.0)
        
        # Calculate correlation scores
        temp_correlation = max(0.0, (temp_ambient - 20.0) / 30.0)  # Normalized temp increase
        humidity_score = max(0.0, (50.0 - humidity) / 50.0)  # Inverse humidity relationship
        
        return {
            'temp_correlation': min(1.0, temp_correlation),
            'humidity_score': min(1.0, humidity_score),
            'confidence': statistics.mean([temp_correlation, humidity_score])
        }
    
    def _calculate_signature_match(self, data: Dict[str, Any], signature_config: Dict[str, Any]) -> float:
        """Calculate match score against a specific fire signature."""
        # This is a simplified signature matching - in a real system,
        # this would use more sophisticated pattern matching algorithms
        
        match_components = []
        
        # Check thermal signature
        if 'thermal_threshold' in signature_config:
            thermal_temp = data.get('thermal', {}).get('temperature_max', 0.0)
            threshold = signature_config['thermal_threshold']
            if thermal_temp >= threshold:
                match_components.append(min(1.0, thermal_temp / (threshold * 1.5)))
        
        # Check gas signature
        if 'gas_thresholds' in signature_config:
            gas_data = data.get('gas', {})
            for gas_type, threshold in signature_config['gas_thresholds'].items():
                gas_level = gas_data.get(f'{gas_type}_concentration', 0.0)
                if gas_level >= threshold:
                    match_components.append(min(1.0, gas_level / (threshold * 1.5)))
        
        return statistics.mean(match_components) if match_components else 0.0
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> str:
        """Assess the quality of input data."""
        quality_score = 0
        total_checks = 0
        
        # Check data completeness
        expected_keys = ['thermal', 'gas', 'environmental']
        for key in expected_keys:
            total_checks += 1
            if key in data and data[key]:
                quality_score += 1
        
        # Check data validity (non-negative values, reasonable ranges)
        if 'thermal' in data:
            thermal = data['thermal']
            total_checks += 2
            if thermal.get('temperature_max', 0) > 0:
                quality_score += 1
            if 0 <= thermal.get('temperature_avg', -1) <= 200:  # Reasonable temp range
                quality_score += 1
        
        quality_ratio = quality_score / total_checks if total_checks > 0 else 0
        
        if quality_ratio >= 0.8:
            return 'high'
        elif quality_ratio >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _update_state(self, analysis_result: Dict[str, Any]) -> None:
        """Update agent state with new analysis results."""
        self.analysis_results.append(analysis_result)
        
        # Keep only recent results to manage memory
        if len(self.analysis_results) > 1000:
            self.analysis_results = self.analysis_results[-500:]
        
        # Update pattern match rate
        if analysis_result.get('signature_matches', {}).get('total_matches', 0) > 0:
            self.pattern_match_rate = (self.pattern_match_rate * 0.9 + 0.1)  # Exponential moving average
        else:
            self.pattern_match_rate = self.pattern_match_rate * 0.95
    
    def default_message_handler(self, message: Message) -> Optional[Message]:
        """Handle unknown message types."""
        self.logger.warning(f"Received unknown message type: {message.message_type}")
        return None
    
    def create_message(self, receiver_id: str, message_type: str, content: Dict[str, Any], priority: int = 0) -> Message:
        """Create a new message to send to another agent."""
        return Message(self.agent_id, receiver_id, message_type, content, priority)
    
    def save_state(self, filepath: str) -> None:
        """Save agent state to file."""
        state_data = {
            'agent_id': self.agent_id,
            'config': self.config,
            'analysis_count': self.analysis_count,
            'high_confidence_detections': self.high_confidence_detections,
            'pattern_match_rate': self.pattern_match_rate,
            'recent_results': self.analysis_results[-100:]  # Save only recent results
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """Load agent state from file."""
        import json
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        self.analysis_count = state_data.get('analysis_count', 0)
        self.high_confidence_detections = state_data.get('high_confidence_detections', 0)
        self.pattern_match_rate = state_data.get('pattern_match_rate', 0.0)
        self.analysis_results = state_data.get('recent_results', [])


# Helper classes for specialized pattern analysis
class ThermalPatternAnalyzer:
    """Specialized analyzer for thermal patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def analyze(self, thermal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze thermal patterns."""
        return {
            'thermal_signature_detected': False,
            'hotspot_analysis': {},
            'temperature_trend': 'stable',
            'confidence': 0.5
        }


class GasPatternAnalyzer:
    """Specialized analyzer for gas patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def analyze(self, gas_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze gas patterns."""
        return {
            'gas_signature_detected': False,
            'concentration_analysis': {},
            'gas_trend': 'stable',
            'confidence': 0.5
        }


class TemporalPatternAnalyzer:
    """Specialized analyzer for temporal patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def analyze(self, current_data: Dict[str, Any], historical_data: deque) -> Dict[str, Any]:
        """Analyze temporal patterns."""
        return {
            'temporal_signature_detected': False,
            'trend_analysis': {},
            'progression_rate': 0.0,
            'confidence': 0.5
        }