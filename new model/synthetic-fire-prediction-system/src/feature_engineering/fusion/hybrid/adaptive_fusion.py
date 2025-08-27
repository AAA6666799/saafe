"""
Adaptive fusion implementation for the synthetic fire prediction system.

This module provides an implementation of adaptive fusion, which adapts
the fusion strategy based on data characteristics.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import importlib

from ...base import FeatureFusion


class AdaptiveFusion(FeatureFusion):
    """
    Implementation of adaptive fusion.
    
    This class adapts the fusion strategy based on data characteristics,
    selecting the most appropriate fusion method for the current data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adaptive fusion component.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.fusion_components = {}
        self._load_fusion_components()
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required parameters
        required_params = ['adaptation_strategy', 'fusion_components']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate adaptation strategy
        valid_strategies = ['confidence', 'quality', 'context', 'ensemble']
        if self.config['adaptation_strategy'] not in valid_strategies:
            raise ValueError(f"Invalid adaptation strategy: {self.config['adaptation_strategy']}. "
                           f"Must be one of {valid_strategies}")
        
        # Validate fusion components
        if not isinstance(self.config['fusion_components'], dict):
            raise ValueError("'fusion_components' must be a dictionary")
        
        # Set default values for optional parameters
        if 'decision_threshold' not in self.config:
            self.config['decision_threshold'] = 0.5
        
        if 'default_component' not in self.config:
            # Use the first component as default
            if self.config['fusion_components']:
                self.config['default_component'] = next(iter(self.config['fusion_components']))
            else:
                raise ValueError("No fusion components specified")
    
    def _load_fusion_components(self) -> None:
        """
        Load fusion components based on configuration.
        """
        for component_id, component_config in self.config['fusion_components'].items():
            if 'type' not in component_config or 'config' not in component_config:
                self.logger.warning(f"Skipping invalid fusion component config: {component_id}")
                continue
            
            component_type = component_config['type']
            
            try:
                # Dynamically import the fusion component class
                module_path, class_name = component_type.rsplit('.', 1)
                module = importlib.import_module(f"..{module_path}", __package__)
                component_class = getattr(module, class_name)
                
                # Initialize the component
                component = component_class(component_config['config'])
                self.fusion_components[component_id] = component
                self.logger.info(f"Loaded fusion component: {component_id} ({component_type})")
            except (ImportError, AttributeError, ValueError) as e:
                self.logger.error(f"Error loading fusion component {component_id}: {str(e)}")
    
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse features from different extractors using an adaptive approach.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the fused features
        """
        self.logger.info("Performing adaptive fusion")
        
        # Select the appropriate fusion component based on the adaptation strategy
        selected_component_id = self._select_fusion_component(
            thermal_features, gas_features, environmental_features
        )
        
        if selected_component_id not in self.fusion_components:
            self.logger.warning(f"Selected component {selected_component_id} not found, using default")
            selected_component_id = self.config.get('default_component', '')
            
            if selected_component_id not in self.fusion_components:
                self.logger.error("Default component not found")
                return {'error': 'No valid fusion component available'}
        
        # Get the selected fusion component
        fusion_component = self.fusion_components[selected_component_id]
        
        # Perform fusion using the selected component
        fused_features = fusion_component.fuse_features(
            thermal_features, gas_features, environmental_features
        )
        
        # Add metadata
        fused_features['fusion_time'] = datetime.now().isoformat()
        fused_features['adaptation_strategy'] = self.config['adaptation_strategy']
        fused_features['selected_component'] = selected_component_id
        fused_features['decision_threshold'] = self.config['decision_threshold']
        
        self.logger.info(f"Adaptive fusion completed using component: {selected_component_id}")
        return fused_features
    
    def _select_fusion_component(self,
                               thermal_features: Dict[str, Any],
                               gas_features: Dict[str, Any],
                               environmental_features: Dict[str, Any]) -> str:
        """
        Select the appropriate fusion component based on the adaptation strategy.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            ID of the selected fusion component
        """
        adaptation_strategy = self.config['adaptation_strategy']
        
        if adaptation_strategy == 'confidence':
            # Select based on confidence scores
            return self._select_by_confidence(thermal_features, gas_features, environmental_features)
        
        elif adaptation_strategy == 'quality':
            # Select based on data quality
            return self._select_by_quality(thermal_features, gas_features, environmental_features)
        
        elif adaptation_strategy == 'context':
            # Select based on context
            return self._select_by_context(thermal_features, gas_features, environmental_features)
        
        elif adaptation_strategy == 'ensemble':
            # Use an ensemble of all components
            return self._select_for_ensemble()
        
        else:
            self.logger.warning(f"Unknown adaptation strategy: {adaptation_strategy}, using default")
            return self.config.get('default_component', '')
    
    def _select_by_confidence(self,
                            thermal_features: Dict[str, Any],
                            gas_features: Dict[str, Any],
                            environmental_features: Dict[str, Any]) -> str:
        """
        Select fusion component based on confidence scores.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            ID of the selected fusion component
        """
        # Extract confidence scores from features
        thermal_confidence = self._extract_confidence(thermal_features)
        gas_confidence = self._extract_confidence(gas_features)
        env_confidence = self._extract_confidence(environmental_features)
        
        # Get confidence thresholds from config
        confidence_thresholds = self.config.get('confidence_thresholds', {})
        
        # Get component mappings
        component_mappings = self.config.get('component_mappings', {})
        
        # Determine the confidence scenario
        if thermal_confidence > confidence_thresholds.get('high', 0.8) and \
           gas_confidence > confidence_thresholds.get('high', 0.8) and \
           env_confidence > confidence_thresholds.get('high', 0.8):
            # All sources have high confidence
            scenario = 'all_high'
        
        elif thermal_confidence < confidence_thresholds.get('low', 0.3) and \
             gas_confidence < confidence_thresholds.get('low', 0.3) and \
             env_confidence < confidence_thresholds.get('low', 0.3):
            # All sources have low confidence
            scenario = 'all_low'
        
        elif thermal_confidence > confidence_thresholds.get('high', 0.8):
            # Thermal has high confidence
            scenario = 'thermal_high'
        
        elif gas_confidence > confidence_thresholds.get('high', 0.8):
            # Gas has high confidence
            scenario = 'gas_high'
        
        elif env_confidence > confidence_thresholds.get('high', 0.8):
            # Environmental has high confidence
            scenario = 'env_high'
        
        else:
            # Mixed confidence
            scenario = 'mixed'
        
        # Select component based on scenario
        selected_component = component_mappings.get(scenario, self.config.get('default_component', ''))
        
        self.logger.info(f"Selected component {selected_component} based on confidence scenario: {scenario}")
        return selected_component
    
    def _select_by_quality(self,
                         thermal_features: Dict[str, Any],
                         gas_features: Dict[str, Any],
                         environmental_features: Dict[str, Any]) -> str:
        """
        Select fusion component based on data quality.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            ID of the selected fusion component
        """
        # Calculate quality scores for each feature set
        thermal_quality = self._calculate_quality(thermal_features)
        gas_quality = self._calculate_quality(gas_features)
        env_quality = self._calculate_quality(environmental_features)
        
        # Get quality thresholds from config
        quality_thresholds = self.config.get('quality_thresholds', {})
        
        # Get component mappings
        component_mappings = self.config.get('component_mappings', {})
        
        # Determine the quality scenario
        if thermal_quality > quality_thresholds.get('high', 0.8) and \
           gas_quality > quality_thresholds.get('high', 0.8) and \
           env_quality > quality_thresholds.get('high', 0.8):
            # All sources have high quality
            scenario = 'all_high_quality'
        
        elif thermal_quality < quality_thresholds.get('low', 0.3) and \
             gas_quality < quality_thresholds.get('low', 0.3) and \
             env_quality < quality_thresholds.get('low', 0.3):
            # All sources have low quality
            scenario = 'all_low_quality'
        
        elif thermal_quality > quality_thresholds.get('high', 0.8):
            # Thermal has high quality
            scenario = 'thermal_high_quality'
        
        elif gas_quality > quality_thresholds.get('high', 0.8):
            # Gas has high quality
            scenario = 'gas_high_quality'
        
        elif env_quality > quality_thresholds.get('high', 0.8):
            # Environmental has high quality
            scenario = 'env_high_quality'
        
        else:
            # Mixed quality
            scenario = 'mixed_quality'
        
        # Select component based on scenario
        selected_component = component_mappings.get(scenario, self.config.get('default_component', ''))
        
        self.logger.info(f"Selected component {selected_component} based on quality scenario: {scenario}")
        return selected_component
    
    def _select_by_context(self,
                         thermal_features: Dict[str, Any],
                         gas_features: Dict[str, Any],
                         environmental_features: Dict[str, Any]) -> str:
        """
        Select fusion component based on context.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            ID of the selected fusion component
        """
        # Extract context indicators from features
        context = self._extract_context(thermal_features, gas_features, environmental_features)
        
        # Get component mappings
        component_mappings = self.config.get('component_mappings', {})
        
        # Determine the context scenario
        if context.get('is_fire_likely', False):
            # Fire is likely
            scenario = 'fire_likely'
        
        elif context.get('is_false_alarm_likely', False):
            # False alarm is likely
            scenario = 'false_alarm_likely'
        
        elif context.get('is_ambiguous', False):
            # Ambiguous situation
            scenario = 'ambiguous'
        
        elif context.get('is_normal', False):
            # Normal situation
            scenario = 'normal'
        
        else:
            # Unknown context
            scenario = 'unknown'
        
        # Select component based on scenario
        selected_component = component_mappings.get(scenario, self.config.get('default_component', ''))
        
        self.logger.info(f"Selected component {selected_component} based on context scenario: {scenario}")
        return selected_component
    
    def _select_for_ensemble(self) -> str:
        """
        Select fusion component for ensemble approach.
        
        Returns:
            ID of the selected fusion component
        """
        # For ensemble, we typically use a component that can combine results from multiple models
        ensemble_component = self.config.get('ensemble_component', self.config.get('default_component', ''))
        
        self.logger.info(f"Selected component {ensemble_component} for ensemble approach")
        return ensemble_component
    
    def _extract_confidence(self, features: Dict[str, Any]) -> float:
        """
        Extract confidence score from features.
        
        Args:
            features: Features to extract confidence from
            
        Returns:
            Confidence score between 0 and 1
        """
        # Look for confidence in the features
        if 'confidence' in features:
            return float(features['confidence'])
        
        if 'probability' in features:
            return float(features['probability'])
        
        if 'risk_score' in features:
            return float(features['risk_score'])
        
        # If no direct confidence indicator is found, calculate based on feature completeness
        return self._calculate_quality(features)
    
    def _calculate_quality(self, features: Dict[str, Any]) -> float:
        """
        Calculate quality score for features.
        
        Args:
            features: Features to calculate quality for
            
        Returns:
            Quality score between 0 and 1
        """
        # Count the number of non-null features
        non_null_count = 0
        total_count = 0
        
        def count_non_null(d, parent_key=''):
            nonlocal non_null_count, total_count
            
            for k, v in d.items():
                if isinstance(v, dict):
                    count_non_null(v, f"{parent_key}{k}_")
                else:
                    total_count += 1
                    if v is not None:
                        non_null_count += 1
        
        count_non_null(features)
        
        # Calculate quality score
        if total_count > 0:
            return non_null_count / total_count
        else:
            return 0.0
    
    def _extract_context(self,
                       thermal_features: Dict[str, Any],
                       gas_features: Dict[str, Any],
                       environmental_features: Dict[str, Any]) -> Dict[str, bool]:
        """
        Extract context indicators from features.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary of context indicators
        """
        context = {
            'is_fire_likely': False,
            'is_false_alarm_likely': False,
            'is_ambiguous': False,
            'is_normal': True  # Default to normal
        }
        
        # Check for fire indicators
        fire_indicators = 0
        
        # Check thermal indicators
        max_temp = None
        for key, value in thermal_features.items():
            if 'max_temperature' in key and isinstance(value, (int, float)):
                max_temp = value
                break
        
        if max_temp is not None and max_temp > 100:  # Example threshold
            fire_indicators += 1
        
        # Check gas indicators
        max_conc = None
        for key, value in gas_features.items():
            if 'concentration' in key and isinstance(value, (int, float)):
                max_conc = value
                break
        
        if max_conc is not None and max_conc > 50:  # Example threshold
            fire_indicators += 1
        
        # Check environmental indicators
        temp_rise = None
        for key, value in environmental_features.items():
            if 'temperature_rise' in key and isinstance(value, (int, float)):
                temp_rise = value
                break
        
        if temp_rise is not None and temp_rise > 5:  # Example threshold
            fire_indicators += 1
        
        # Determine context based on indicators
        if fire_indicators >= 2:
            context['is_fire_likely'] = True
            context['is_normal'] = False
        elif fire_indicators == 1:
            context['is_ambiguous'] = True
            context['is_normal'] = False
        
        # Check for false alarm indicators
        if max_temp is not None and max_temp > 80 and max_temp < 100:
            if max_conc is not None and max_conc < 20:
                context['is_false_alarm_likely'] = True
                context['is_normal'] = False
        
        return context
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # For adaptive fusion, we use the risk score from the selected component
        # if it's available, otherwise we calculate it based on the fused features
        
        # Check if there's a risk score already calculated
        if 'risk_score' in fused_features:
            return float(fused_features['risk_score'])
        
        # Check if there's a fused probability or confidence
        if 'fused_probability' in fused_features:
            return float(fused_features['fused_probability'])
        
        if 'fused_confidence' in fused_features:
            fused_decision = fused_features.get('fused_decision', False)
            fused_confidence = float(fused_features['fused_confidence'])
            
            if fused_decision:
                return fused_confidence
            else:
                return 0.1 * (1.0 - fused_confidence)
        
        # If no direct risk indicators are available, calculate based on available features
        risk_indicators = []
        
        # Check for thermal indicators
        if 'max_temperature' in fused_features and isinstance(fused_features['max_temperature'], (int, float)):
            max_temp = fused_features['max_temperature']
            if max_temp > 100:  # Example threshold
                risk_indicators.append(min(1.0, (max_temp - 100) / 100))
        
        # Check for gas indicators
        if 'gas_concentration' in fused_features and isinstance(fused_features['gas_concentration'], (int, float)):
            gas_conc = fused_features['gas_concentration']
            if gas_conc > 50:  # Example threshold
                risk_indicators.append(min(1.0, (gas_conc - 50) / 100))
        
        # Check for environmental indicators
        if 'temperature_rise' in fused_features and isinstance(fused_features['temperature_rise'], (int, float)):
            temp_rise = fused_features['temperature_rise']
            if temp_rise > 5:  # Example threshold
                risk_indicators.append(min(1.0, temp_rise / 20))
        
        # Calculate overall risk score
        if risk_indicators:
            # Use a weighted average of the top 3 risk indicators
            risk_indicators.sort(reverse=True)
            top_indicators = risk_indicators[:3]
            weights = [0.5, 0.3, 0.2][:len(top_indicators)]
            
            risk_score = sum(indicator * weight for indicator, weight in zip(top_indicators, weights))
        else:
            # If no risk indicators are available, use a default low risk score
            risk_score = 0.1
        
        self.logger.info(f"Calculated risk score: {risk_score}")
        return risk_score