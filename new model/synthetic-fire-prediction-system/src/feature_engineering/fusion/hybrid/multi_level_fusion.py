"""
Multi-level fusion implementation for the synthetic fire prediction system.

This module provides an implementation of multi-level fusion, which combines
features at multiple levels of the processing pipeline.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import importlib

from ...base import FeatureFusion


class MultiLevelFusion(FeatureFusion):
    """
    Implementation of multi-level fusion.
    
    This class combines features at multiple levels of the processing pipeline,
    including data-level, feature-level, and decision-level fusion.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the multi-level fusion component.
        
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
        required_params = ['levels', 'fusion_components']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate levels
        if not isinstance(self.config['levels'], list) or not self.config['levels']:
            raise ValueError("'levels' must be a non-empty list")
        
        # Validate fusion components
        if not isinstance(self.config['fusion_components'], dict):
            raise ValueError("'fusion_components' must be a dictionary")
        
        # Set default values for optional parameters
        if 'decision_threshold' not in self.config:
            self.config['decision_threshold'] = 0.5
    
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
        Fuse features from different extractors using a multi-level approach.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the fused features
        """
        self.logger.info("Performing multi-level fusion")
        
        # Initialize feature dictionary with input features
        current_features = {
            'thermal': thermal_features,
            'gas': gas_features,
            'environmental': environmental_features
        }
        
        # Initialize level results
        level_results = []
        
        # Process each level in the fusion pipeline
        for level_idx, level in enumerate(self.config['levels']):
            level_id = level.get('id', f"level_{level_idx}")
            level_type = level.get('type', '')
            
            self.logger.info(f"Processing fusion level: {level_id} ({level_type})")
            
            if level_type == 'data':
                # Data-level fusion
                level_result = self._process_data_level(level, current_features)
            
            elif level_type == 'feature':
                # Feature-level fusion
                level_result = self._process_feature_level(level, current_features)
            
            elif level_type == 'decision':
                # Decision-level fusion
                level_result = self._process_decision_level(level, current_features)
            
            else:
                self.logger.warning(f"Unknown level type: {level_type}, skipping")
                continue
            
            # Add level metadata
            level_result['level_id'] = level_id
            level_result['level_type'] = level_type
            
            # Update current features with level result
            output_key = level.get('output_key', f"level_{level_idx}_output")
            current_features[output_key] = level_result
            
            # Add to level results
            level_results.append(level_result)
        
        # Get final output from the last level
        if level_results:
            final_output = level_results[-1]
        else:
            self.logger.warning("No levels were processed")
            final_output = {}
        
        # Add metadata
        final_output['fusion_time'] = datetime.now().isoformat()
        final_output['levels'] = self.config['levels']
        final_output['decision_threshold'] = self.config['decision_threshold']
        final_output['level_results'] = level_results
        
        self.logger.info("Multi-level fusion completed")
        return final_output
    
    def _process_data_level(self, level: Dict[str, Any], current_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data-level fusion.
        
        Args:
            level: Level configuration
            current_features: Current features dictionary
            
        Returns:
            Dictionary containing the fused features
        """
        # Get fusion component for this level
        component_id = level.get('component', '')
        if not component_id or component_id not in self.fusion_components:
            self.logger.warning(f"Invalid fusion component for data level: {component_id}")
            return {}
        
        # Get input sources
        input_sources = level.get('inputs', ['thermal', 'gas', 'environmental'])
        
        # Get raw data for each source
        thermal_data = current_features.get(input_sources[0], {}) if len(input_sources) > 0 else {}
        gas_data = current_features.get(input_sources[1], {}) if len(input_sources) > 1 else {}
        env_data = current_features.get(input_sources[2], {}) if len(input_sources) > 2 else {}
        
        # Perform data-level fusion
        fusion_component = self.fusion_components[component_id]
        fused_data = fusion_component.fuse_features(thermal_data, gas_data, env_data)
        
        # Add level-specific metadata
        fused_data['fusion_level'] = 'data'
        fused_data['fusion_component'] = component_id
        
        return fused_data
    
    def _process_feature_level(self, level: Dict[str, Any], current_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process feature-level fusion.
        
        Args:
            level: Level configuration
            current_features: Current features dictionary
            
        Returns:
            Dictionary containing the fused features
        """
        # Get fusion component for this level
        component_id = level.get('component', '')
        if not component_id or component_id not in self.fusion_components:
            self.logger.warning(f"Invalid fusion component for feature level: {component_id}")
            return {}
        
        # Get input sources
        input_sources = level.get('inputs', ['thermal', 'gas', 'environmental'])
        
        # Get features for each source
        thermal_features = current_features.get(input_sources[0], {}) if len(input_sources) > 0 else {}
        gas_features = current_features.get(input_sources[1], {}) if len(input_sources) > 1 else {}
        env_features = current_features.get(input_sources[2], {}) if len(input_sources) > 2 else {}
        
        # Apply feature transformations if specified
        if 'transformations' in level:
            thermal_features = self._apply_transformations(thermal_features, level['transformations'].get('thermal', []))
            gas_features = self._apply_transformations(gas_features, level['transformations'].get('gas', []))
            env_features = self._apply_transformations(env_features, level['transformations'].get('environmental', []))
        
        # Perform feature-level fusion
        fusion_component = self.fusion_components[component_id]
        fused_features = fusion_component.fuse_features(thermal_features, gas_features, env_features)
        
        # Add level-specific metadata
        fused_features['fusion_level'] = 'feature'
        fused_features['fusion_component'] = component_id
        
        return fused_features
    
    def _process_decision_level(self, level: Dict[str, Any], current_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process decision-level fusion.
        
        Args:
            level: Level configuration
            current_features: Current features dictionary
            
        Returns:
            Dictionary containing the fused features
        """
        # Get fusion component for this level
        component_id = level.get('component', '')
        if not component_id or component_id not in self.fusion_components:
            self.logger.warning(f"Invalid fusion component for decision level: {component_id}")
            return {}
        
        # Get input sources
        input_sources = level.get('inputs', ['thermal', 'gas', 'environmental'])
        
        # Get features for each source
        thermal_features = current_features.get(input_sources[0], {}) if len(input_sources) > 0 else {}
        gas_features = current_features.get(input_sources[1], {}) if len(input_sources) > 1 else {}
        env_features = current_features.get(input_sources[2], {}) if len(input_sources) > 2 else {}
        
        # Extract decisions if specified
        if 'decision_extraction' in level:
            thermal_features = self._extract_decision(thermal_features, level['decision_extraction'].get('thermal', {}))
            gas_features = self._extract_decision(gas_features, level['decision_extraction'].get('gas', {}))
            env_features = self._extract_decision(env_features, level['decision_extraction'].get('environmental', {}))
        
        # Perform decision-level fusion
        fusion_component = self.fusion_components[component_id]
        fused_decision = fusion_component.fuse_features(thermal_features, gas_features, env_features)
        
        # Add level-specific metadata
        fused_decision['fusion_level'] = 'decision'
        fused_decision['fusion_component'] = component_id
        
        return fused_decision
    
    def _apply_transformations(self, features: Dict[str, Any], transformations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply transformations to features.
        
        Args:
            features: Features to transform
            transformations: List of transformations to apply
            
        Returns:
            Dictionary containing the transformed features
        """
        transformed = features.copy()
        
        for transform in transformations:
            transform_type = transform.get('type', '')
            
            if transform_type == 'normalization':
                # Apply normalization
                method = transform.get('method', 'min_max')
                transformed = self._normalize_features(transformed, method)
            
            elif transform_type == 'selection':
                # Apply feature selection
                method = transform.get('method', 'threshold')
                threshold = transform.get('threshold', 0.5)
                top_k = transform.get('top_k', 10)
                transformed = self._select_features(transformed, method, threshold, top_k)
            
            elif transform_type == 'aggregation':
                # Apply feature aggregation
                method = transform.get('method', 'mean')
                transformed = self._aggregate_features(transformed, method)
            
            elif transform_type == 'filtering':
                # Apply feature filtering
                filter_type = transform.get('filter', 'threshold')
                threshold = transform.get('threshold', 0.5)
                transformed = self._filter_features(transformed, filter_type, threshold)
        
        return transformed
    
    def _extract_decision(self, features: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract decision from features.
        
        Args:
            features: Features to extract decision from
            config: Decision extraction configuration
            
        Returns:
            Dictionary containing the decision
        """
        decision_method = config.get('method', 'threshold')
        threshold = config.get('threshold', self.config['decision_threshold'])
        
        if decision_method == 'threshold':
            # Extract decision based on a specific feature
            feature_name = config.get('feature', '')
            
            if feature_name and feature_name in features:
                feature_value = features[feature_name]
                
                if isinstance(feature_value, (int, float)):
                    decision = feature_value >= threshold
                    confidence = abs(feature_value - threshold) / max(1.0, threshold)
                    confidence = min(1.0, confidence)
                    
                    return {
                        'decision': decision,
                        'confidence': confidence,
                        'feature': feature_name,
                        'threshold': threshold
                    }
            
            # If feature not found, check for common decision indicators
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    if 'probability' in key or 'confidence' in key or 'risk_score' in key:
                        decision = value >= threshold
                        confidence = abs(value - threshold) / max(1.0, threshold)
                        confidence = min(1.0, confidence)
                        
                        return {
                            'decision': decision,
                            'confidence': confidence,
                            'feature': key,
                            'threshold': threshold
                        }
        
        elif decision_method == 'model':
            # Extract decision from a model output
            if 'decision' in features:
                decision = features['decision']
                confidence = features.get('confidence', 0.5)
                
                return {
                    'decision': decision,
                    'confidence': confidence,
                    'method': 'model'
                }
        
        # Default decision
        return {
            'decision': False,
            'confidence': 0.5,
            'method': 'default'
        }
    
    def _normalize_features(self, features: Dict[str, Any], method: str) -> Dict[str, Any]:
        """
        Normalize features.
        
        Args:
            features: Features to normalize
            method: Normalization method
            
        Returns:
            Dictionary containing the normalized features
        """
        # Create a copy of the features
        normalized = features.copy()
        
        # Extract numeric features
        numeric_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)) and key not in ['level_id', 'level_type', 'fusion_component']:
                numeric_features[key] = value
        
        if not numeric_features:
            return normalized
        
        if method == 'min_max':
            # Min-max normalization
            min_val = min(numeric_features.values())
            max_val = max(numeric_features.values())
            
            if max_val > min_val:
                for key in numeric_features:
                    normalized[key] = (numeric_features[key] - min_val) / (max_val - min_val)
        
        elif method == 'z_score':
            # Z-score normalization
            mean = sum(numeric_features.values()) / len(numeric_features)
            std = np.std(list(numeric_features.values()))
            
            if std > 0:
                for key in numeric_features:
                    normalized[key] = (numeric_features[key] - mean) / std
        
        # Add normalization metadata
        normalized['normalization_method'] = method
        
        return normalized
    
    def _select_features(self, features: Dict[str, Any], method: str, threshold: float, top_k: int) -> Dict[str, Any]:
        """
        Select features.
        
        Args:
            features: Features to select from
            method: Selection method
            threshold: Threshold for selection
            top_k: Number of top features to select
            
        Returns:
            Dictionary containing the selected features
        """
        # Create a new dictionary for selected features
        selected = {}
        
        # Extract numeric features
        numeric_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)) and key not in ['level_id', 'level_type', 'fusion_component']:
                numeric_features[key] = value
        
        if not numeric_features:
            return features
        
        if method == 'threshold':
            # Select features above threshold
            for key, value in numeric_features.items():
                if value >= threshold:
                    selected[key] = value
        
        elif method == 'top_k':
            # Select top k features
            sorted_features = sorted(numeric_features.items(), key=lambda x: x[1], reverse=True)
            for key, value in sorted_features[:top_k]:
                selected[key] = value
        
        # Add metadata
        selected['level_id'] = features.get('level_id', '')
        selected['level_type'] = features.get('level_type', '')
        selected['selection_method'] = method
        
        return selected
    
    def _aggregate_features(self, features: Dict[str, Any], method: str) -> Dict[str, Any]:
        """
        Aggregate features.
        
        Args:
            features: Features to aggregate
            method: Aggregation method
            
        Returns:
            Dictionary containing the aggregated features
        """
        # Create a new dictionary for aggregated features
        aggregated = {}
        
        # Extract numeric features
        numeric_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)) and key not in ['level_id', 'level_type', 'fusion_component']:
                numeric_features[key] = value
        
        if not numeric_features:
            return features
        
        if method == 'mean':
            # Calculate mean
            aggregated['mean'] = sum(numeric_features.values()) / len(numeric_features)
        
        elif method == 'max':
            # Calculate max
            aggregated['max'] = max(numeric_features.values())
        
        elif method == 'min':
            # Calculate min
            aggregated['min'] = min(numeric_features.values())
        
        elif method == 'sum':
            # Calculate sum
            aggregated['sum'] = sum(numeric_features.values())
        
        # Add metadata
        aggregated['level_id'] = features.get('level_id', '')
        aggregated['level_type'] = features.get('level_type', '')
        aggregated['aggregation_method'] = method
        
        return aggregated
    
    def _filter_features(self, features: Dict[str, Any], filter_type: str, threshold: float) -> Dict[str, Any]:
        """
        Filter features.
        
        Args:
            features: Features to filter
            filter_type: Type of filter to apply
            threshold: Threshold for filtering
            
        Returns:
            Dictionary containing the filtered features
        """
        # Create a copy of the features
        filtered = features.copy()
        
        if filter_type == 'threshold':
            # Filter features based on threshold
            for key, value in list(filtered.items()):
                if isinstance(value, (int, float)) and key not in ['level_id', 'level_type', 'fusion_component']:
                    if value < threshold:
                        del filtered[key]
        
        elif filter_type == 'percentile':
            # Filter features based on percentile
            numeric_features = {k: v for k, v in features.items() 
                              if isinstance(v, (int, float)) and k not in ['level_id', 'level_type', 'fusion_component']}
            
            if numeric_features:
                values = list(numeric_features.values())
                percentile_threshold = np.percentile(values, threshold * 100)
                
                for key, value in list(filtered.items()):
                    if isinstance(value, (int, float)) and key not in ['level_id', 'level_type', 'fusion_component']:
                        if value < percentile_threshold:
                            del filtered[key]
        
        # Add metadata
        filtered['filter_type'] = filter_type
        filtered['filter_threshold'] = threshold
        
        return filtered
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # For multi-level fusion, we use the risk score from the final level
        # if it's available, otherwise we calculate it based on the fused features
        
        # Check if there's a risk score already calculated
        if 'risk_score' in fused_features:
            return float(fused_features['risk_score'])
        
        # Check level results for decision-level fusion
        level_results = fused_features.get('level_results', [])
        for level in reversed(level_results):  # Start from the last level
            if level.get('level_type') == 'decision':
                if 'decision' in level and 'confidence' in level:
                    decision = level['decision']
                    confidence = level['confidence']
                    
                    if decision:
                        return confidence
                    else:
                        return 0.1 * (1.0 - confidence)
        
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