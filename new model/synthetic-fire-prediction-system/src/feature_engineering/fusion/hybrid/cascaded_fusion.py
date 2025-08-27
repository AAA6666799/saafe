"""
Cascaded fusion implementation for the synthetic fire prediction system.

This module provides an implementation of cascaded fusion, which combines
features from different sources using a cascaded approach.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import importlib

from ...base import FeatureFusion


class CascadedFusion(FeatureFusion):
    """
    Implementation of cascaded fusion.
    
    This class combines features from different sources (thermal, gas, environmental)
    using a cascaded approach, where features are fused sequentially in a pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cascaded fusion component.
        
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
        required_params = ['cascade', 'fusion_components']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate cascade
        if not isinstance(self.config['cascade'], list) or not self.config['cascade']:
            raise ValueError("'cascade' must be a non-empty list")
        
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
        Fuse features from different extractors using a cascaded approach.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the fused features
        """
        self.logger.info("Performing cascaded fusion")
        
        # Initialize feature dictionary with input features
        all_features = {
            'thermal': thermal_features,
            'gas': gas_features,
            'environmental': environmental_features
        }
        
        # Process the cascade
        cascade_steps = self.config['cascade']
        if not cascade_steps:
            self.logger.error("No steps specified in cascade")
            return {'error': 'No steps specified in cascade'}
        
        # Initialize cascade results
        cascade_results = []
        
        # Process each step in the cascade
        current_features = {
            'thermal': thermal_features,
            'gas': gas_features,
            'environmental': environmental_features
        }
        
        for step_idx, step in enumerate(cascade_steps):
            step_id = step.get('id', f"step_{step_idx}")
            step_type = step.get('type', '')
            
            if step_type == 'fusion':
                # This is a fusion step
                fusion_component_id = step.get('component', '')
                if not fusion_component_id or fusion_component_id not in self.fusion_components:
                    self.logger.warning(f"Invalid fusion component for step {step_id}: {fusion_component_id}")
                    continue
                
                # Get input sources
                input_sources = step.get('inputs', ['thermal', 'gas', 'environmental'])
                
                # Get input features
                input_features = {}
                for source in input_sources:
                    if source in current_features:
                        input_features[source] = current_features[source]
                
                # Perform fusion
                fusion_component = self.fusion_components[fusion_component_id]
                
                # Extract the required features for the fusion component
                thermal_input = input_features.get('thermal', {})
                gas_input = input_features.get('gas', {})
                env_input = input_features.get('environmental', {})
                
                # Perform fusion
                fused = fusion_component.fuse_features(thermal_input, gas_input, env_input)
                
                # Add step metadata
                fused['step_id'] = step_id
                fused['step_type'] = step_type
                fused['fusion_component'] = fusion_component_id
                
                # Update current features with fused features
                output_source = step.get('output', 'fused')
                current_features[output_source] = fused
                
                # Add to cascade results
                cascade_results.append({
                    'step_id': step_id,
                    'step_type': step_type,
                    'fusion_component': fusion_component_id,
                    'output_source': output_source,
                    'fused_features': fused
                })
            
            elif step_type == 'transform':
                # This is a transform step
                transform_type = step.get('transform', '')
                if not transform_type:
                    self.logger.warning(f"No transform type specified for step: {step_id}")
                    continue
                
                # Get input source
                input_source = step.get('input', '')
                if not input_source or input_source not in current_features:
                    self.logger.warning(f"Invalid input source for step {step_id}: {input_source}")
                    continue
                
                # Get input features
                input_features = current_features[input_source]
                
                # Apply transform
                transformed = self._apply_transform(transform_type, input_features, step)
                
                # Add step metadata
                transformed['step_id'] = step_id
                transformed['step_type'] = step_type
                transformed['transform_type'] = transform_type
                
                # Update current features with transformed features
                output_source = step.get('output', input_source)
                current_features[output_source] = transformed
                
                # Add to cascade results
                cascade_results.append({
                    'step_id': step_id,
                    'step_type': step_type,
                    'transform_type': transform_type,
                    'input_source': input_source,
                    'output_source': output_source,
                    'transformed_features': transformed
                })
            
            elif step_type == 'decision':
                # This is a decision step
                decision_source = step.get('source', '')
                if not decision_source or decision_source not in current_features:
                    self.logger.warning(f"Invalid decision source for step {step_id}: {decision_source}")
                    continue
                
                # Get decision features
                decision_features = current_features[decision_source]
                
                # Make decision
                decision, confidence = self._make_decision(decision_features, step)
                
                # Add step metadata
                decision_result = {
                    'step_id': step_id,
                    'step_type': step_type,
                    'decision_source': decision_source,
                    'decision': decision,
                    'confidence': confidence
                }
                
                # Update current features with decision result
                output_source = step.get('output', 'decision')
                current_features[output_source] = decision_result
                
                # Add to cascade results
                cascade_results.append(decision_result)
            
            else:
                self.logger.warning(f"Unknown step type for step {step_id}: {step_type}")
        
        # Get final output
        final_output_source = self.config.get('final_output', 'fused')
        final_output = current_features.get(final_output_source, {})
        
        # Add metadata
        final_output['fusion_time'] = datetime.now().isoformat()
        final_output['cascade'] = self.config['cascade']
        final_output['decision_threshold'] = self.config['decision_threshold']
        final_output['cascade_results'] = cascade_results
        
        self.logger.info("Cascaded fusion completed")
        return final_output
    
    def _apply_transform(self, transform_type: str, features: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a transform to features.
        
        Args:
            transform_type: Type of transform to apply
            features: Features to transform
            config: Transform configuration
            
        Returns:
            Dictionary containing the transformed features
        """
        if transform_type == 'normalization':
            # Apply normalization
            method = config.get('method', 'min_max')
            return self._normalize_features(features, method)
        
        elif transform_type == 'selection':
            # Apply feature selection
            method = config.get('method', 'threshold')
            threshold = config.get('threshold', 0.5)
            top_k = config.get('top_k', 10)
            return self._select_features(features, method, threshold, top_k)
        
        elif transform_type == 'aggregation':
            # Apply feature aggregation
            method = config.get('method', 'mean')
            return self._aggregate_features(features, method)
        
        elif transform_type == 'filtering':
            # Apply feature filtering
            filter_type = config.get('filter', 'threshold')
            threshold = config.get('threshold', 0.5)
            return self._filter_features(features, filter_type, threshold)
        
        else:
            self.logger.warning(f"Unknown transform type: {transform_type}")
            return features
    
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
            if isinstance(value, (int, float)) and key not in ['step_id', 'step_type', 'fusion_component']:
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
            if isinstance(value, (int, float)) and key not in ['step_id', 'step_type', 'fusion_component']:
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
        selected['step_id'] = features.get('step_id', '')
        selected['step_type'] = features.get('step_type', '')
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
            if isinstance(value, (int, float)) and key not in ['step_id', 'step_type', 'fusion_component']:
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
        aggregated['step_id'] = features.get('step_id', '')
        aggregated['step_type'] = features.get('step_type', '')
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
                if isinstance(value, (int, float)) and key not in ['step_id', 'step_type', 'fusion_component']:
                    if value < threshold:
                        del filtered[key]
        
        elif filter_type == 'percentile':
            # Filter features based on percentile
            numeric_features = {k: v for k, v in features.items() 
                              if isinstance(v, (int, float)) and k not in ['step_id', 'step_type', 'fusion_component']}
            
            if numeric_features:
                values = list(numeric_features.values())
                percentile_threshold = np.percentile(values, threshold * 100)
                
                for key, value in list(filtered.items()):
                    if isinstance(value, (int, float)) and key not in ['step_id', 'step_type', 'fusion_component']:
                        if value < percentile_threshold:
                            del filtered[key]
        
        # Add metadata
        filtered['filter_type'] = filter_type
        filtered['filter_threshold'] = threshold
        
        return filtered
    
    def _make_decision(self, features: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Make a decision based on features.
        
        Args:
            features: Features to base the decision on
            config: Decision configuration
            
        Returns:
            Tuple of (decision, confidence)
        """
        decision_method = config.get('method', 'threshold')
        threshold = config.get('threshold', self.config['decision_threshold'])
        
        if decision_method == 'threshold':
            # Make decision based on a specific feature
            feature_name = config.get('feature', '')
            
            if feature_name and feature_name in features:
                feature_value = features[feature_name]
                
                if isinstance(feature_value, (int, float)):
                    decision = feature_value >= threshold
                    confidence = abs(feature_value - threshold) / max(1.0, threshold)
                    confidence = min(1.0, confidence)
                    
                    return decision, confidence
            
            # If feature not found, check for common decision indicators
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    if 'probability' in key or 'confidence' in key or 'risk_score' in key:
                        decision = value >= threshold
                        confidence = abs(value - threshold) / max(1.0, threshold)
                        confidence = min(1.0, confidence)
                        
                        return decision, confidence
        
        elif decision_method == 'voting':
            # Make decision based on voting
            positive_votes = 0
            negative_votes = 0
            
            for key, value in features.items():
                if isinstance(value, dict) and 'decision' in value:
                    if value['decision']:
                        positive_votes += 1
                    else:
                        negative_votes += 1
            
            total_votes = positive_votes + negative_votes
            if total_votes > 0:
                decision = positive_votes >= negative_votes
                confidence = abs(positive_votes - negative_votes) / total_votes
                
                return decision, confidence
        
        # Default decision
        return False, 0.5
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # For cascaded fusion, we use the risk score from the final step
        # if it's available, otherwise we calculate it based on the fused features
        
        # Check if there's a risk score already calculated
        if 'risk_score' in fused_features:
            return float(fused_features['risk_score'])
        
        # Check cascade results for decision steps
        cascade_results = fused_features.get('cascade_results', [])
        for step in reversed(cascade_results):  # Start from the last step
            if step.get('step_type') == 'decision':
                decision = step.get('decision', False)
                confidence = step.get('confidence', 0.5)
                
                if decision:
                    return confidence
                else:
                    return 0.1 * (1.0 - confidence)
        
        # If no decision steps found, check for fused probability or confidence
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