"""
Hierarchical fusion implementation for the synthetic fire prediction system.

This module provides an implementation of hierarchical fusion, which combines
features from different sources using a hierarchical approach.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import importlib

from ...base import FeatureFusion


class HierarchicalFusion(FeatureFusion):
    """
    Implementation of hierarchical fusion.
    
    This class combines features from different sources (thermal, gas, environmental)
    using a hierarchical approach, where features are fused at multiple levels in a
    hierarchical structure.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hierarchical fusion component.
        
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
        required_params = ['hierarchy', 'fusion_components']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate hierarchy
        if not isinstance(self.config['hierarchy'], dict):
            raise ValueError("'hierarchy' must be a dictionary")
        
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
        Fuse features from different extractors using a hierarchical approach.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the fused features
        """
        self.logger.info("Performing hierarchical fusion")
        
        # Initialize feature dictionary with input features
        all_features = {
            'thermal': thermal_features,
            'gas': gas_features,
            'environmental': environmental_features
        }
        
        # Process the hierarchy
        root_node = self.config['hierarchy'].get('root', '')
        if not root_node:
            self.logger.error("No root node specified in hierarchy")
            return {'error': 'No root node specified in hierarchy'}
        
        # Process the hierarchy starting from the root node
        fused_features = self._process_hierarchy_node(root_node, all_features)
        
        # Add metadata
        fused_features['fusion_time'] = datetime.now().isoformat()
        fused_features['hierarchy'] = self.config['hierarchy']
        fused_features['decision_threshold'] = self.config['decision_threshold']
        
        self.logger.info("Hierarchical fusion completed")
        return fused_features
    
    def _process_hierarchy_node(self, node_id: str, all_features: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a node in the hierarchy.
        
        Args:
            node_id: ID of the node to process
            all_features: Dictionary of all available features
            
        Returns:
            Dictionary containing the processed features
        """
        # Get node configuration
        node_config = self.config['hierarchy'].get(node_id, {})
        if not node_config:
            self.logger.warning(f"No configuration found for node: {node_id}")
            return {}
        
        # Get node type
        node_type = node_config.get('type', '')
        
        if node_type == 'fusion':
            # This is a fusion node
            fusion_component_id = node_config.get('component', '')
            if not fusion_component_id or fusion_component_id not in self.fusion_components:
                self.logger.warning(f"Invalid fusion component for node {node_id}: {fusion_component_id}")
                return {}
            
            # Get input nodes
            input_nodes = node_config.get('inputs', [])
            if not input_nodes:
                self.logger.warning(f"No input nodes specified for fusion node: {node_id}")
                return {}
            
            # Process input nodes
            input_features = {}
            for input_id in input_nodes:
                if input_id in all_features:
                    # This is a direct input (thermal, gas, environmental)
                    input_features[input_id] = all_features[input_id]
                else:
                    # This is another node in the hierarchy
                    processed_features = self._process_hierarchy_node(input_id, all_features)
                    input_features[input_id] = processed_features
            
            # Perform fusion
            fusion_component = self.fusion_components[fusion_component_id]
            
            # Extract the required features for the fusion component
            thermal_input = input_features.get('thermal', {})
            gas_input = input_features.get('gas', {})
            env_input = input_features.get('environmental', {})
            
            # For other inputs, use the first available as thermal, second as gas, third as environmental
            other_inputs = [v for k, v in input_features.items() if k not in ['thermal', 'gas', 'environmental']]
            if other_inputs:
                if not thermal_input and len(other_inputs) > 0:
                    thermal_input = other_inputs[0]
                if not gas_input and len(other_inputs) > 1:
                    gas_input = other_inputs[1]
                if not env_input and len(other_inputs) > 2:
                    env_input = other_inputs[2]
            
            # Perform fusion
            fused = fusion_component.fuse_features(thermal_input, gas_input, env_input)
            
            # Add node metadata
            fused['node_id'] = node_id
            fused['node_type'] = node_type
            fused['fusion_component'] = fusion_component_id
            
            return fused
        
        elif node_type == 'feature':
            # This is a feature node (direct pass-through)
            feature_source = node_config.get('source', '')
            if not feature_source or feature_source not in all_features:
                self.logger.warning(f"Invalid feature source for node {node_id}: {feature_source}")
                return {}
            
            # Get the features
            features = all_features[feature_source]
            
            # Add node metadata
            features_with_metadata = features.copy()
            features_with_metadata['node_id'] = node_id
            features_with_metadata['node_type'] = node_type
            features_with_metadata['feature_source'] = feature_source
            
            return features_with_metadata
        
        elif node_type == 'transform':
            # This is a transform node
            transform_type = node_config.get('transform', '')
            if not transform_type:
                self.logger.warning(f"No transform type specified for node: {node_id}")
                return {}
            
            # Get input node
            input_node = node_config.get('input', '')
            if not input_node:
                self.logger.warning(f"No input node specified for transform node: {node_id}")
                return {}
            
            # Process input node
            if input_node in all_features:
                # This is a direct input (thermal, gas, environmental)
                input_features = all_features[input_node]
            else:
                # This is another node in the hierarchy
                input_features = self._process_hierarchy_node(input_node, all_features)
            
            # Apply transform
            transformed = self._apply_transform(transform_type, input_features, node_config)
            
            # Add node metadata
            transformed['node_id'] = node_id
            transformed['node_type'] = node_type
            transformed['transform_type'] = transform_type
            
            return transformed
        
        else:
            self.logger.warning(f"Unknown node type for node {node_id}: {node_type}")
            return {}
    
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
            if isinstance(value, (int, float)) and key not in ['node_id', 'node_type', 'feature_source']:
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
            if isinstance(value, (int, float)) and key not in ['node_id', 'node_type', 'feature_source']:
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
        selected['node_id'] = features.get('node_id', '')
        selected['node_type'] = features.get('node_type', '')
        selected['feature_source'] = features.get('feature_source', '')
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
            if isinstance(value, (int, float)) and key not in ['node_id', 'node_type', 'feature_source']:
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
        aggregated['node_id'] = features.get('node_id', '')
        aggregated['node_type'] = features.get('node_type', '')
        aggregated['feature_source'] = features.get('feature_source', '')
        aggregated['aggregation_method'] = method
        
        return aggregated
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # For hierarchical fusion, we use the risk score from the root node
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