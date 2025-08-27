"""
Weighted feature combination implementation for the synthetic fire prediction system.

This module provides an implementation of weighted feature combination, which combines
features from different sources using learned weights.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from collections import defaultdict
import json
import os

from ...base import FeatureFusion


class WeightedFeatureCombination(FeatureFusion):
    """
    Implementation of weighted feature combination.
    
    This class combines features from different sources (thermal, gas, environmental)
    using learned weights to create a unified representation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the weighted feature combination component.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.weights = {}
        self._load_weights()
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required parameters
        required_params = ['normalization', 'weight_source']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate normalization
        valid_normalizations = ['min_max', 'z_score', 'none']
        if self.config['normalization'] not in valid_normalizations:
            raise ValueError(f"Invalid normalization: {self.config['normalization']}. "
                           f"Must be one of {valid_normalizations}")
        
        # Validate weight source
        valid_sources = ['config', 'file', 'auto']
        if self.config['weight_source'] not in valid_sources:
            raise ValueError(f"Invalid weight source: {self.config['weight_source']}. "
                           f"Must be one of {valid_sources}")
        
        # Set default values for optional parameters
        if 'default_weight' not in self.config:
            self.config['default_weight'] = 1.0
        
        if 'weights_file' not in self.config:
            self.config['weights_file'] = 'weights.json'
        
        if 'auto_weight_method' not in self.config:
            self.config['auto_weight_method'] = 'variance'
    
    def _load_weights(self) -> None:
        """
        Load feature weights based on the configured weight source.
        """
        weight_source = self.config['weight_source']
        
        if weight_source == 'config':
            # Load weights from config
            self.weights = self.config.get('weights', {})
            self.logger.info(f"Loaded {len(self.weights)} weights from config")
        
        elif weight_source == 'file':
            # Load weights from file
            weights_file = self.config['weights_file']
            if os.path.exists(weights_file):
                try:
                    with open(weights_file, 'r') as f:
                        self.weights = json.load(f)
                    self.logger.info(f"Loaded {len(self.weights)} weights from file: {weights_file}")
                except Exception as e:
                    self.logger.error(f"Error loading weights from file: {str(e)}")
                    self.weights = {}
            else:
                self.logger.warning(f"Weights file not found: {weights_file}")
                self.weights = {}
        
        elif weight_source == 'auto':
            # Weights will be automatically calculated during fusion
            self.logger.info("Weights will be automatically calculated during fusion")
            self.weights = {}
        
        else:
            self.logger.warning(f"Unknown weight source: {weight_source}")
            self.weights = {}
    
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse features from different extractors using weighted combination.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the fused features
        """
        self.logger.info("Performing weighted feature combination")
        
        # Extract feature values from each source
        thermal_values = self._extract_feature_values(thermal_features, 'thermal')
        gas_values = self._extract_feature_values(gas_features, 'gas')
        env_values = self._extract_feature_values(environmental_features, 'environmental')
        
        # Create DataFrames for each source
        thermal_df = pd.DataFrame([thermal_values])
        gas_df = pd.DataFrame([gas_values])
        env_df = pd.DataFrame([env_values])
        
        # Normalize features if required
        if self.config['normalization'] != 'none':
            thermal_df = self._normalize_features(thermal_df)
            gas_df = self._normalize_features(gas_df)
            env_df = self._normalize_features(env_df)
        
        # Combine all features into a single DataFrame
        all_features_df = pd.concat([thermal_df, gas_df, env_df], axis=1)
        
        # Calculate weights if using auto weighting
        if self.config['weight_source'] == 'auto':
            self._calculate_auto_weights(all_features_df)
        
        # Apply weights to features
        weighted_features = self._apply_weights(all_features_df)
        
        # Create metadata
        metadata = {
            'fusion_time': datetime.now().isoformat(),
            'feature_counts': {
                'thermal': len(thermal_df.columns),
                'gas': len(gas_df.columns),
                'environmental': len(env_df.columns),
                'total': len(all_features_df.columns),
                'weighted': len(weighted_features)
            },
            'normalization': self.config['normalization'],
            'weight_source': self.config['weight_source'],
            'weight_count': len(self.weights)
        }
        
        # Create fused features dictionary
        fused_features = {
            'weighted_features': weighted_features,
            'feature_names': list(weighted_features.keys()),
            'weights': self.weights
        }
        
        # Include metadata if configured
        if self.config.get('include_metadata', True):
            fused_features.update(metadata)
        
        self.logger.info(f"Weighted feature combination completed with {metadata['feature_counts']['weighted']} features")
        return fused_features
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # Extract weighted features
        weighted_features = fused_features.get('weighted_features', {})
        
        if not weighted_features:
            self.logger.warning("No weighted features available for risk score calculation")
            return 0.0
        
        # Define risk indicators based on feature patterns
        risk_indicators = []
        
        # Check thermal risk indicators
        thermal_keys = [k for k in weighted_features.keys() if k.startswith('thermal_')]
        for key in thermal_keys:
            if 'max_temperature' in key and isinstance(weighted_features[key], (int, float)):
                max_temp = weighted_features[key]
                # Higher temperatures indicate higher risk
                if max_temp > 100:  # Example threshold
                    risk_indicators.append(min(1.0, (max_temp - 100) / 100))
            
            if 'hotspot_count' in key and isinstance(weighted_features[key], (int, float)):
                hotspot_count = weighted_features[key]
                # More hotspots indicate higher risk
                if hotspot_count > 3:  # Example threshold
                    risk_indicators.append(min(1.0, hotspot_count / 10))
        
        # Check gas risk indicators
        gas_keys = [k for k in weighted_features.keys() if k.startswith('gas_')]
        for key in gas_keys:
            if 'concentration' in key and isinstance(weighted_features[key], (int, float)):
                concentration = weighted_features[key]
                # Higher gas concentrations indicate higher risk
                if concentration > 50:  # Example threshold
                    risk_indicators.append(min(1.0, (concentration - 50) / 100))
        
        # Check environmental risk indicators
        env_keys = [k for k in weighted_features.keys() if k.startswith('environmental_')]
        for key in env_keys:
            if 'temperature_rise' in key and isinstance(weighted_features[key], (int, float)):
                temp_rise = weighted_features[key]
                # Rapid temperature rise indicates higher risk
                if temp_rise > 5:  # Example threshold
                    risk_indicators.append(min(1.0, temp_rise / 20))
        
        # Calculate overall risk score using weighted sum of risk indicators
        if risk_indicators:
            # Get weights for risk indicators
            weights = []
            for i in range(len(risk_indicators)):
                weights.append(0.5 if i == 0 else (0.3 if i == 1 else 0.2))
            
            # Normalize weights to sum to 1
            weights = [w / sum(weights) for w in weights]
            
            # Calculate weighted sum
            risk_score = sum(indicator * weight for indicator, weight in zip(risk_indicators, weights))
        else:
            # If no risk indicators are available, use a default low risk score
            risk_score = 0.1
        
        self.logger.info(f"Calculated risk score: {risk_score}")
        return risk_score
    
    def _extract_feature_values(self, features: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """
        Extract feature values from a features dictionary.
        
        Args:
            features: Features dictionary
            prefix: Prefix for feature names
            
        Returns:
            Dictionary of feature values with prefixed keys
        """
        # Flatten nested dictionary
        flat_dict = {}
        
        def flatten(d, parent_key=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    flatten(v, f"{parent_key}{k}_")
                elif isinstance(v, (list, tuple)) and len(v) > 0 and not isinstance(v[0], dict):
                    # For lists of simple types, use the first element
                    flat_dict[f"{parent_key}{k}"] = v[0] if v else None
                elif not isinstance(v, (list, tuple)):
                    flat_dict[f"{parent_key}{k}"] = v
        
        flatten(features)
        
        # Add prefix to keys
        return {f"{prefix}_{k}": v for k, v in flat_dict.items()}
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using the specified normalization method.
        
        Args:
            df: DataFrame to normalize
            
        Returns:
            Normalized DataFrame
        """
        norm_method = self.config['normalization']
        
        if norm_method == 'min_max':
            # Min-max normalization to [0, 1] range
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif norm_method == 'z_score':
            # Z-score normalization
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:
                        df[col] = (df[col] - mean) / std
        
        return df
    
    def _calculate_auto_weights(self, df: pd.DataFrame) -> None:
        """
        Calculate weights automatically based on the specified method.
        
        Args:
            df: DataFrame containing all features
        """
        method = self.config.get('auto_weight_method', 'variance')
        
        if method == 'variance':
            # Calculate weights based on feature variance
            # Features with higher variance get higher weights
            variances = df.var()
            total_variance = variances.sum()
            
            if total_variance > 0:
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        self.weights[col] = float(variances[col] / total_variance)
                    else:
                        self.weights[col] = self.config.get('default_weight', 1.0)
            else:
                # If total variance is zero, use default weight for all features
                for col in df.columns:
                    self.weights[col] = self.config.get('default_weight', 1.0)
        
        elif method == 'equal':
            # Equal weights for all features
            for col in df.columns:
                self.weights[col] = 1.0 / len(df.columns)
        
        elif method == 'source':
            # Weights based on source type
            source_weights = self.config.get('source_weights', {
                'thermal': 0.4,
                'gas': 0.4,
                'environmental': 0.2
            })
            
            # Count features by source
            source_counts = {
                'thermal': len([col for col in df.columns if col.startswith('thermal_')]),
                'gas': len([col for col in df.columns if col.startswith('gas_')]),
                'environmental': len([col for col in df.columns if col.startswith('environmental_')])
            }
            
            # Calculate weights
            for col in df.columns:
                if col.startswith('thermal_'):
                    self.weights[col] = source_weights.get('thermal', 0.33) / max(1, source_counts['thermal'])
                elif col.startswith('gas_'):
                    self.weights[col] = source_weights.get('gas', 0.33) / max(1, source_counts['gas'])
                elif col.startswith('environmental_'):
                    self.weights[col] = source_weights.get('environmental', 0.33) / max(1, source_counts['environmental'])
                else:
                    self.weights[col] = self.config.get('default_weight', 1.0) / len(df.columns)
        
        else:
            self.logger.warning(f"Unknown auto weight method: {method}, using equal weights")
            # Equal weights for all features
            for col in df.columns:
                self.weights[col] = 1.0 / len(df.columns)
        
        self.logger.info(f"Calculated {len(self.weights)} weights using method: {method}")
    
    def _apply_weights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply weights to features.
        
        Args:
            df: DataFrame containing all features
            
        Returns:
            Dictionary of weighted features
        """
        weighted_features = {}
        default_weight = self.config.get('default_weight', 1.0)
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Apply weight to numeric features
                weight = self.weights.get(col, default_weight)
                weighted_features[col] = float(df[col].iloc[0] * weight) if not df.empty else 0.0
            else:
                # Non-numeric features are included without weighting
                weighted_features[col] = df[col].iloc[0] if not df.empty else None
        
        return weighted_features
    
    def save_weights(self, filepath: str) -> None:
        """
        Save the current weights to a file.
        
        Args:
            filepath: Path to save the weights
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.weights, f, indent=2)
            self.logger.info(f"Saved {len(self.weights)} weights to file: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving weights to file: {str(e)}")