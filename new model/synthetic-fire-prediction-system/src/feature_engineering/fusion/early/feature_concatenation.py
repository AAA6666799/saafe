"""
Feature concatenation implementation for the synthetic fire prediction system.

This module provides an implementation of feature concatenation, which combines
features from different sources by concatenating them into a single feature vector.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import logging
from datetime import datetime

from ...base import FeatureFusion


class FeatureConcatenation(FeatureFusion):
    """
    Implementation of feature concatenation.
    
    This class combines features from different sources (thermal, gas, environmental)
    by concatenating them into a single feature vector.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature concatenation component.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required parameters
        required_params = ['normalization', 'feature_selection']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate normalization
        valid_normalizations = ['min_max', 'z_score', 'none']
        if self.config['normalization'] not in valid_normalizations:
            raise ValueError(f"Invalid normalization: {self.config['normalization']}. "
                           f"Must be one of {valid_normalizations}")
        
        # Validate feature selection
        valid_selections = ['all', 'top_k', 'threshold', 'none']
        if self.config['feature_selection'] not in valid_selections:
            raise ValueError(f"Invalid feature selection: {self.config['feature_selection']}. "
                           f"Must be one of {valid_selections}")
        
        # Set default values for optional parameters
        if 'top_k' not in self.config:
            self.config['top_k'] = 10
        
        if 'threshold' not in self.config:
            self.config['threshold'] = 0.5
        
        if 'include_metadata' not in self.config:
            self.config['include_metadata'] = True
    
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse features from different extractors by concatenation.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the fused features
        """
        self.logger.info("Performing feature concatenation")
        
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
        
        # Concatenate features
        concatenated_df = pd.concat([thermal_df, gas_df, env_df], axis=1)
        
        # Apply feature selection if required
        if self.config['feature_selection'] != 'none':
            concatenated_df = self._select_features(concatenated_df)
        
        # Create metadata
        metadata = {
            'fusion_time': datetime.now().isoformat(),
            'feature_counts': {
                'thermal': len(thermal_df.columns),
                'gas': len(gas_df.columns),
                'environmental': len(env_df.columns),
                'total': len(concatenated_df.columns)
            },
            'normalization': self.config['normalization'],
            'feature_selection': self.config['feature_selection']
        }
        
        # Create fused features dictionary
        fused_features = {
            'concatenated_features': concatenated_df.to_dict(orient='records')[0] if not concatenated_df.empty else {},
            'feature_names': list(concatenated_df.columns)
        }
        
        # Include metadata if configured
        if self.config.get('include_metadata', True):
            fused_features.update(metadata)
        
        self.logger.info(f"Feature concatenation completed with {metadata['feature_counts']['total']} features")
        return fused_features
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # Extract concatenated features
        concatenated_features = fused_features.get('concatenated_features', {})
        
        if not concatenated_features:
            self.logger.warning("No concatenated features available for risk score calculation")
            return 0.0
        
        # Define risk indicators based on feature patterns
        risk_indicators = []
        
        # Check thermal risk indicators
        thermal_keys = [k for k in concatenated_features.keys() if k.startswith('thermal_')]
        for key in thermal_keys:
            if 'max_temperature' in key and isinstance(concatenated_features[key], (int, float)):
                max_temp = concatenated_features[key]
                # Higher temperatures indicate higher risk
                if max_temp > 100:  # Example threshold
                    risk_indicators.append(min(1.0, (max_temp - 100) / 100))
            
            if 'hotspot_count' in key and isinstance(concatenated_features[key], (int, float)):
                hotspot_count = concatenated_features[key]
                # More hotspots indicate higher risk
                if hotspot_count > 3:  # Example threshold
                    risk_indicators.append(min(1.0, hotspot_count / 10))
        
        # Check gas risk indicators
        gas_keys = [k for k in concatenated_features.keys() if k.startswith('gas_')]
        for key in gas_keys:
            if 'concentration' in key and isinstance(concatenated_features[key], (int, float)):
                concentration = concatenated_features[key]
                # Higher gas concentrations indicate higher risk
                if concentration > 50:  # Example threshold
                    risk_indicators.append(min(1.0, (concentration - 50) / 100))
        
        # Check environmental risk indicators
        env_keys = [k for k in concatenated_features.keys() if k.startswith('environmental_')]
        for key in env_keys:
            if 'temperature_rise' in key and isinstance(concatenated_features[key], (int, float)):
                temp_rise = concatenated_features[key]
                # Rapid temperature rise indicates higher risk
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
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select features based on the specified selection method.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            DataFrame with selected features
        """
        selection_method = self.config['feature_selection']
        
        if selection_method == 'all':
            # Keep all features
            return df
        
        elif selection_method == 'top_k':
            # Select top k features with highest variance
            k = min(self.config['top_k'], len(df.columns))
            
            # Calculate variance for each feature
            variances = df.var().sort_values(ascending=False)
            
            # Select top k features
            selected_features = variances.index[:k]
            
            return df[selected_features]
        
        elif selection_method == 'threshold':
            # Select features with variance above threshold
            threshold = self.config['threshold']
            
            # Calculate variance for each feature
            variances = df.var()
            
            # Select features with variance above threshold
            selected_features = variances[variances > threshold].index
            
            return df[selected_features]
        
        else:
            self.logger.warning(f"Unknown feature selection method: {selection_method}, keeping all features")
            return df