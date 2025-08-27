"""
Feature averaging implementation for the synthetic fire prediction system.

This module provides an implementation of feature averaging, which combines
features from different sources by averaging them.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from collections import defaultdict

from ...base import FeatureFusion


class FeatureAveraging(FeatureFusion):
    """
    Implementation of feature averaging.
    
    This class combines features from different sources (thermal, gas, environmental)
    by averaging similar features to create a unified representation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature averaging component.
        
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
        required_params = ['feature_mapping', 'normalization']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate normalization
        valid_normalizations = ['min_max', 'z_score', 'none']
        if self.config['normalization'] not in valid_normalizations:
            raise ValueError(f"Invalid normalization: {self.config['normalization']}. "
                           f"Must be one of {valid_normalizations}")
        
        # Validate feature mapping
        if not isinstance(self.config['feature_mapping'], dict):
            raise ValueError("'feature_mapping' must be a dictionary")
        
        # Set default values for optional parameters
        if 'default_strategy' not in self.config:
            self.config['default_strategy'] = 'mean'
        
        if 'include_unmapped' not in self.config:
            self.config['include_unmapped'] = True
    
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse features from different extractors by averaging.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the fused features
        """
        self.logger.info("Performing feature averaging")
        
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
        
        # Group features according to mapping
        feature_groups = self._group_features(thermal_df, gas_df, env_df)
        
        # Average features within each group
        averaged_features = self._average_feature_groups(feature_groups)
        
        # Include unmapped features if configured
        if self.config.get('include_unmapped', True):
            unmapped_features = self._get_unmapped_features(thermal_df, gas_df, env_df, feature_groups)
            averaged_features.update(unmapped_features)
        
        # Create metadata
        metadata = {
            'fusion_time': datetime.now().isoformat(),
            'feature_counts': {
                'thermal': len(thermal_df.columns),
                'gas': len(gas_df.columns),
                'environmental': len(env_df.columns),
                'averaged': len(feature_groups),
                'unmapped': len(unmapped_features) if self.config.get('include_unmapped', True) else 0,
                'total': len(averaged_features)
            },
            'normalization': self.config['normalization'],
            'averaging_strategy': self.config.get('default_strategy', 'mean')
        }
        
        # Create fused features dictionary
        fused_features = {
            'averaged_features': averaged_features,
            'feature_names': list(averaged_features.keys())
        }
        
        # Include metadata if configured
        if self.config.get('include_metadata', True):
            fused_features.update(metadata)
        
        self.logger.info(f"Feature averaging completed with {metadata['feature_counts']['total']} features")
        return fused_features
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # Extract averaged features
        averaged_features = fused_features.get('averaged_features', {})
        
        if not averaged_features:
            self.logger.warning("No averaged features available for risk score calculation")
            return 0.0
        
        # Define risk indicators based on feature patterns
        risk_indicators = []
        
        # Check temperature-related features
        if 'temperature' in averaged_features and isinstance(averaged_features['temperature'], (int, float)):
            temperature = averaged_features['temperature']
            # Higher temperatures indicate higher risk
            if temperature > 80:  # Example threshold
                risk_indicators.append(min(1.0, (temperature - 80) / 100))
        
        # Check hotspot-related features
        if 'hotspot_count' in averaged_features and isinstance(averaged_features['hotspot_count'], (int, float)):
            hotspot_count = averaged_features['hotspot_count']
            # More hotspots indicate higher risk
            if hotspot_count > 3:  # Example threshold
                risk_indicators.append(min(1.0, hotspot_count / 10))
        
        # Check gas-related features
        if 'gas_concentration' in averaged_features and isinstance(averaged_features['gas_concentration'], (int, float)):
            gas_concentration = averaged_features['gas_concentration']
            # Higher gas concentrations indicate higher risk
            if gas_concentration > 50:  # Example threshold
                risk_indicators.append(min(1.0, (gas_concentration - 50) / 100))
        
        # Check rate of change features
        if 'temperature_rise' in averaged_features and isinstance(averaged_features['temperature_rise'], (int, float)):
            temp_rise = averaged_features['temperature_rise']
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
    
    def _group_features(self, thermal_df: pd.DataFrame, gas_df: pd.DataFrame, env_df: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        """
        Group features according to the feature mapping.
        
        Args:
            thermal_df: Thermal features DataFrame
            gas_df: Gas features DataFrame
            env_df: Environmental features DataFrame
            
        Returns:
            Dictionary mapping group names to lists of (feature_name, value) tuples
        """
        feature_mapping = self.config['feature_mapping']
        feature_groups = defaultdict(list)
        
        # Process all DataFrames
        for df, source in [(thermal_df, 'thermal'), (gas_df, 'gas'), (env_df, 'environmental')]:
            for col in df.columns:
                # Check if this feature is mapped
                mapped = False
                for group_name, patterns in feature_mapping.items():
                    if not isinstance(patterns, list):
                        patterns = [patterns]
                    
                    for pattern in patterns:
                        if pattern in col:
                            # Add feature to group with its value
                            value = df[col].iloc[0] if not df.empty else None
                            feature_groups[group_name].append((col, value))
                            mapped = True
                            break
                    
                    if mapped:
                        break
        
        return feature_groups
    
    def _average_feature_groups(self, feature_groups: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Any]:
        """
        Average features within each group.
        
        Args:
            feature_groups: Dictionary mapping group names to lists of (feature_name, value) tuples
            
        Returns:
            Dictionary of averaged features
        """
        averaged_features = {}
        strategy = self.config.get('default_strategy', 'mean')
        
        for group_name, features in feature_groups.items():
            # Extract values, filtering out None and non-numeric values
            values = [value for _, value in features if value is not None and isinstance(value, (int, float))]
            
            if not values:
                continue
            
            # Apply averaging strategy
            if strategy == 'mean':
                averaged_features[group_name] = np.mean(values)
            elif strategy == 'median':
                averaged_features[group_name] = np.median(values)
            elif strategy == 'max':
                averaged_features[group_name] = np.max(values)
            elif strategy == 'min':
                averaged_features[group_name] = np.min(values)
            else:
                self.logger.warning(f"Unknown averaging strategy: {strategy}, using mean")
                averaged_features[group_name] = np.mean(values)
        
        return averaged_features
    
    def _get_unmapped_features(self, thermal_df: pd.DataFrame, gas_df: pd.DataFrame, env_df: pd.DataFrame, 
                             feature_groups: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Any]:
        """
        Get features that were not mapped to any group.
        
        Args:
            thermal_df: Thermal features DataFrame
            gas_df: Gas features DataFrame
            env_df: Environmental features DataFrame
            feature_groups: Dictionary of feature groups
            
        Returns:
            Dictionary of unmapped features
        """
        # Get all mapped feature names
        mapped_features = set()
        for features in feature_groups.values():
            mapped_features.update(name for name, _ in features)
        
        # Get all feature names
        all_features = set()
        for df in [thermal_df, gas_df, env_df]:
            all_features.update(df.columns)
        
        # Get unmapped features
        unmapped_features = all_features - mapped_features
        
        # Create dictionary of unmapped features with their values
        unmapped_dict = {}
        for feature in unmapped_features:
            if feature in thermal_df.columns:
                unmapped_dict[feature] = thermal_df[feature].iloc[0] if not thermal_df.empty else None
            elif feature in gas_df.columns:
                unmapped_dict[feature] = gas_df[feature].iloc[0] if not gas_df.empty else None
            elif feature in env_df.columns:
                unmapped_dict[feature] = env_df[feature].iloc[0] if not env_df.empty else None
        
        return unmapped_dict