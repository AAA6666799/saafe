"""
Mutual information-based feature selection implementation for the synthetic fire prediction system.

This module provides an implementation of mutual information-based feature selection,
which selects features based on their mutual information with the target variable.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from ...base import FeatureFusion


class MutualInformationSelection(FeatureFusion):
    """
    Implementation of mutual information-based feature selection.
    
    This class selects features based on their mutual information with the target variable,
    which measures how much information the feature provides about the target.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the mutual information-based feature selection component.
        
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
        required_params = ['mi_threshold']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate MI threshold
        if not isinstance(self.config['mi_threshold'], (int, float)) or \
           self.config['mi_threshold'] < 0:
            raise ValueError("'mi_threshold' must be a non-negative number")
        
        # Set default values for optional parameters
        if 'target_variable' not in self.config:
            self.config['target_variable'] = 'risk_score'
        
        if 'target_type' not in self.config:
            self.config['target_type'] = 'continuous'
        
        if 'max_features' not in self.config:
            self.config['max_features'] = None
        
        if 'discrete_features' not in self.config:
            self.config['discrete_features'] = False
    
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select features based on mutual information.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the selected features
        """
        self.logger.info("Performing mutual information-based feature selection")
        
        # Convert features to DataFrames
        thermal_df = self._to_dataframe(thermal_features, 'thermal')
        gas_df = self._to_dataframe(gas_features, 'gas')
        env_df = self._to_dataframe(environmental_features, 'environmental')
        
        # Combine all features into a single DataFrame
        all_features_df = pd.concat([thermal_df, gas_df, env_df], axis=1)
        
        # Create or extract target variable
        target_variable = self.config['target_variable']
        
        if target_variable in all_features_df.columns:
            # Use existing column as target
            target = all_features_df[target_variable]
            # Remove target from features
            all_features_df = all_features_df.drop(columns=[target_variable])
        else:
            # Create synthetic target variable based on available features
            target = self._create_synthetic_target(all_features_df)
        
        # Select features based on mutual information
        selected_features = self._select_features_by_mutual_information(all_features_df, target)
        
        # Create result dictionary
        result = {
            'selection_time': datetime.now().isoformat(),
            'mi_threshold': self.config['mi_threshold'],
            'target_variable': target_variable,
            'target_type': self.config['target_type'],
            'original_feature_count': len(all_features_df.columns),
            'selected_feature_count': len(selected_features),
            'selected_features': selected_features,
            'selected_feature_values': {
                feature: all_features_df[feature].iloc[0] if not all_features_df.empty else None
                for feature in selected_features
            }
        }
        
        self.logger.info(f"Selected {len(selected_features)} features out of {len(all_features_df.columns)}")
        return result
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # For feature selection, we calculate the risk score based on the selected features
        selected_feature_values = fused_features.get('selected_feature_values', {})
        
        if not selected_feature_values:
            self.logger.warning("No selected features available for risk score calculation")
            return 0.1
        
        # Calculate risk score based on selected features
        risk_indicators = []
        
        # Check for thermal indicators
        thermal_features = {k: v for k, v in selected_feature_values.items() if k.startswith('thermal_')}
        for key, value in thermal_features.items():
            if 'max_temperature' in key and isinstance(value, (int, float)):
                max_temp = value
                if max_temp > 100:  # Example threshold
                    risk_indicators.append(min(1.0, (max_temp - 100) / 100))
            
            if 'hotspot_count' in key and isinstance(value, (int, float)):
                hotspot_count = value
                if hotspot_count > 3:  # Example threshold
                    risk_indicators.append(min(1.0, hotspot_count / 10))
        
        # Check for gas indicators
        gas_features = {k: v for k, v in selected_feature_values.items() if k.startswith('gas_')}
        for key, value in gas_features.items():
            if 'concentration' in key and isinstance(value, (int, float)):
                concentration = value
                if concentration > 50:  # Example threshold
                    risk_indicators.append(min(1.0, (concentration - 50) / 100))
        
        # Check for environmental indicators
        env_features = {k: v for k, v in selected_feature_values.items() if k.startswith('environmental_')}
        for key, value in env_features.items():
            if 'temperature_rise' in key and isinstance(value, (int, float)):
                temp_rise = value
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
    
    def _to_dataframe(self, features: Dict[str, Any], prefix: str) -> pd.DataFrame:
        """
        Convert features dictionary to a pandas DataFrame with prefixed column names.
        
        Args:
            features: Features dictionary
            prefix: Prefix for column names
            
        Returns:
            DataFrame with prefixed column names
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
        
        # Create DataFrame with prefixed column names
        df = pd.DataFrame({f"{prefix}_{k}": [v] for k, v in flat_dict.items()})
        
        return df
    
    def _create_synthetic_target(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Create a synthetic target variable based on available features.
        
        Args:
            features_df: DataFrame containing features
            
        Returns:
            Series containing the synthetic target variable
        """
        # This is a simplified implementation; in a real system, you would
        # use domain knowledge to create a more meaningful target variable
        
        # Check for known risk indicators
        risk_indicators = []
        
        # Check for thermal indicators
        thermal_cols = [col for col in features_df.columns if col.startswith('thermal_')]
        for col in thermal_cols:
            if 'max_temperature' in col and pd.api.types.is_numeric_dtype(features_df[col]):
                max_temp = features_df[col].iloc[0] if not features_df.empty else 0
                if max_temp > 100:  # Example threshold
                    risk_indicators.append(min(1.0, (max_temp - 100) / 100))
            
            if 'hotspot_count' in col and pd.api.types.is_numeric_dtype(features_df[col]):
                hotspot_count = features_df[col].iloc[0] if not features_df.empty else 0
                if hotspot_count > 3:  # Example threshold
                    risk_indicators.append(min(1.0, hotspot_count / 10))
        
        # Check for gas indicators
        gas_cols = [col for col in features_df.columns if col.startswith('gas_')]
        for col in gas_cols:
            if 'concentration' in col and pd.api.types.is_numeric_dtype(features_df[col]):
                concentration = features_df[col].iloc[0] if not features_df.empty else 0
                if concentration > 50:  # Example threshold
                    risk_indicators.append(min(1.0, (concentration - 50) / 100))
        
        # Check for environmental indicators
        env_cols = [col for col in features_df.columns if col.startswith('environmental_')]
        for col in env_cols:
            if 'temperature_rise' in col and pd.api.types.is_numeric_dtype(features_df[col]):
                temp_rise = features_df[col].iloc[0] if not features_df.empty else 0
                if temp_rise > 5:  # Example threshold
                    risk_indicators.append(min(1.0, temp_rise / 20))
        
        # Calculate synthetic target
        if risk_indicators:
            # Use a weighted average of the top 3 risk indicators
            risk_indicators.sort(reverse=True)
            top_indicators = risk_indicators[:3]
            weights = [0.5, 0.3, 0.2][:len(top_indicators)]
            
            synthetic_target = sum(indicator * weight for indicator, weight in zip(top_indicators, weights))
        else:
            # If no risk indicators are available, use a default low risk score
            synthetic_target = 0.1
        
        # For classification, convert to binary target
        if self.config['target_type'] == 'discrete':
            threshold = self.config.get('classification_threshold', 0.5)
            synthetic_target = 1 if synthetic_target >= threshold else 0
        
        return pd.Series([synthetic_target])
    
    def _select_features_by_mutual_information(self, features_df: pd.DataFrame, target: pd.Series) -> List[str]:
        """
        Select features based on mutual information with target.
        
        Args:
            features_df: DataFrame containing features
            target: Series containing the target variable
            
        Returns:
            List of selected feature names
        """
        # Filter numeric columns
        numeric_df = features_df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            self.logger.warning("No numeric features available for mutual information analysis")
            return []
        
        # Handle missing values
        numeric_df = numeric_df.fillna(0)
        
        # Determine if target is discrete or continuous
        target_type = self.config['target_type']
        discrete_features = self.config['discrete_features']
        
        # Calculate mutual information
        try:
            if target_type == 'discrete':
                # For classification
                mi_scores = mutual_info_classif(
                    numeric_df, 
                    target, 
                    discrete_features=discrete_features
                )
            else:
                # For regression
                mi_scores = mutual_info_regression(
                    numeric_df, 
                    target, 
                    discrete_features=discrete_features
                )
            
            # Create dictionary of feature names and MI scores
            mi_dict = dict(zip(numeric_df.columns, mi_scores))
            
            # Sort features by MI score
            sorted_features = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Select features with MI score above threshold
            mi_threshold = self.config['mi_threshold']
            selected_features = [feature for feature, score in sorted_features if score >= mi_threshold]
            
            # Limit number of features if specified
            max_features = self.config['max_features']
            if max_features is not None and len(selected_features) > max_features:
                selected_features = selected_features[:max_features]
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error calculating mutual information: {str(e)}")
            return []