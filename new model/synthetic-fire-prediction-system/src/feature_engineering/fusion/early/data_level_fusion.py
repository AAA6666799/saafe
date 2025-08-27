"""
Data-level fusion implementation for the synthetic fire prediction system.

This module provides an implementation of data-level fusion, which combines
raw data from different sources before feature extraction.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import logging
from datetime import datetime

from ...base import FeatureFusion


class DataLevelFusion(FeatureFusion):
    """
    Implementation of data-level fusion.
    
    This class combines raw data from different sources (thermal, gas, environmental)
    before feature extraction to create a unified data representation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data-level fusion component.
        
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
        required_params = ['fusion_method', 'normalization']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate fusion method
        valid_methods = ['concatenation', 'averaging', 'weighted']
        if self.config['fusion_method'] not in valid_methods:
            raise ValueError(f"Invalid fusion method: {self.config['fusion_method']}. "
                           f"Must be one of {valid_methods}")
        
        # Validate normalization
        valid_normalizations = ['min_max', 'z_score', 'none']
        if self.config['normalization'] not in valid_normalizations:
            raise ValueError(f"Invalid normalization: {self.config['normalization']}. "
                           f"Must be one of {valid_normalizations}")
        
        # Set default values for optional parameters
        if 'weights' not in self.config:
            self.config['weights'] = {
                'thermal': 1.0,
                'gas': 1.0,
                'environmental': 1.0
            }
    
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse features from different extractors at the data level.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the fused features
        """
        self.logger.info("Performing data-level fusion")
        
        # Convert features to DataFrames if they're not already
        thermal_df = self._to_dataframe(thermal_features, 'thermal')
        gas_df = self._to_dataframe(gas_features, 'gas')
        env_df = self._to_dataframe(environmental_features, 'environmental')
        
        # Normalize data if required
        if self.config['normalization'] != 'none':
            thermal_df = self._normalize_data(thermal_df)
            gas_df = self._normalize_data(gas_df)
            env_df = self._normalize_data(env_df)
        
        # Perform fusion based on the selected method
        fusion_method = self.config['fusion_method']
        
        if fusion_method == 'concatenation':
            fused_data = self._concatenate_data(thermal_df, gas_df, env_df)
        elif fusion_method == 'averaging':
            fused_data = self._average_data(thermal_df, gas_df, env_df)
        elif fusion_method == 'weighted':
            fused_data = self._weighted_combine_data(thermal_df, gas_df, env_df)
        else:
            self.logger.warning(f"Unknown fusion method: {fusion_method}, using concatenation")
            fused_data = self._concatenate_data(thermal_df, gas_df, env_df)
        
        # Convert fused data back to dictionary
        fused_features = {
            'fusion_method': fusion_method,
            'normalization': self.config['normalization'],
            'fusion_time': datetime.now().isoformat(),
            'fused_data': fused_data.to_dict(orient='records')[0] if not fused_data.empty else {},
            'feature_count': len(fused_data.columns),
            'source_feature_counts': {
                'thermal': len(thermal_df.columns),
                'gas': len(gas_df.columns),
                'environmental': len(env_df.columns)
            }
        }
        
        self.logger.info(f"Data-level fusion completed with {fused_features['feature_count']} features")
        return fused_features
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # Extract fused data
        fused_data = fused_features.get('fused_data', {})
        
        if not fused_data:
            self.logger.warning("No fused data available for risk score calculation")
            return 0.0
        
        # Define risk indicators based on feature patterns
        risk_indicators = []
        
        # Check thermal risk indicators
        if 'thermal_max_max_temperature' in fused_data:
            max_temp = fused_data['thermal_max_max_temperature']
            # Higher temperatures indicate higher risk
            if max_temp > 100:  # Example threshold
                risk_indicators.append(min(1.0, (max_temp - 100) / 100))
        
        # Check gas risk indicators
        if 'gas_max_concentration' in fused_data:
            max_conc = fused_data['gas_max_concentration']
            # Higher gas concentrations indicate higher risk
            if max_conc > 50:  # Example threshold
                risk_indicators.append(min(1.0, (max_conc - 50) / 100))
        
        # Check environmental risk indicators
        if 'environmental_temperature_rise' in fused_data:
            temp_rise = fused_data['environmental_temperature_rise']
            # Rapid temperature rise indicates higher risk
            if temp_rise > 5:  # Example threshold
                risk_indicators.append(min(1.0, temp_rise / 20))
        
        # Calculate overall risk score
        if risk_indicators:
            # Use the maximum risk indicator as the overall risk score
            risk_score = max(risk_indicators)
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
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data using the specified normalization method.
        
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
    
    def _concatenate_data(self, thermal_df: pd.DataFrame, gas_df: pd.DataFrame, env_df: pd.DataFrame) -> pd.DataFrame:
        """
        Concatenate data from different sources.
        
        Args:
            thermal_df: Thermal features DataFrame
            gas_df: Gas features DataFrame
            env_df: Environmental features DataFrame
            
        Returns:
            Concatenated DataFrame
        """
        # Concatenate horizontally (along columns)
        return pd.concat([thermal_df, gas_df, env_df], axis=1)
    
    def _average_data(self, thermal_df: pd.DataFrame, gas_df: pd.DataFrame, env_df: pd.DataFrame) -> pd.DataFrame:
        """
        Average data from different sources.
        
        Args:
            thermal_df: Thermal features DataFrame
            gas_df: Gas features DataFrame
            env_df: Environmental features DataFrame
            
        Returns:
            Averaged DataFrame
        """
        # This is a simplified implementation that assumes the features are comparable
        # In a real system, you would need to ensure the features are semantically compatible
        
        # Create a new DataFrame for the averaged features
        averaged_df = pd.DataFrame()
        
        # Get all numeric columns from each DataFrame
        thermal_numeric = thermal_df.select_dtypes(include=[np.number])
        gas_numeric = gas_df.select_dtypes(include=[np.number])
        env_numeric = env_df.select_dtypes(include=[np.number])
        
        # Calculate the average of each feature across all sources
        for col in thermal_numeric.columns:
            averaged_df[f"avg_{col}"] = thermal_numeric[col]
        
        for col in gas_numeric.columns:
            averaged_df[f"avg_{col}"] = gas_numeric[col]
        
        for col in env_numeric.columns:
            averaged_df[f"avg_{col}"] = env_numeric[col]
        
        return averaged_df
    
    def _weighted_combine_data(self, thermal_df: pd.DataFrame, gas_df: pd.DataFrame, env_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine data from different sources using weights.
        
        Args:
            thermal_df: Thermal features DataFrame
            gas_df: Gas features DataFrame
            env_df: Environmental features DataFrame
            
        Returns:
            Weighted combined DataFrame
        """
        # Get weights from config
        weights = self.config['weights']
        thermal_weight = weights.get('thermal', 1.0)
        gas_weight = weights.get('gas', 1.0)
        env_weight = weights.get('environmental', 1.0)
        
        # Create a new DataFrame for the weighted features
        weighted_df = pd.DataFrame()
        
        # Get all numeric columns from each DataFrame
        thermal_numeric = thermal_df.select_dtypes(include=[np.number])
        gas_numeric = gas_df.select_dtypes(include=[np.number])
        env_numeric = env_df.select_dtypes(include=[np.number])
        
        # Apply weights to each feature
        for col in thermal_numeric.columns:
            weighted_df[f"weighted_{col}"] = thermal_numeric[col] * thermal_weight
        
        for col in gas_numeric.columns:
            weighted_df[f"weighted_{col}"] = gas_numeric[col] * gas_weight
        
        for col in env_numeric.columns:
            weighted_df[f"weighted_{col}"] = env_numeric[col] * env_weight
        
        return weighted_df