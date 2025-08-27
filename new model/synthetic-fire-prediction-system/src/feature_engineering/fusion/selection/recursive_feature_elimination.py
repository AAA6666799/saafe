"""
Recursive Feature Elimination implementation for the synthetic fire prediction system.

This module provides an implementation of Recursive Feature Elimination (RFE),
which recursively removes features and builds a model using the remaining features.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from ...base import FeatureFusion


class RecursiveFeatureElimination(FeatureFusion):
    """
    Implementation of Recursive Feature Elimination (RFE).
    
    This class selects features by recursively considering smaller and smaller sets of features,
    pruning the least important features at each step based on a model's feature importance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RFE component.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.rfe = None
        self.feature_names = []
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required parameters
        required_params = ['n_features_to_select']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate n_features_to_select
        n_features = self.config['n_features_to_select']
        if not isinstance(n_features, (int, float)) or \
           (isinstance(n_features, int) and n_features < 1) or \
           (isinstance(n_features, float) and (n_features <= 0 or n_features > 1)):
            raise ValueError("'n_features_to_select' must be either a positive integer or a float between 0 and 1")
        
        # Set default values for optional parameters
        if 'estimator_type' not in self.config:
            self.config['estimator_type'] = 'random_forest'
        
        if 'target_type' not in self.config:
            self.config['target_type'] = 'continuous'
        
        if 'target_variable' not in self.config:
            self.config['target_variable'] = 'risk_score'
        
        if 'step' not in self.config:
            self.config['step'] = 1
    
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the selected features
        """
        self.logger.info("Performing Recursive Feature Elimination")
        
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
        
        # Filter numeric columns
        numeric_df = all_features_df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            self.logger.warning("No numeric features available for RFE")
            return {'error': 'No numeric features available for RFE'}
        
        # Store original feature names
        self.feature_names = numeric_df.columns.tolist()
        
        # Handle missing values
        numeric_df = numeric_df.fillna(0)
        
        # Apply RFE
        selected_features, ranking, support = self._apply_rfe(numeric_df, target)
        
        # Get selected feature values
        selected_feature_values = {}
        for feature in selected_features:
            if feature in all_features_df.columns:
                selected_feature_values[feature] = all_features_df[feature].iloc[0] if not all_features_df.empty else None
        
        # Create result dictionary
        result = {
            'selection_time': datetime.now().isoformat(),
            'n_features_to_select': self.config['n_features_to_select'],
            'estimator_type': self.config['estimator_type'],
            'target_type': self.config['target_type'],
            'target_variable': target_variable,
            'original_feature_count': len(self.feature_names),
            'selected_feature_count': len(selected_features),
            'selected_features': selected_features,
            'feature_ranking': {feature: int(rank) for feature, rank in zip(self.feature_names, ranking)},
            'selected_feature_values': selected_feature_values
        }
        
        self.logger.info(f"Selected {len(selected_features)} features out of {len(self.feature_names)}")
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
    
    def _apply_rfe(self, features_df: pd.DataFrame, target: pd.Series) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Apply Recursive Feature Elimination to the features.
        
        Args:
            features_df: DataFrame containing features
            target: Series containing the target variable
            
        Returns:
            Tuple of (selected_features, ranking, support)
        """
        # Extract feature values as a numpy array
        X = features_df.values
        y = target.values
        
        # Create estimator based on configuration
        estimator_type = self.config['estimator_type']
        target_type = self.config['target_type']
        
        if target_type == 'discrete':
            # Classification
            if estimator_type == 'random_forest':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = LogisticRegression(random_state=42)
        else:
            # Regression
            if estimator_type == 'random_forest':
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                estimator = LinearRegression()
        
        # Initialize RFE
        n_features_to_select = self.config['n_features_to_select']
        step = self.config['step']
        
        self.rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
        
        try:
            # Apply RFE
            self.rfe.fit(X, y)
            
            # Get selected features
            support = self.rfe.support_
            ranking = self.rfe.ranking_
            
            selected_features = [feature for feature, selected in zip(features_df.columns, support) if selected]
            
            return selected_features, ranking, support
            
        except Exception as e:
            self.logger.error(f"Error applying RFE: {str(e)}")
            return [], np.array([]), np.array([])
    
    def transform_new_data(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Transform new data using the fitted RFE model.
        
        Args:
            features: Features to transform
            
        Returns:
            DataFrame with selected features
        """
        if self.rfe is None:
            self.logger.error("RFE model not fitted. Call fuse_features first.")
            return pd.DataFrame()
        
        # Convert features to DataFrame
        df = pd.DataFrame(features, index=[0])
        
        # Filter to include only the original features used for fitting
        common_features = [col for col in df.columns if col in self.feature_names]
        df = df[common_features]
        
        # Fill missing columns with zeros
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns to match the original order
        df = df[self.feature_names]
        
        # Handle missing values
        df = df.fillna(0)
        
        # Extract feature values as a numpy array
        X = df.values
        
        # Apply RFE transformation
        X_transformed = self.rfe.transform(X)
        
        # Get selected feature names
        selected_features = [feature for feature, selected in zip(self.feature_names, self.rfe.support_) if selected]
        
        # Create DataFrame with selected features
        transformed_df = pd.DataFrame(X_transformed, columns=selected_features)
        
        return transformed_df