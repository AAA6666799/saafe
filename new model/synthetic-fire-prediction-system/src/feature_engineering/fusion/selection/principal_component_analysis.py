"""
Principal Component Analysis implementation for the synthetic fire prediction system.

This module provides an implementation of Principal Component Analysis (PCA),
which reduces dimensionality by transforming features into principal components.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ...base import FeatureFusion


class PrincipalComponentAnalysis(FeatureFusion):
    """
    Implementation of Principal Component Analysis (PCA).
    
    This class reduces dimensionality by transforming features into principal components,
    which are linear combinations of the original features that capture the maximum variance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PCA component.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.pca = None
        self.scaler = None
        self.feature_names = []
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required parameters
        required_params = ['n_components']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate n_components
        n_components = self.config['n_components']
        if not isinstance(n_components, (int, float)) or \
           (isinstance(n_components, int) and n_components < 1) or \
           (isinstance(n_components, float) and (n_components <= 0 or n_components > 1)):
            raise ValueError("'n_components' must be either a positive integer or a float between 0 and 1")
        
        # Set default values for optional parameters
        if 'standardize' not in self.config:
            self.config['standardize'] = True
        
        if 'whiten' not in self.config:
            self.config['whiten'] = False
        
        if 'svd_solver' not in self.config:
            self.config['svd_solver'] = 'auto'
    
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reduce dimensionality using PCA.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the transformed features
        """
        self.logger.info("Performing Principal Component Analysis")
        
        # Convert features to DataFrames
        thermal_df = self._to_dataframe(thermal_features, 'thermal')
        gas_df = self._to_dataframe(gas_features, 'gas')
        env_df = self._to_dataframe(environmental_features, 'environmental')
        
        # Combine all features into a single DataFrame
        all_features_df = pd.concat([thermal_df, gas_df, env_df], axis=1)
        
        # Filter numeric columns
        numeric_df = all_features_df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            self.logger.warning("No numeric features available for PCA")
            return {'error': 'No numeric features available for PCA'}
        
        # Store original feature names
        self.feature_names = numeric_df.columns.tolist()
        
        # Handle missing values
        numeric_df = numeric_df.fillna(0)
        
        # Apply PCA
        transformed_features, explained_variance, components = self._apply_pca(numeric_df)
        
        # Create result dictionary
        result = {
            'transformation_time': datetime.now().isoformat(),
            'n_components': self.config['n_components'],
            'standardize': self.config['standardize'],
            'whiten': self.config['whiten'],
            'svd_solver': self.config['svd_solver'],
            'original_feature_count': len(self.feature_names),
            'transformed_feature_count': transformed_features.shape[1],
            'explained_variance_ratio': explained_variance.tolist(),
            'total_explained_variance': sum(explained_variance),
            'principal_components': {
                f"PC{i+1}": {
                    'values': transformed_features[0, i],
                    'explained_variance': float(explained_variance[i]),
                    'loadings': {
                        feature: float(loading) for feature, loading in zip(self.feature_names, components[i])
                    }
                } for i in range(transformed_features.shape[1])
            }
        }
        
        self.logger.info(f"Reduced dimensionality from {len(self.feature_names)} to {transformed_features.shape[1]} components")
        return result
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # For PCA, we calculate the risk score based on the principal components
        principal_components = fused_features.get('principal_components', {})
        
        if not principal_components:
            self.logger.warning("No principal components available for risk score calculation")
            return 0.1
        
        # Calculate risk score based on the first few principal components
        # This is a simplified approach; in a real system, you would use a more sophisticated method
        
        # Get the first principal component (which explains the most variance)
        pc1 = principal_components.get('PC1', {})
        pc1_value = pc1.get('values', 0.0)
        
        # Check if PC1 value is a list or array
        if isinstance(pc1_value, (list, np.ndarray)):
            pc1_value = pc1_value[0] if len(pc1_value) > 0 else 0.0
        
        # Normalize PC1 value to [0, 1] range
        # This is a simplified approach; in a real system, you would use domain knowledge
        # to interpret the principal components
        
        # Assume PC1 values typically range from -5 to 5
        normalized_pc1 = (pc1_value + 5) / 10
        normalized_pc1 = min(1.0, max(0.0, normalized_pc1))
        
        # Check if we have a second principal component
        pc2_value = 0.0
        if 'PC2' in principal_components:
            pc2 = principal_components.get('PC2', {})
            pc2_value = pc2.get('values', 0.0)
            
            if isinstance(pc2_value, (list, np.ndarray)):
                pc2_value = pc2_value[0] if len(pc2_value) > 0 else 0.0
            
            # Normalize PC2 value
            normalized_pc2 = (pc2_value + 5) / 10
            normalized_pc2 = min(1.0, max(0.0, normalized_pc2))
        else:
            normalized_pc2 = 0.5  # Default value
        
        # Combine PC1 and PC2 with weights based on explained variance
        pc1_variance = pc1.get('explained_variance', 0.7)  # Default to 0.7 if not available
        pc2_variance = principal_components.get('PC2', {}).get('explained_variance', 0.2)  # Default to 0.2
        
        total_variance = pc1_variance + pc2_variance
        if total_variance > 0:
            pc1_weight = pc1_variance / total_variance
            pc2_weight = pc2_variance / total_variance
        else:
            pc1_weight = 0.8  # Default weights
            pc2_weight = 0.2
        
        # Calculate weighted risk score
        risk_score = (normalized_pc1 * pc1_weight) + (normalized_pc2 * pc2_weight)
        
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
    
    def _apply_pca(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply PCA to the features.
        
        Args:
            features_df: DataFrame containing features
            
        Returns:
            Tuple of (transformed_features, explained_variance_ratio, components)
        """
        # Extract feature values as a numpy array
        X = features_df.values
        
        # Standardize features if configured
        if self.config['standardize']:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        # Initialize PCA
        n_components = self.config['n_components']
        svd_solver = self.config['svd_solver']
        whiten = self.config['whiten']
        
        self.pca = PCA(n_components=n_components, svd_solver=svd_solver, whiten=whiten)
        
        # Apply PCA transformation
        transformed_features = self.pca.fit_transform(X)
        
        # Get explained variance ratio and components
        explained_variance_ratio = self.pca.explained_variance_ratio_
        components = self.pca.components_
        
        return transformed_features, explained_variance_ratio, components
    
    def transform_new_data(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Transform new data using the fitted PCA model.
        
        Args:
            features: Features to transform
            
        Returns:
            Transformed features as a numpy array
        """
        if self.pca is None:
            self.logger.error("PCA model not fitted. Call fuse_features first.")
            return np.array([])
        
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
        
        # Standardize features if a scaler was used
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Apply PCA transformation
        transformed_features = self.pca.transform(X)
        
        return transformed_features
    
    def inverse_transform(self, transformed_features: np.ndarray) -> pd.DataFrame:
        """
        Inverse transform PCA components back to original feature space.
        
        Args:
            transformed_features: PCA-transformed features
            
        Returns:
            DataFrame with reconstructed features
        """
        if self.pca is None:
            self.logger.error("PCA model not fitted. Call fuse_features first.")
            return pd.DataFrame()
        
        # Apply inverse transformation
        reconstructed = self.pca.inverse_transform(transformed_features)
        
        # Inverse standardize if a scaler was used
        if self.scaler is not None:
            reconstructed = self.scaler.inverse_transform(reconstructed)
        
        # Convert to DataFrame with original feature names
        reconstructed_df = pd.DataFrame(reconstructed, columns=self.feature_names)
        
        return reconstructed_df