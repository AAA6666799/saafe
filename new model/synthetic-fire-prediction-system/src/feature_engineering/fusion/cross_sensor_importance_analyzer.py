"""
Cross-Sensor Feature Importance Analyzer for FLIR+SCD41 Fire Detection System.

This module provides advanced analysis of feature importance across thermal and gas sensors,
enabling dynamic feature selection and optimization of the fusion process.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class CrossSensorImportanceAnalyzer:
    """
    Analyzer for cross-sensor feature importance and dynamic feature selection.
    
    This class provides comprehensive analysis of feature importance across
    thermal and gas sensors, enabling optimized sensor fusion and dynamic
    feature selection based on input patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cross-sensor importance analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_trained = False
        self.scaler = StandardScaler()
        self.feature_importance_model = None
        self.mutual_info_scores = {}
        self.statistical_scores = {}
        self.cross_sensor_correlations = {}
        
        # Configuration parameters
        self.thermal_feature_prefixes = self.config.get('thermal_feature_prefixes', 
                                                     ['t_', 'temp', 'thermal'])
        self.gas_feature_prefixes = self.config.get('gas_feature_prefixes', 
                                                  ['gas', 'co2', 'carbon'])
        self.importance_method = self.config.get('importance_method', 'mutual_information')
        self.selection_threshold = self.config.get('selection_threshold', 0.01)
        self.cross_validation_folds = self.config.get('cross_validation_folds', 5)
        
        logger.info("Cross-Sensor Importance Analyzer initialized")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CrossSensorImportanceAnalyzer':
        """
        Fit the importance analyzer on training data.
        
        Args:
            X: Combined feature DataFrame (thermal + gas features)
            y: Target labels
            
        Returns:
            Trained CrossSensorImportanceAnalyzer instance
        """
        logger.info("Training Cross-Sensor Importance Analyzer")
        start_time = datetime.now()
        
        try:
            # Store feature names and identify sensor types
            self.feature_names = list(X.columns)
            self._identify_sensor_features(X)
            
            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
            
            # Compute different importance measures
            self._compute_mutual_information_importance(X_scaled, y)
            self._compute_statistical_importance(X_scaled, y)
            self._compute_model_based_importance(X_scaled, y)
            self._compute_cross_sensor_correlations(X_scaled)
            
            self.is_trained = True
            training_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Cross-Sensor Importance Analyzer trained in {training_time:.2f}s")
            return self
            
        except Exception as e:
            logger.error(f"Failed to train Cross-Sensor Importance Analyzer: {str(e)}")
            raise
    
    def _identify_sensor_features(self, X: pd.DataFrame):
        """
        Identify thermal and gas features based on naming conventions.
        
        Args:
            X: Feature DataFrame
        """
        self.thermal_features = []
        self.gas_features = []
        
        for col in X.columns:
            is_thermal = any(prefix in col.lower() for prefix in self.thermal_feature_prefixes)
            is_gas = any(prefix in col.lower() for prefix in self.gas_feature_prefixes)
            
            if is_thermal:
                self.thermal_features.append(col)
            elif is_gas:
                self.gas_features.append(col)
            # If neither, assume it's a derived feature - we'll categorize based on content
            else:
                # Simple heuristic: if contains temperature-related terms
                if any(temp_term in col.lower() for temp_term in ['temp', 'thermal', 'heat']):
                    self.thermal_features.append(col)
                # If contains gas-related terms
                elif any(gas_term in col.lower() for gas_term in ['gas', 'co2', 'carbon', 'methane']):
                    self.gas_features.append(col)
                # Default to thermal if unsure
                else:
                    self.thermal_features.append(col)
        
        logger.info(f"Identified {len(self.thermal_features)} thermal features and {len(self.gas_features)} gas features")
    
    def _compute_mutual_information_importance(self, X: pd.DataFrame, y: pd.Series):
        """
        Compute feature importance using mutual information.
        
        Args:
            X: Feature DataFrame
            y: Target labels
        """
        try:
            # Compute mutual information scores
            mi_scores = mutual_info_classif(X, y, random_state=42)
            
            # Store as dictionary
            self.mutual_info_scores = {
                feature: float(score) for feature, score in zip(X.columns, mi_scores)
            }
            
            logger.info("Computed mutual information importance scores")
            
        except Exception as e:
            logger.warning(f"Failed to compute mutual information scores: {str(e)}")
            self.mutual_info_scores = {feature: 0.0 for feature in X.columns}
    
    def _compute_statistical_importance(self, X: pd.DataFrame, y: pd.Series):
        """
        Compute feature importance using statistical tests.
        
        Args:
            X: Feature DataFrame
            y: Target labels
        """
        try:
            # Compute F-scores using ANOVA F-test
            f_scores, p_values = f_classif(X, y)
            
            # Store as dictionary
            self.statistical_scores = {
                feature: {
                    'f_score': float(f_score),
                    'p_value': float(p_value)
                } for feature, (f_score, p_value) in zip(X.columns, zip(f_scores, p_values))
            }
            
            logger.info("Computed statistical importance scores")
            
        except Exception as e:
            logger.warning(f"Failed to compute statistical scores: {str(e)}")
            self.statistical_scores = {
                feature: {'f_score': 0.0, 'p_value': 1.0} for feature in X.columns
            }
    
    def _compute_model_based_importance(self, X: pd.DataFrame, y: pd.Series):
        """
        Compute feature importance using a Random Forest model.
        
        Args:
            X: Feature DataFrame
            y: Target labels
        """
        try:
            # Train a Random Forest model for importance analysis
            self.feature_importance_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.feature_importance_model.fit(X, y)
            
            # Extract feature importances
            importances = self.feature_importance_model.feature_importances_
            
            self.model_importance_scores = {
                feature: float(imp) for feature, imp in zip(X.columns, importances)
            }
            
            logger.info("Computed model-based importance scores")
            
        except Exception as e:
            logger.warning(f"Failed to compute model-based scores: {str(e)}")
            self.model_importance_scores = {feature: 0.0 for feature in X.columns}
    
    def _compute_cross_sensor_correlations(self, X: pd.DataFrame):
        """
        Compute cross-sensor correlations between thermal and gas features.
        
        Args:
            X: Feature DataFrame
        """
        try:
            if not self.thermal_features or not self.gas_features:
                logger.warning("Cannot compute cross-sensor correlations: missing feature categories")
                return
            
            # Compute correlation matrix
            correlation_matrix = X.corr()
            
            # Extract cross-correlations between thermal and gas features
            cross_correlations = {}
            for t_feature in self.thermal_features:
                for g_feature in self.gas_features:
                    if t_feature in correlation_matrix.index and g_feature in correlation_matrix.columns:
                        corr_value = correlation_matrix.loc[t_feature, g_feature]
                        cross_correlations[f"{t_feature}_x_{g_feature}"] = float(corr_value)
            
            self.cross_sensor_correlations = cross_correlations
            logger.info(f"Computed {len(cross_correlations)} cross-sensor correlations")
            
        except Exception as e:
            logger.warning(f"Failed to compute cross-sensor correlations: {str(e)}")
            self.cross_sensor_correlations = {}
    
    def get_feature_importance(self, method: str = 'combined') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            method: Importance computation method ('mutual_info', 'statistical', 'model_based', 'combined')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before getting importance scores")
        
        if method == 'mutual_info':
            return self.mutual_info_scores.copy()
        elif method == 'statistical':
            # Use F-scores for statistical importance
            return {feature: scores['f_score'] for feature, scores in self.statistical_scores.items()}
        elif method == 'model_based':
            return getattr(self, 'model_importance_scores', {})
        elif method == 'combined':
            # Combine all importance measures with weighted averaging
            combined_scores = {}
            
            # Get all available scores
            mi_scores = self.mutual_info_scores
            stat_scores = {feature: scores['f_score'] for feature, scores in self.statistical_scores.items()}
            model_scores = getattr(self, 'model_importance_scores', {})
            
            # Normalize scores to 0-1 range
            all_scores = [mi_scores, stat_scores, model_scores]
            normalized_scores = []
            
            for scores in all_scores:
                if scores:
                    max_score = max(scores.values())
                    if max_score > 0:
                        normalized = {feature: score/max_score for feature, score in scores.items()}
                    else:
                        normalized = scores
                    normalized_scores.append(normalized)
                else:
                    normalized_scores.append({})
            
            # Compute weighted average (equal weights)
            for feature in self.feature_names:
                scores = [norm_scores.get(feature, 0.0) for norm_scores in normalized_scores]
                # Only count non-zero scores
                valid_scores = [score for score in scores if score > 0]
                combined_scores[feature] = float(np.mean(valid_scores)) if valid_scores else 0.0
            
            return combined_scores
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def get_sensor_importance(self) -> Dict[str, float]:
        """
        Get importance scores for each sensor type.
        
        Returns:
            Dictionary with sensor type importance scores
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before getting sensor importance")
        
        # Get combined feature importance
        feature_importance = self.get_feature_importance('combined')
        
        # Compute average importance for each sensor type
        thermal_importance = [
            feature_importance.get(feature, 0.0) for feature in self.thermal_features
        ]
        gas_importance = [
            feature_importance.get(feature, 0.0) for feature in self.gas_features
        ]
        
        avg_thermal_importance = float(np.mean(thermal_importance)) if thermal_importance else 0.0
        avg_gas_importance = float(np.mean(gas_importance)) if gas_importance else 0.0
        
        return {
            'thermal_sensor_importance': avg_thermal_importance,
            'gas_sensor_importance': avg_gas_importance,
            'thermal_features_count': len(self.thermal_features),
            'gas_features_count': len(self.gas_features),
            'importance_ratio': float(avg_thermal_importance / (avg_gas_importance + 1e-8))
        }
    
    def select_features(self, X: pd.DataFrame, 
                       threshold: Optional[float] = None,
                       method: str = 'combined',
                       max_features: Optional[int] = None) -> List[str]:
        """
        Select important features based on importance scores.
        
        Args:
            X: Feature DataFrame (used for column names)
            threshold: Minimum importance threshold (if None, uses config threshold)
            method: Importance computation method
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before feature selection")
        
        # Get importance scores
        importance_scores = self.get_feature_importance(method)
        
        # Determine threshold
        if threshold is None:
            threshold = self.selection_threshold
        
        # Select features above threshold
        selected_features = [
            feature for feature, importance in importance_scores.items()
            if importance >= threshold
        ]
        
        # Limit to max_features if specified
        if max_features and len(selected_features) > max_features:
            # Sort by importance and take top features
            feature_importance_pairs = [
                (feature, importance_scores[feature]) for feature in selected_features
            ]
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            selected_features = [pair[0] for pair in feature_importance_pairs[:max_features]]
        
        logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")
        
        return selected_features
    
    def get_cross_sensor_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive cross-sensor analysis.
        
        Returns:
            Dictionary with cross-sensor analysis results
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before cross-sensor analysis")
        
        # Get sensor importance
        sensor_importance = self.get_sensor_importance()
        
        # Analyze cross-correlations
        strong_correlations = {
            pair: corr for pair, corr in self.cross_sensor_correlations.items()
            if abs(corr) > 0.5
        }
        
        moderate_correlations = {
            pair: corr for pair, corr in self.cross_sensor_correlations.items()
            if 0.3 <= abs(corr) <= 0.5
        }
        
        return {
            'sensor_importance': sensor_importance,
            'cross_correlations': {
                'total': len(self.cross_sensor_correlations),
                'strong': len(strong_correlations),
                'moderate': len(moderate_correlations),
                'strong_correlations': strong_correlations,
                'moderate_correlations': moderate_correlations
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def validate_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                 selected_features: List[str]) -> Dict[str, Any]:
        """
        Validate feature selection using cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target labels
            selected_features: List of selected features
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Select features
            X_selected = X[selected_features]
            
            # Train validation model
            validation_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            # Compute cross-validation scores
            cv_scores = cross_val_score(
                validation_model, X_selected, y, 
                cv=self.cross_validation_folds, 
                scoring='roc_auc'
            )
            
            return {
                'cv_mean_score': float(np.mean(cv_scores)),
                'cv_std_score': float(np.std(cv_scores)),
                'selected_features_count': len(selected_features),
                'total_features_count': len(X.columns),
                'feature_reduction_ratio': float(1.0 - len(selected_features) / len(X.columns))
            }
            
        except Exception as e:
            logger.error(f"Failed to validate feature selection: {str(e)}")
            return {
                'cv_mean_score': 0.0,
                'cv_std_score': 0.0,
                'selected_features_count': len(selected_features),
                'total_features_count': len(X.columns),
                'feature_reduction_ratio': 0.0,
                'error': str(e)
            }
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis report.
        
        Returns:
            Dictionary with complete analysis report
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before generating report")
        
        # Get all analysis components
        feature_importance = self.get_feature_importance('combined')
        sensor_importance = self.get_sensor_importance()
        cross_sensor_analysis = self.get_cross_sensor_analysis()
        
        # Top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:10]  # Top 10 features
        
        return {
            'feature_importance': feature_importance,
            'sensor_importance': sensor_importance,
            'cross_sensor_analysis': cross_sensor_analysis,
            'top_features': [
                {'feature': feature, 'importance': float(importance)}
                for feature, importance in top_features
            ],
            'total_features_analyzed': len(feature_importance),
            'report_timestamp': datetime.now().isoformat()
        }


class DynamicFeatureSelector:
    """
    Dynamic feature selector that adapts feature selection based on input patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dynamic feature selector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.analyzer = None
        self.adaptation_threshold = self.config.get('adaptation_threshold', 0.1)
        self.context_window = self.config.get('context_window', 50)
        
        # Internal state
        self.recent_inputs = []
        self.selection_history = []
        
        logger.info("Dynamic Feature Selector initialized")
    
    def initialize_analyzer(self, X: pd.DataFrame, y: pd.Series):
        """
        Initialize the importance analyzer with training data.
        
        Args:
            X: Training feature DataFrame
            y: Training labels
        """
        self.analyzer = CrossSensorImportanceAnalyzer(self.config)
        self.analyzer.fit(X, y)
        logger.info("Importance analyzer initialized and trained")
    
    def select_features_dynamic(self, X: pd.DataFrame, 
                              context_features: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Dynamically select features based on current input patterns.
        
        Args:
            X: Current input DataFrame
            context_features: Optional context features for adaptation
            
        Returns:
            List of dynamically selected feature names
        """
        if self.analyzer is None:
            raise ValueError("Analyzer must be initialized before dynamic feature selection")
        
        # Store current input for context
        self.recent_inputs.append(X)
        if len(self.recent_inputs) > self.context_window:
            self.recent_inputs.pop(0)
        
        # For now, use static selection based on pre-computed importance
        # In a more advanced implementation, this would adapt based on recent patterns
        selected_features = self.analyzer.select_features(X)
        
        # Store selection history
        self.selection_history.append({
            'timestamp': datetime.now().isoformat(),
            'selected_features': selected_features,
            'count': len(selected_features)
        })
        
        if len(self.selection_history) > self.context_window:
            self.selection_history.pop(0)
        
        return selected_features
    
    def get_adaptation_insights(self) -> Dict[str, Any]:
        """
        Get insights about feature selection adaptation.
        
        Returns:
            Dictionary with adaptation insights
        """
        if not self.selection_history:
            return {'message': 'No selection history available'}
        
        # Analyze selection patterns
        selection_counts = [entry['count'] for entry in self.selection_history]
        
        return {
            'average_selected_features': float(np.mean(selection_counts)),
            'std_selected_features': float(np.std(selection_counts)),
            'min_selected_features': int(np.min(selection_counts)),
            'max_selected_features': int(np.max(selection_counts)),
            'total_selections': len(self.selection_history),
            'adaptation_timestamp': datetime.now().isoformat()
        }