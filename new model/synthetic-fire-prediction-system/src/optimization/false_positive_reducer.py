"""
False Positive Reduction for FLIR+SCD41 Fire Detection System.

This module implements advanced false positive reduction techniques including
statistical filtering, pattern recognition, and contextual analysis.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

class FalsePositiveMetrics:
    """Container for false positive metrics."""
    
    def __init__(self):
        self.overall_rate = 0.0
        self.by_category = {}
        self.trend = 0.0  # Positive = increasing, Negative = decreasing
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_rate': self.overall_rate,
            'by_category': self.by_category,
            'trend': self.trend,
            'timestamp': self.timestamp.isoformat()
        }

class FalsePositiveReducer:
    """Implements advanced false positive reduction techniques."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize false positive reducer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.baseline_false_positive_rate = self.config.get('baseline_fpr', 0.182)  # 18.2% baseline
        self.target_reduction = self.config.get('target_reduction', 0.5)  # 50% reduction target
        self.analysis_window = self.config.get('analysis_window', 1000)  # Number of samples to analyze
        self.false_positive_history = deque(maxlen=10000)
        self.discrimination_threshold = self.config.get('discrimination_threshold', 0.7)
        
        # False positive categories
        self.fp_categories = [
            'sunlight_heating',
            'hvac_effect',
            'cooking',
            'steam_dust',
            'other'
        ]
        
        # Category-specific thresholds
        self.category_thresholds = self.config.get('category_thresholds', {
            'sunlight_heating': 0.8,
            'hvac_effect': 0.75,
            'cooking': 0.7,
            'steam_dust': 0.65,
            'other': 0.6
        })
        
        logger.info("False Positive Reducer initialized")
    
    def analyze_false_positives(self, predictions: np.ndarray, 
                              ground_truth: np.ndarray,
                              features: pd.DataFrame = None) -> FalsePositiveMetrics:
        """
        Analyze false positive patterns in predictions.
        
        Args:
            predictions: Model predictions (0 or 1)
            ground_truth: Actual labels (0 or 1)
            features: Optional feature data for detailed analysis
            
        Returns:
            FalsePositiveMetrics object
        """
        metrics = FalsePositiveMetrics()
        
        try:
            # Calculate basic false positive rate
            false_positives = np.sum((predictions == 1) & (ground_truth == 0))
            total_negatives = np.sum(ground_truth == 0)
            
            if total_negatives > 0:
                metrics.overall_rate = false_positives / total_negatives
            else:
                metrics.overall_rate = 0.0
            
            # Add to history
            self.false_positive_history.append({
                'rate': metrics.overall_rate,
                'timestamp': datetime.now(),
                'sample_count': len(predictions)
            })
            
            # Calculate trend (slope of recent rates)
            if len(self.false_positive_history) >= 2:
                recent_history = list(self.false_positive_history)[-10:]  # Last 10 measurements
                rates = [entry['rate'] for entry in recent_history]
                if len(rates) >= 2:
                    x = np.arange(len(rates))
                    slope = np.polyfit(x, rates, 1)[0] if len(rates) > 1 else 0
                    metrics.trend = slope
            
            # Category-specific analysis if features provided
            if features is not None and len(features) == len(predictions):
                metrics.by_category = self._analyze_by_category(predictions, ground_truth, features)
            
        except Exception as e:
            logger.warning(f"Error analyzing false positives: {e}")
            metrics.overall_rate = 0.0
            metrics.by_category = {}
            metrics.trend = 0.0
        
        return metrics
    
    def _analyze_by_category(self, predictions: np.ndarray, 
                           ground_truth: np.ndarray,
                           features: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze false positives by category.
        
        Args:
            predictions: Model predictions
            ground_truth: Actual labels
            features: Feature data
            
        Returns:
            Dictionary with category false positive rates
        """
        category_rates = {}
        
        try:
            # Identify false positives
            false_positive_mask = (predictions == 1) & (ground_truth == 0)
            fp_features = features[false_positive_mask]
            
            if len(fp_features) == 0:
                # No false positives to analyze
                return {category: 0.0 for category in self.fp_categories}
            
            # Analyze each category
            for category in self.fp_categories:
                rate = self._calculate_category_rate(category, fp_features, features)
                category_rates[category] = rate
                
        except Exception as e:
            logger.warning(f"Error in category analysis: {e}")
            category_rates = {category: 0.0 for category in self.fp_categories}
        
        return category_rates
    
    def _calculate_category_rate(self, category: str, fp_features: pd.DataFrame, 
                               all_features: pd.DataFrame) -> float:
        """
        Calculate false positive rate for a specific category.
        
        Args:
            category: Category name
            fp_features: Features for false positive samples
            all_features: All feature data
            
        Returns:
            Category false positive rate
        """
        try:
            if category == 'sunlight_heating':
                # Sunlight heating: High temperature, low CO2 change
                high_temp = fp_features['t_max'] > 40.0
                low_co2_change = abs(fp_features['gas_delta']) < 10.0
                category_mask = high_temp & low_co2_change
                category_fps = fp_features[category_mask]
                
            elif category == 'hvac_effect':
                # HVAC effect: Moderate temperature changes, normal CO2
                temp_change = abs(fp_features['t_max'] - fp_features['t_mean'])
                moderate_temp = (5.0 < temp_change) & (temp_change < 20.0)
                normal_co2 = (400.0 <= fp_features['gas_val']) & (fp_features['gas_val'] <= 1000.0)
                category_mask = moderate_temp & normal_co2
                category_fps = fp_features[category_mask]
                
            elif category == 'cooking':
                # Cooking: Localized heating, moderate CO2
                localized_heating = fp_features['t_hot_area_pct'] < 5.0
                moderate_co2 = (600.0 <= fp_features['gas_val']) & (fp_features['gas_val'] <= 1500.0)
                category_mask = localized_heating & moderate_co2
                category_fps = fp_features[category_mask]
                
            elif category == 'steam_dust':
                # Steam/dust: Moderate temperature, may have CO2 fluctuations
                moderate_temp = fp_features['t_max'] > 35.0
                category_mask = moderate_temp
                category_fps = fp_features[category_mask]
                
            else:  # 'other'
                # For other categories, we might use clustering or other methods
                # For now, we'll estimate based on remaining false positives
                categorized_count = sum([
                    self._calculate_category_count('sunlight_heating', fp_features, all_features),
                    self._calculate_category_count('hvac_effect', fp_features, all_features),
                    self._calculate_category_count('cooking', fp_features, all_features),
                    self._calculate_category_count('steam_dust', fp_features, all_features)
                ])
                total_fps = len(fp_features)
                category_count = max(0, total_fps - categorized_count)
                category_fps = fp_features.head(category_count) if category_count > 0 else pd.DataFrame()
            
            # Calculate rate
            total_negatives = len(all_features[all_features['ground_truth'] == 0])
            if total_negatives > 0:
                return len(category_fps) / total_negatives
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating rate for category {category}: {e}")
            return 0.0
    
    def _calculate_category_count(self, category: str, fp_features: pd.DataFrame, 
                                all_features: pd.DataFrame) -> int:
        """Helper method to calculate category count."""
        try:
            if category == 'sunlight_heating':
                high_temp = fp_features['t_max'] > 40.0
                low_co2_change = abs(fp_features['gas_delta']) < 10.0
                category_mask = high_temp & low_co2_change
                return len(fp_features[category_mask])
                
            elif category == 'hvac_effect':
                temp_change = abs(fp_features['t_max'] - fp_features['t_mean'])
                moderate_temp = (5.0 < temp_change) & (temp_change < 20.0)
                normal_co2 = (400.0 <= fp_features['gas_val']) & (fp_features['gas_val'] <= 1000.0)
                category_mask = moderate_temp & normal_co2
                return len(fp_features[category_mask])
                
            elif category == 'cooking':
                localized_heating = fp_features['t_hot_area_pct'] < 5.0
                moderate_co2 = (600.0 <= fp_features['gas_val']) & (fp_features['gas_val'] <= 1500.0)
                category_mask = localized_heating & moderate_co2
                return len(fp_features[category_mask])
                
            elif category == 'steam_dust':
                moderate_temp = fp_features['t_max'] > 35.0
                category_mask = moderate_temp
                return len(fp_features[category_mask])
                
        except Exception as e:
            logger.warning(f"Error calculating count for category {category}: {e}")
            return 0
    
    def apply_false_positive_filter(self, predictions: np.ndarray, 
                                  confidence_scores: np.ndarray,
                                  features: pd.DataFrame = None) -> np.ndarray:
        """
        Apply false positive filtering to predictions.
        
        Args:
            predictions: Original predictions (0 or 1)
            confidence_scores: Confidence scores for predictions
            features: Optional feature data for discrimination
            
        Returns:
            Filtered predictions with reduced false positives
        """
        filtered_predictions = predictions.copy()
        
        try:
            # Basic confidence-based filtering
            low_confidence_mask = confidence_scores < self.discrimination_threshold
            filtered_predictions[low_confidence_mask & (predictions == 1)] = 0
            
            # Advanced filtering if features provided
            if features is not None and len(features) == len(predictions):
                # Apply category-specific discrimination
                for i, (pred, conf, feature_row) in enumerate(zip(predictions, confidence_scores, features.iterrows())):
                    if pred == 1:  # Only filter positive predictions
                        category_scores = self._calculate_category_discrimination_scores(feature_row[1])
                        max_category_score = max(category_scores.values()) if category_scores else 0
                        
                        # If discrimination score is high, it's likely a false positive
                        if max_category_score > self.discrimination_threshold:
                            filtered_predictions[i] = 0
            
        except Exception as e:
            logger.warning(f"Error applying false positive filter: {e}")
        
        return filtered_predictions
    
    def _calculate_category_discrimination_scores(self, features: pd.Series) -> Dict[str, float]:
        """
        Calculate discrimination scores for each false positive category.
        
        Args:
            features: Feature values for a single sample
            
        Returns:
            Dictionary with discrimination scores for each category
        """
        scores = {}
        
        try:
            # Sunlight heating discrimination score
            temp_score = min(1.0, max(0.0, (features['t_max'] - 30.0) / 20.0))  # High temp indicator
            co2_score = min(1.0, max(0.0, (15.0 - abs(features['gas_delta'])) / 15.0))  # Low CO2 change indicator
            scores['sunlight_heating'] = (temp_score + co2_score) / 2
            
            # HVAC effect discrimination score
            temp_change = abs(features['t_max'] - features['t_mean'])
            temp_change_score = min(1.0, max(0.0, abs(temp_change - 12.5) / 15.0))  # Moderate change indicator
            co2_normal_score = min(1.0, max(0.0, (1000.0 - abs(features['gas_val'] - 700.0)) / 1000.0))  # Normal CO2 indicator
            scores['hvac_effect'] = (temp_change_score + co2_normal_score) / 2
            
            # Cooking discrimination score
            localized_score = min(1.0, max(0.0, (10.0 - features['t_hot_area_pct']) / 10.0))  # Localized heating indicator
            co2_moderate_score = min(1.0, max(0.0, (1500.0 - abs(features['gas_val'] - 1050.0)) / 1500.0))  # Moderate CO2 indicator
            scores['cooking'] = (localized_score + co2_moderate_score) / 2
            
            # Steam/dust discrimination score
            temp_moderate_score = min(1.0, max(0.0, abs(features['t_max'] - 50.0) / 30.0))  # Moderate temp indicator
            scores['steam_dust'] = temp_moderate_score
            
        except Exception as e:
            logger.warning(f"Error calculating discrimination scores: {e}")
            scores = {category: 0.0 for category in self.fp_categories}
        
        return scores
    
    def get_reduction_metrics(self) -> Dict[str, Any]:
        """
        Get false positive reduction metrics.
        
        Returns:
            Dictionary with reduction metrics
        """
        if len(self.false_positive_history) < 2:
            return {'status': 'insufficient_data'}
        
        # Get current and baseline rates
        current_rate = self.false_positive_history[-1]['rate']
        reduction_percentage = (self.baseline_false_positive_rate - current_rate) / self.baseline_false_positive_rate * 100
        
        # Calculate trend
        rates = [entry['rate'] for entry in self.false_positive_history]
        trend = np.polyfit(np.arange(len(rates)), rates, 1)[0] if len(rates) > 1 else 0
        
        return {
            'status': 'success',
            'baseline_rate': self.baseline_false_positive_rate,
            'current_rate': current_rate,
            'reduction_percentage': reduction_percentage,
            'target_reduction': self.target_reduction * 100,
            'achieved_target': reduction_percentage >= (self.target_reduction * 100),
            'trend': trend,
            'improving': trend < 0,  # Negative trend means decreasing false positives
            'total_samples_analyzed': sum(entry['sample_count'] for entry in self.false_positive_history)
        }
    
    def generate_false_positive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive false positive analysis report.
        
        Returns:
            Dictionary with false positive report
        """
        if len(self.false_positive_history) == 0:
            return {'status': 'no_data'}
        
        # Get current metrics
        current_metrics = self.false_positive_history[-1]
        
        # Get historical statistics
        all_rates = [entry['rate'] for entry in self.false_positive_history]
        
        # Get reduction metrics
        reduction_metrics = self.get_reduction_metrics()
        
        report = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'total_analysis_cycles': len(self.false_positive_history),
            'current_metrics': {
                'overall_rate': current_metrics['rate'],
                'sample_count': current_metrics['sample_count']
            },
            'historical_statistics': {
                'min_rate': np.min(all_rates) if all_rates else 0,
                'max_rate': np.max(all_rates) if all_rates else 0,
                'mean_rate': np.mean(all_rates) if all_rates else 0,
                'std_rate': np.std(all_rates) if all_rates else 0
            },
            'reduction_metrics': reduction_metrics,
            'target_metrics': {
                'baseline_fpr': self.baseline_false_positive_rate,
                'target_fpr': self.baseline_false_positive_rate * (1 - self.target_reduction),
                'target_reduction_percentage': self.target_reduction * 100
            }
        }
        
        return report

# Convenience function
def create_false_positive_reducer(config: Dict[str, Any] = None) -> FalsePositiveReducer:
    """
    Create a false positive reducer instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FalsePositiveReducer instance
    """
    return FalsePositiveReducer(config)

__all__ = ['FalsePositiveMetrics', 'FalsePositiveReducer', 'create_false_positive_reducer']