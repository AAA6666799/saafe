"""
Confidence scoring and uncertainty estimation for fire prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

class ConfidenceScorer:
    """
    Confidence scorer for model predictions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the confidence scorer."""
        self.config = config
        self.methods = config.get('methods', ['probability_based'])
        self.calibration_method = config.get('calibration_method', 'isotonic')
        self.calibration_models = {}
    
    def calculate_confidence(self, predictions: np.ndarray, probabilities: np.ndarray, method: str = 'probability_based') -> np.ndarray:
        """Calculate confidence scores for predictions."""
        if method == 'probability_based':
            # Use maximum probability as confidence
            return np.max(probabilities, axis=1) if len(probabilities.shape) > 1 else np.abs(probabilities - 0.5) + 0.5
        elif method == 'entropy_based':
            # Use entropy for confidence (lower entropy = higher confidence)
            if len(probabilities.shape) > 1:
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
                return 1 - entropy / np.log(probabilities.shape[1])
            else:
                # Binary case
                p = probabilities
                entropy = -(p * np.log(p + 1e-10) + (1-p) * np.log(1-p + 1e-10))
                return 1 - entropy / np.log(2)
        else:
            return np.ones_like(predictions) * 0.5

class UncertaintyEstimator:
    """
    Uncertainty estimation for model predictions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the uncertainty estimator."""
        self.config = config
        self.epistemic_estimation = config.get('epistemic_estimation', True)
        self.aleatoric_estimation = config.get('aleatoric_estimation', True)
        self.bootstrap_samples = config.get('bootstrap_samples', 100)
    
    def estimate_uncertainty(self, model, data: pd.DataFrame, method: str = 'bootstrap') -> Dict[str, np.ndarray]:
        """Estimate uncertainty in model predictions."""
        if method == 'bootstrap':
            return self._bootstrap_uncertainty(model, data)
        else:
            # Simple variance-based uncertainty
            predictions, probabilities = model.predict(data)
            uncertainty = np.var(probabilities) * np.ones_like(predictions)
            return {
                'epistemic': uncertainty,
                'aleatoric': uncertainty * 0.5,
                'total': uncertainty * 1.5
            }
    
    def _bootstrap_uncertainty(self, model, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Bootstrap-based uncertainty estimation."""
        n_samples = len(data)
        all_predictions = []
        
        for _ in range(min(10, self.bootstrap_samples)):  # Limit for speed
            # Simple resampling simulation
            indices = np.random.choice(n_samples, n_samples, replace=True)
            sampled_data = data.iloc[indices]
            _, probabilities = model.predict(sampled_data)
            all_predictions.append(probabilities)
        
        all_predictions = np.array(all_predictions)
        epistemic = np.var(all_predictions, axis=0)
        aleatoric = np.mean(all_predictions * (1 - all_predictions), axis=0)
        total = epistemic + aleatoric
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total
        }

class EnsembleVarianceAnalyzer:
    """
    Variance analysis for ensemble models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the variance analyzer."""
        self.config = config
        self.analysis_methods = config.get('analysis_methods', ['prediction_variance'])
        self.diversity_measures = config.get('diversity_measures', ['disagreement'])
    
    def analyze_variance(self, ensemble_predictions: List[np.ndarray], ensemble_probabilities: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze variance in ensemble predictions."""
        results = {}
        
        if 'prediction_variance' in self.analysis_methods:
            pred_array = np.array(ensemble_predictions)
            prob_array = np.array(ensemble_probabilities)
            
            results['prediction_variance'] = np.var(pred_array, axis=0)
            results['probability_variance'] = np.var(prob_array, axis=0)
        
        if 'disagreement' in self.diversity_measures:
            pred_array = np.array(ensemble_predictions)
            disagreement = np.mean(pred_array != pred_array[0], axis=0)
            results['disagreement'] = disagreement
        
        return results

__all__ = ['ConfidenceScorer', 'UncertaintyEstimator', 'EnsembleVarianceAnalyzer']