#!/usr/bin/env python3
"""
Test script for the simple ensemble manager.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd

# Add the current directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_ensemble():
    """Test the simple ensemble manager."""
    logger.info("Testing Simple Ensemble Manager")
    
    try:
        from src.ml.ensemble.simple_ensemble_manager import SimpleEnsembleManager
        
        # Create ensemble manager
        ensemble = SimpleEnsembleManager({
            'ensemble_method': 'weighted_average',
            'confidence_threshold': 0.6
        })
        
        # Create sample data
        thermal_features = pd.DataFrame({
            't_mean': [25.5],
            't_max': [45.8],
            't_hot_area_pct': [8.5],
            't_std': [3.2],
            't_p95': [38.2],
            't_hot_largest_blob_pct': [5.1],
            't_grad_mean': [2.1],
            't_grad_std': [0.8],
            't_diff_mean': [1.5],
            't_diff_std': [0.6],
            'flow_mag_mean': [1.2],
            'flow_mag_std': [0.4],
            'tproxy_val': [42.3],
            'tproxy_delta': [8.7],
            'tproxy_vel': [3.1]
        })
        
        gas_features = pd.DataFrame({
            'gas_val': [485.0],
            'gas_delta': [25.0],
            'gas_vel': [25.0]
        })
        
        y_train = pd.Series([1])  # Fire detected
        
        # Test ensemble info before training
        info = ensemble.get_model_info()
        logger.info(f"Ensemble info: {info}")
        
        # Test that prediction fails before training
        try:
            result = ensemble.predict(thermal_features, gas_features)
            logger.error("Prediction should have failed before training")
            return False
        except ValueError as e:
            logger.info(f"Correctly failed prediction before training: {e}")
        
        # Create and train the default ensemble
        ensemble.create_default_ensemble()
        training_result = ensemble.train(thermal_features, gas_features, y_train)
        logger.info(f"Training result: {training_result}")
        
        # Test prediction after training
        result = ensemble.predict(thermal_features, gas_features)
        logger.info(f"Ensemble prediction result: {result}")
        
        if 'fire_detected' not in result:
            logger.error("Prediction result should contain 'fire_detected'")
            return False
        
        if 'confidence_score' not in result:
            logger.error("Prediction result should contain 'confidence_score'")
            return False
        
        # Test with different weights
        ensemble2 = SimpleEnsembleManager()
        ensemble2.create_default_ensemble()
        ensemble2.train(thermal_features, gas_features, y_train)
        
        # Manually adjust weights to test different scenarios
        ensemble2.model_weights['thermal_model'] = 0.9
        ensemble2.model_weights['gas_model'] = 0.2
        ensemble2.model_weights['fusion_model'] = 0.8
        
        result2 = ensemble2.predict(thermal_features, gas_features)
        logger.info(f"Weighted ensemble result: {result2}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing simple ensemble: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("Starting Simple Ensemble Test")
    logger.info("=" * 50)
    
    result = test_simple_ensemble()
    
    logger.info("\n" + "=" * 50)
    if result:
        logger.info("✅ Simple Ensemble Test PASSED")
        logger.info("🎉 Simple ensemble manager is working correctly.")
        return 0
    else:
        logger.info("❌ Simple Ensemble Test FAILED")
        logger.info("⚠️  Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())