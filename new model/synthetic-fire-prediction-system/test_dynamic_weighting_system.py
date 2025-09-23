#!/usr/bin/env python3
"""
Test script for the Dynamic Weighting System.
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

def test_dynamic_weighting_system():
    """Test the dynamic weighting system."""
    logger.info("Testing Dynamic Weighting System")
    
    try:
        from src.ml.ensemble.dynamic_weighting_system import DynamicWeightingSystem, EnvironmentalConditionAdapter, ConfidenceBasedVotingSystem
        
        # Create dynamic weighting system
        dynamic_ensemble = DynamicWeightingSystem({
            'weighting_strategy': 'performance_adaptive',
            'confidence_threshold': 0.6,
            'adaptation_rate': 0.15
        })
        
        # Create sample data
        thermal_features = pd.DataFrame({
            't_mean': [25.5, 30.2, 22.1],
            't_max': [45.8, 65.3, 35.7],
            't_hot_area_pct': [8.5, 25.3, 2.1],
            't_std': [3.2, 8.7, 1.8],
            't_p95': [38.2, 58.9, 30.4],
            't_hot_largest_blob_pct': [5.1, 18.7, 1.2],
            't_grad_mean': [2.1, 5.3, 1.2],
            't_grad_std': [0.8, 2.1, 0.5],
            't_diff_mean': [1.5, 3.8, 0.9],
            't_diff_std': [0.6, 1.9, 0.3],
            'flow_mag_mean': [1.2, 3.1, 0.8],
            'flow_mag_std': [0.4, 1.2, 0.2],
            'tproxy_val': [42.3, 62.8, 32.6],
            'tproxy_delta': [8.7, 22.4, 3.1],
            'tproxy_vel': [3.1, 8.9, 1.5]
        })
        
        gas_features = pd.DataFrame({
            'gas_val': [485.0, 1250.0, 420.0],
            'gas_delta': [25.0, 350.0, 15.0],
            'gas_vel': [25.0, 350.0, 15.0]
        })
        
        y_train = pd.Series([0, 1, 0])  # Mixed: no fire, fire, no fire
        
        # Test ensemble info before training
        info = dynamic_ensemble.get_model_info()
        logger.info(f"Dynamic ensemble info: {info}")
        
        # Test that prediction fails before training
        try:
            result = dynamic_ensemble.predict(thermal_features, gas_features)
            logger.error("Prediction should have failed before training")
            return False
        except ValueError as e:
            logger.info(f"Correctly failed prediction before training: {e}")
        
        # Create and train the default ensemble
        dynamic_ensemble.create_default_ensemble()
        training_result = dynamic_ensemble.train(thermal_features, gas_features, y_train)
        logger.info(f"Training result: {training_result}")
        
        # Test prediction after training without environmental conditions
        result1 = dynamic_ensemble.predict(thermal_features.iloc[[0]], gas_features.iloc[[0]])
        logger.info(f"Dynamic ensemble prediction (no env conditions): {result1}")
        
        # Test prediction with environmental conditions
        environmental_conditions = {
            'temperature': 65.3,  # High temperature
            'co2': 1250.0,        # High CO2
            'humidity': 45.0      # Normal humidity
        }
        
        result2 = dynamic_ensemble.predict(
            thermal_features.iloc[[1]], 
            gas_features.iloc[[1]], 
            environmental_conditions=environmental_conditions
        )
        logger.info(f"Dynamic ensemble prediction (with env conditions): {result2}")
        
        # Check model contributions
        if 'model_contributions' in result2:
            logger.info(f"Model contributions: {result2['model_contributions']}")
        
        # Test weight history
        weight_history = dynamic_ensemble.get_weight_history()
        logger.info(f"Weight history length: {len(weight_history)}")
        
        # Test performance summary
        performance_summary = dynamic_ensemble.get_model_performance_summary()
        logger.info(f"Performance summary: {performance_summary}")
        
        # Test EnvironmentalConditionAdapter
        logger.info("Testing EnvironmentalConditionAdapter...")
        env_adapter = EnvironmentalConditionAdapter({
            'condition_weights': {
                'temperature': 0.4,
                'co2': 0.4,
                'humidity': 0.2
            }
        })
        
        base_weights = {
            'thermal_model': 1.0,
            'gas_model': 1.0,
            'fusion_model': 1.0
        }
        
        test_conditions = {
            'temperature': 75.0,  # Very high
            'co2': 2500.0,        # High
            'humidity': 85.0      # High
        }
        
        adapted_weights = env_adapter.adapt_weights_for_conditions(base_weights, test_conditions)
        logger.info(f"Base weights: {base_weights}")
        logger.info(f"Adapted weights: {adapted_weights}")
        
        # Test ConfidenceBasedVotingSystem
        logger.info("Testing ConfidenceBasedVotingSystem...")
        voting_system = ConfidenceBasedVotingSystem({
            'confidence_threshold': 0.6,
            'voting_method': 'weighted'
        })
        
        predictions = [0.3, 0.8, 0.65]  # Three model predictions
        confidences = [0.4, 0.9, 0.7]   # Three model confidences
        weights = [0.8, 1.2, 1.0]       # Three model weights
        
        voting_result = voting_system.vote(predictions, confidences, weights)
        logger.info(f"Voting result: {voting_result}")
        
        # Test different voting methods
        voting_system_threshold = ConfidenceBasedVotingSystem({
            'confidence_threshold': 0.6,
            'voting_method': 'threshold'
        })
        
        voting_result_threshold = voting_system_threshold.vote(predictions, confidences)
        logger.info(f"Threshold voting result: {voting_result_threshold}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing dynamic weighting system: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("Starting Dynamic Weighting System Test")
    logger.info("=" * 50)
    
    result = test_dynamic_weighting_system()
    
    logger.info("\n" + "=" * 50)
    if result:
        logger.info("✅ Dynamic Weighting System Test PASSED")
        logger.info("🎉 Dynamic weighting system is working correctly.")
        return 0
    else:
        logger.info("❌ Dynamic Weighting System Test FAILED")
        logger.info("⚠️  Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())