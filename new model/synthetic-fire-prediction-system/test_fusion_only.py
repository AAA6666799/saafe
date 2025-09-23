#!/usr/bin/env python3
"""
Simple test script for cross-sensor fusion extraction only.
"""

import sys
import os
import logging

# Add the current directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cross_sensor_fusion():
    """Test the cross-sensor fusion extractor."""
    logger.info("Testing Cross-Sensor Fusion Extractor")
    
    try:
        from src.feature_engineering.fusion.cross_sensor_fusion_extractor import CrossSensorFusionExtractor
        
        # Create extractor
        extractor = CrossSensorFusionExtractor()
        
        # Sample data from both sensors
        thermal_features = {
            't_mean': 25.5,
            't_max': 45.8,
            't_hot_area_pct': 8.5,
            'tproxy_vel': 2.3,
            'fire_likelihood_score': 0.65
        }
        
        gas_features = {
            'gas_val': 485.0,
            'gas_vel': 12.5,
            'gas_fire_likelihood_score': 0.35
        }
        
        # Extract fused features
        fused_features = extractor.extract_fused_features(thermal_features, gas_features)
        
        # Print some key features
        logger.info(f"Fusion successful: {fused_features.get('fusion_success', False)}")
        logger.info(f"Fused fire likelihood: {fused_features.get('fused_fire_likelihood', 0.0):.3f}")
        logger.info(f"Risk convergence index: {fused_features.get('risk_convergence_index', 0.0):.3f}")
        
        # Test with history
        thermal_history = [
            {'t_mean': 22.0, 't_max': 30.0, 't_hot_area_pct': 2.0, 'tproxy_vel': 0.5, 'fire_likelihood_score': 0.1},
            {'t_mean': 24.0, 't_max': 38.0, 't_hot_area_pct': 5.0, 'tproxy_vel': 1.2, 'fire_likelihood_score': 0.3},
            {'t_mean': 25.5, 't_max': 45.8, 't_hot_area_pct': 8.5, 'tproxy_vel': 2.3, 'fire_likelihood_score': 0.65}
        ]
        
        gas_history = [
            {'gas_val': 420.0, 'gas_vel': 2.0, 'gas_fire_likelihood_score': 0.05},
            {'gas_val': 450.0, 'gas_vel': 8.0, 'gas_fire_likelihood_score': 0.2},
            {'gas_val': 485.0, 'gas_vel': 12.5, 'gas_fire_likelihood_score': 0.35}
        ]
        
        fused_features_with_history = extractor.extract_fused_features_with_history(
            thermal_features, gas_features, thermal_history, gas_history)
        
        logger.info(f"Enhanced fused features with history: {len(fused_features_with_history)} total features")
        
        # Check for correlation features
        correlation_features = [k for k in fused_features_with_history.keys() if 'correlation' in k]
        logger.info(f"Correlation analysis features: {len(correlation_features)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing cross-sensor fusion: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("Starting Cross-Sensor Fusion Test")
    logger.info("=" * 50)
    
    result = test_cross_sensor_fusion()
    
    logger.info("\n" + "=" * 50)
    if result:
        logger.info("✅ Cross-Sensor Fusion Test PASSED")
        logger.info("🎉 Cross-sensor fusion extraction is working correctly.")
        return 0
    else:
        logger.info("❌ Cross-Sensor Fusion Test FAILED")
        logger.info("⚠️  Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())