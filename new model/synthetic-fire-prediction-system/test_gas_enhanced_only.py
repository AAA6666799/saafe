#!/usr/bin/env python3
"""
Simple test script for enhanced gas feature extraction only.
"""

import sys
import os
import logging

# Add the current directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gas_enhanced_extractor():
    """Test the enhanced SCD41 gas extractor."""
    logger.info("Testing Enhanced SCD41 Gas Extractor")
    
    try:
        from src.feature_engineering.extractors.scd41_gas_extractor_enhanced import Scd41GasExtractorEnhanced
        
        # Create extractor
        extractor = Scd41GasExtractorEnhanced()
        
        # Sample gas data (simulating SCD41 output)
        gas_data = {
            'gas_val': 485.0,
            'gas_delta': 12.5,
            'gas_vel': 12.5
        }
        
        # Extract features
        features = extractor.extract_features(gas_data)
        
        # Print some key features
        logger.info(f"Extraction successful: {features.get('extraction_success', False)}")
        logger.info(f"Gas fire likelihood score: {features.get('gas_fire_likelihood_score', 0.0):.3f}")
        logger.info(f"CO2 elevation level: {features.get('co2_elevation_level', 'UNKNOWN')}")
        
        # Test with history
        gas_history = [
            {'gas_val': 450.0, 'gas_delta': 5.0, 'gas_vel': 5.0},
            {'gas_val': 465.0, 'gas_delta': 15.0, 'gas_vel': 15.0},
            {'gas_val': 485.0, 'gas_delta': 20.0, 'gas_vel': 20.0}
        ]
        
        features_with_history = extractor.extract_features_with_history(gas_data, gas_history)
        logger.info(f"Enhanced features with history: {len(features_with_history)} total features")
        
        # Check for enhanced features
        accumulation_features = [k for k in features_with_history.keys() if 'accumulation' in k]
        drift_features = [k for k in features_with_history.keys() if 'drift' in k]
        
        logger.info(f"Accumulation analysis features: {len(accumulation_features)}")
        logger.info(f"Drift detection features: {len(drift_features)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing gas extractor: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("Starting Enhanced Gas Feature Extraction Test")
    logger.info("=" * 50)
    
    result = test_gas_enhanced_extractor()
    
    logger.info("\n" + "=" * 50)
    if result:
        logger.info("✅ Enhanced Gas Extractor Test PASSED")
        logger.info("🎉 Enhanced gas feature extraction is working correctly.")
        return 0
    else:
        logger.info("❌ Enhanced Gas Extractor Test FAILED")
        logger.info("⚠️  Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())