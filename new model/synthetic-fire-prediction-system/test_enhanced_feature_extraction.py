#!/usr/bin/env python3
"""
Test script for enhanced feature extraction.

This script tests the new enhanced feature extractors for FLIR thermal and SCD41 gas sensors.
"""

import sys
import os
import json
import logging
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_thermal_enhanced_extractor():
    """Test the enhanced FLIR thermal extractor."""
    logger.info("Testing Enhanced FLIR Thermal Extractor")
    
    try:
        from src.feature_engineering.extractors.flir_thermal_extractor_enhanced import FlirThermalExtractorEnhanced
        
        # Create extractor
        extractor = FlirThermalExtractorEnhanced()
        
        # Sample thermal data (simulating FLIR Lepton 3.5 output)
        thermal_data = {
            't_mean': 25.5,
            't_std': 3.2,
            't_max': 45.8,
            't_p95': 38.2,
            't_hot_area_pct': 8.5,
            't_hot_largest_blob_pct': 4.2,
            't_grad_mean': 4.1,
            't_grad_std': 1.8,
            't_diff_mean': 1.2,
            't_diff_std': 0.8,
            'flow_mag_mean': 0.5,
            'flow_mag_std': 0.3,
            'tproxy_val': 30.1,
            'tproxy_delta': 2.3,
            'tproxy_vel': 2.3
        }
        
        # Extract features
        features = extractor.extract_features(thermal_data)
        
        # Print some key features
        logger.info(f"Extraction successful: {features.get('extraction_success', False)}")
        logger.info(f"Fire likelihood score: {features.get('fire_likelihood_score', 0.0):.3f}")
        logger.info(f"Blob analysis features: {len([k for k in features.keys() if 'blob' in k])} found")
        logger.info(f"Edge sharpness features: {len([k for k in features.keys() if 'edge' in k])} found")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing thermal extractor: {str(e)}")
        return False

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
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing gas extractor: {str(e)}")
        return False

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
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing cross-sensor fusion: {str(e)}")
        return False

def main():
    """Main test function."""
    logger.info("Starting Enhanced Feature Extraction Tests")
    logger.info("=" * 50)
    
    # Test each component
    tests = [
        ("Enhanced Thermal Extractor", test_thermal_enhanced_extractor),
        ("Enhanced Gas Extractor", test_gas_enhanced_extractor),
        ("Cross-Sensor Fusion", test_cross_sensor_fusion)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.info(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced feature extraction is working correctly.")
        return 0
    else:
        logger.info("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())