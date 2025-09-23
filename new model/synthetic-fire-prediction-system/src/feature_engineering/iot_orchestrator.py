"""
IoT Feature Extraction Orchestrator for FLIR Lepton 3.5 + SCD41 sensors.

This module provides the orchestration layer for coordinating feature extraction
specifically for IoT devices (FLIR Lepton 3.5 thermal cameras and SCD41 CO₂ sensors).
"""

from typing import Dict, Any, List, Optional, Union
import os
import json
import logging
import time
from datetime import datetime
import uuid
import numpy as np

from .framework import FeatureExtractionFramework
from .extractors.flir_thermal_extractor import FlirThermalExtractor
from .extractors.scd41_gas_extractor import Scd41GasExtractor


class IoTFeatureExtractionOrchestrator:
    """
    Orchestrator for IoT feature extraction jobs.
    
    This class coordinates the execution of feature extraction specifically for
    FLIR Lepton 3.5 thermal cameras and SCD41 CO₂ sensors, handling the
    specific data formats and processing requirements of these IoT devices.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the IoT feature extraction orchestrator.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature extractors for FLIR and SCD41
        self.flir_extractor = FlirThermalExtractor(
            config.get('feature_engineering', {}).get('thermal', {})
        )
        self.scd41_extractor = Scd41GasExtractor(
            config.get('feature_engineering', {}).get('gas', {})
        )
        
        # Initialize framework if needed for other components
        self.framework = None
        if 'framework_config' in config:
            self.framework = FeatureExtractionFramework(config['framework_config'])
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check that we have the required feature lists
        fe_config = self.config.get('feature_engineering', {})
        if 'thermal' not in fe_config or 'gas' not in fe_config:
            raise ValueError("Missing feature engineering configuration for thermal or gas sensors")
        
        thermal_features = fe_config['thermal'].get('features', [])
        gas_features = fe_config['gas'].get('features', [])
        
        if len(thermal_features) != 15:
            self.logger.warning(f"Expected 15 thermal features, got {len(thermal_features)}")
        
        if len(gas_features) != 3:
            self.logger.warning(f"Expected 3 gas features, got {len(gas_features)}")
    
    def process_iot_data(self, 
                        thermal_data: Dict[str, Any], 
                        gas_data: Dict[str, Any],
                        output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process data from FLIR Lepton 3.5 and SCD41 sensors.
        
        Args:
            thermal_data: Dictionary containing FLIR thermal data with 15 features
            gas_data: Dictionary containing SCD41 gas data with 3 features
            output_path: Optional path to save extracted features
            
        Returns:
            Dictionary containing processed features and metadata
        """
        self.logger.info("Processing IoT data from FLIR Lepton 3.5 and SCD41 sensors")
        
        # Initialize results dictionary
        results = {
            'metadata': {
                'processing_time': datetime.now().isoformat(),
                'feature_version': str(uuid.uuid4()),
                'data_sources': ['flir_lepton_3_5', 'sensirion_scd41']
            },
            'features': {}
        }
        
        try:
            # Extract features from FLIR thermal data
            self.logger.info("Extracting features from FLIR Lepton 3.5 thermal data")
            thermal_features = self.flir_extractor.extract_features(thermal_data)
            results['features']['thermal'] = thermal_features
            
            # Extract features from SCD41 gas data
            self.logger.info("Extracting features from SCD41 CO₂ gas data")
            gas_features = self.scd41_extractor.extract_features(gas_data)
            results['features']['gas'] = gas_features
            
            # Combine features if fusion is enabled
            if self.config.get('feature_engineering', {}).get('fusion', {}).get('enabled', False):
                self.logger.info("Combining thermal and gas features")
                fused_features = self._fuse_features(thermal_features, gas_features)
                results['features']['fused'] = fused_features
            
            # Save results if output path is provided
            if output_path:
                self._save_results(results, output_path)
            
            self.logger.info("IoT data processing completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing IoT data: {str(e)}")
            results['error'] = str(e)
            raise
    
    def _fuse_features(self, 
                      thermal_features: Dict[str, Any], 
                      gas_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse thermal and gas features into a single feature set.
        
        Args:
            thermal_features: Dictionary containing thermal features
            gas_features: Dictionary containing gas features
            
        Returns:
            Dictionary containing fused features
        """
        fused_features = {}
        
        # Add all thermal features
        for key, value in thermal_features.items():
            if key not in ['timestamp', 'extraction_success']:
                fused_features[f"thermal_{key}"] = value
        
        # Add all gas features
        for key, value in gas_features.items():
            if key not in ['timestamp', 'extraction_success']:
                fused_features[f"gas_{key}"] = value
        
        # Add metadata
        fused_features['timestamp'] = datetime.now().isoformat()
        fused_features['feature_count'] = len(fused_features) - 1  # Exclude timestamp
        
        return fused_features
    
    def _save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save processing results to file.
        
        Args:
            results: Dictionary containing processing results
            output_path: Path to save results
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save results as JSON
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results to {output_path}: {str(e)}")
    
    def process_batch_data(self, 
                          data_batch: List[Dict[str, Dict[str, Any]]],
                          output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of IoT data.
        
        Args:
            data_batch: List of dictionaries containing thermal and gas data
            output_dir: Optional directory to save results
            
        Returns:
            List of dictionaries containing processed features
        """
        self.logger.info(f"Processing batch of {len(data_batch)} IoT data points")
        
        results = []
        
        for i, data_point in enumerate(data_batch):
            try:
                thermal_data = data_point.get('thermal', {})
                gas_data = data_point.get('gas', {})
                
                # Process individual data point
                result = self.process_iot_data(thermal_data, gas_data)
                results.append(result)
                
                # Save individual result if output directory is provided
                if output_dir:
                    output_path = os.path.join(output_dir, f"result_{i:06d}.json")
                    self._save_results(result, output_path)
                
                # Log progress every 100 data points
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(data_batch)} data points")
            
            except Exception as e:
                self.logger.error(f"Error processing data point {i}: {str(e)}")
                # Add error result
                error_result = {
                    'metadata': {
                        'processing_time': datetime.now().isoformat(),
                        'error': str(e),
                        'data_point_index': i
                    },
                    'features': {},
                    'error': str(e)
                }
                results.append(error_result)
        
        self.logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get the names of all features that can be extracted.
        
        Returns:
            Dictionary mapping sensor types to lists of feature names
        """
        return {
            'thermal': self.flir_extractor.get_feature_names(),
            'gas': self.scd41_extractor.get_feature_names(),
            'fused': self.flir_extractor.get_feature_names() + self.scd41_extractor.get_feature_names()
        }
    
    def validate_data_format(self, 
                           thermal_data: Dict[str, Any], 
                           gas_data: Dict[str, Any]) -> bool:
        """
        Validate that the input data has the correct format for FLIR and SCD41 sensors.
        
        Args:
            thermal_data: Dictionary containing thermal data
            gas_data: Dictionary containing gas data
            
        Returns:
            True if data format is valid, False otherwise
        """
        try:
            # Validate thermal data using FLIR extractor
            self.flir_extractor._validate_thermal_data(thermal_data)
            
            # Validate gas data using SCD41 extractor
            self.scd41_extractor._validate_gas_data(gas_data)
            
            return True
        except Exception as e:
            self.logger.error(f"Data format validation failed: {str(e)}")
            return False


# Convenience function for creating IoT feature extraction orchestrator
def create_iot_feature_extractor(config: Dict[str, Any]) -> IoTFeatureExtractionOrchestrator:
    """
    Create an IoT feature extraction orchestrator with default configuration.
    
    Args:
        config: Configuration dictionary for IoT feature extraction
        
    Returns:
        Configured IoTFeatureExtractionOrchestrator instance
    """
    return IoTFeatureExtractionOrchestrator(config)