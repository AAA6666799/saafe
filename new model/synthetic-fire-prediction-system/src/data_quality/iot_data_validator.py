"""
IoT Data Validator for FLIR+SCD41 Sensors.

This module provides data validation and quality checking for IoT sensor data
from FLIR Lepton 3.5 thermal cameras and SCD41 CO₂ sensors.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class DataQualityLevel(Enum):
    """Enumeration of data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class ValidationResult:
    """Data class for validation results."""
    is_valid: bool
    quality_level: DataQualityLevel
    issues: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


class IoTDataValidator:
    """
    Validator for IoT sensor data quality.
    
    This class provides methods to validate and assess the quality of data
    received from FLIR thermal cameras and SCD41 CO₂ sensors.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the IoT data validator.
        
        Args:
            config: Configuration dictionary for validation thresholds
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default validation thresholds
        self.flir_thresholds = self.config.get('flir', {
            'min_temperature': -40.0,      # Celsius
            'max_temperature': 330.0,      # Celsius
            'max_missing_features': 3,     # Number of missing features allowed
            'min_hot_area_pct': 0.0,       # Minimum hot area percentage
            'max_hot_area_pct': 100.0,     # Maximum hot area percentage
            'max_temperature_std': 50.0,   # Maximum standard deviation
            'min_data_points': 10,         # Minimum data points for statistical validation
        })
        
        self.scd41_thresholds = self.config.get('scd41', {
            'min_co2': 400.0,              # ppm
            'max_co2': 40000.0,            # ppm
            'max_missing_features': 1,     # Number of missing features allowed
            'max_delta': 1000.0,           # Maximum change in CO2 (ppm/minute)
            'min_data_points': 5,          # Minimum data points for statistical validation
        })
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'quality_distribution': {
                level.value: 0 for level in DataQualityLevel
            }
        }
        
        self.logger.info("Initialized IoT Data Validator")
    
    def validate_sensor_data(self, sensor_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate sensor data from FLIR and SCD41 sensors.
        
        Args:
            sensor_data: Dictionary containing sensor data with 'flir' and 'scd41' keys
            
        Returns:
            ValidationResult containing validation results
        """
        self.validation_stats['total_validations'] += 1
        
        issues = []
        recommendations = []
        
        # Validate FLIR data
        flir_validation = self._validate_flir_data(sensor_data.get('flir', {}))
        issues.extend(flir_validation.issues)
        recommendations.extend(flir_validation.recommendations)
        
        # Validate SCD41 data
        scd41_validation = self._validate_scd41_data(sensor_data.get('scd41', {}))
        issues.extend(scd41_validation.issues)
        recommendations.extend(scd41_validation.recommendations)
        
        # Determine overall validity
        is_valid = len(issues) == 0
        if is_valid:
            self.validation_stats['passed_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
        
        # Determine quality level
        quality_level = self._determine_quality_level(issues, flir_validation, scd41_validation)
        self.validation_stats['quality_distribution'][quality_level.value] += 1
        
        # Create metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'sensor_data',
            'flir_sensors_validated': len(sensor_data.get('flir', {})),
            'scd41_sensors_validated': len(sensor_data.get('scd41', {})),
            'total_issues': len(issues)
        }
        
        result = ValidationResult(
            is_valid=is_valid,
            quality_level=quality_level,
            issues=issues,
            recommendations=recommendations,
            metadata=metadata
        )
        
        self.logger.debug(f"Validation result: {result}")
        return result
    
    def _validate_flir_data(self, flir_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate FLIR thermal camera data.
        
        Args:
            flir_data: Dictionary containing FLIR sensor data
            
        Returns:
            ValidationResult for FLIR data
        """
        issues = []
        recommendations = []
        
        if not flir_data:
            issues.append("No FLIR data provided")
            recommendations.append("Check MQTT connection or sensor status")
            return ValidationResult(
                is_valid=False,
                quality_level=DataQualityLevel.UNUSABLE,
                issues=issues,
                recommendations=recommendations,
                metadata={'sensor_type': 'flir', 'issue_count': len(issues)}
            )
        
        # Validate each FLIR sensor
        for sensor_id, sensor_data in flir_data.items():
            sensor_issues, sensor_recommendations = self._validate_single_flir_sensor(
                sensor_id, sensor_data
            )
            issues.extend(sensor_issues)
            recommendations.extend(sensor_recommendations)
        
        # Determine FLIR data quality
        flir_quality = self._determine_flir_quality(issues)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            quality_level=flir_quality,
            issues=issues,
            recommendations=recommendations,
            metadata={'sensor_type': 'flir', 'sensor_count': len(flir_data)}
        )
    
    def _validate_single_flir_sensor(self, sensor_id: str, sensor_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Validate data from a single FLIR sensor.
        
        Args:
            sensor_id: ID of the FLIR sensor
            sensor_data: Data from the FLIR sensor
            
        Returns:
            Tuple of (issues, recommendations)
        """
        issues = []
        recommendations = []
        
        # Check for required features
        required_features = [
            't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
            't_grad_mean', 'tproxy_val', 'timestamp'
        ]
        
        missing_features = [feat for feat in required_features if feat not in sensor_data]
        if missing_features:
            issues.append(f"FLIR sensor {sensor_id} missing features: {missing_features}")
            if len(missing_features) > self.flir_thresholds['max_missing_features']:
                recommendations.append(f"Sensor {sensor_id} has too many missing features, check sensor connectivity")
        
        # Validate temperature ranges
        if 't_max' in sensor_data:
            t_max = sensor_data['t_max']
            if not (self.flir_thresholds['min_temperature'] <= t_max <= self.flir_thresholds['max_temperature']):
                issues.append(f"FLIR sensor {sensor_id} t_max out of range: {t_max}°C")
                recommendations.append(f"Check calibration of FLIR sensor {sensor_id}")
        
        if 't_mean' in sensor_data:
            t_mean = sensor_data['t_mean']
            if not (self.flir_thresholds['min_temperature'] <= t_mean <= self.flir_thresholds['max_temperature']):
                issues.append(f"FLIR sensor {sensor_id} t_mean out of range: {t_mean}°C")
        
        # Validate hot area percentage
        if 't_hot_area_pct' in sensor_data:
            hot_area = sensor_data['t_hot_area_pct']
            if not (self.flir_thresholds['min_hot_area_pct'] <= hot_area <= self.flir_thresholds['max_hot_area_pct']):
                issues.append(f"FLIR sensor {sensor_id} hot area percentage out of range: {hot_area}%")
        
        # Validate temperature standard deviation
        if 't_std' in sensor_data:
            t_std = sensor_data['t_std']
            if t_std > self.flir_thresholds['max_temperature_std']:
                issues.append(f"FLIR sensor {sensor_id} temperature std too high: {t_std}°C")
                recommendations.append(f"Check for noise or calibration issues in FLIR sensor {sensor_id}")
        
        # Validate timestamp
        if 'timestamp' in sensor_data:
            try:
                timestamp = datetime.fromisoformat(sensor_data['timestamp'].replace('Z', '+00:00'))
                time_diff = datetime.now() - timestamp
                if abs(time_diff.total_seconds()) > 300:  # 5 minutes
                    issues.append(f"FLIR sensor {sensor_id} data is stale (>5 minutes old)")
                    recommendations.append(f"Check connectivity of FLIR sensor {sensor_id}")
            except ValueError:
                issues.append(f"FLIR sensor {sensor_id} has invalid timestamp format")
        
        return issues, recommendations
    
    def _validate_scd41_data(self, scd41_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate SCD41 CO₂ sensor data.
        
        Args:
            scd41_data: Dictionary containing SCD41 sensor data
            
        Returns:
            ValidationResult for SCD41 data
        """
        issues = []
        recommendations = []
        
        if not scd41_data:
            issues.append("No SCD41 data provided")
            recommendations.append("Check MQTT connection or sensor status")
            return ValidationResult(
                is_valid=False,
                quality_level=DataQualityLevel.UNUSABLE,
                issues=issues,
                recommendations=recommendations,
                metadata={'sensor_type': 'scd41', 'issue_count': len(issues)}
            )
        
        # Validate each SCD41 sensor
        for sensor_id, sensor_data in scd41_data.items():
            sensor_issues, sensor_recommendations = self._validate_single_scd41_sensor(
                sensor_id, sensor_data
            )
            issues.extend(sensor_issues)
            recommendations.extend(sensor_recommendations)
        
        # Determine SCD41 data quality
        scd41_quality = self._determine_scd41_quality(issues)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            quality_level=scd41_quality,
            issues=issues,
            recommendations=recommendations,
            metadata={'sensor_type': 'scd41', 'sensor_count': len(scd41_data)}
        )
    
    def _validate_single_scd41_sensor(self, sensor_id: str, sensor_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Validate data from a single SCD41 sensor.
        
        Args:
            sensor_id: ID of the SCD41 sensor
            sensor_data: Data from the SCD41 sensor
            
        Returns:
            Tuple of (issues, recommendations)
        """
        issues = []
        recommendations = []
        
        # Check for required features
        required_features = ['gas_val', 'co2_concentration', 'timestamp']
        missing_features = [feat for feat in required_features if feat not in sensor_data]
        if missing_features:
            issues.append(f"SCD41 sensor {sensor_id} missing features: {missing_features}")
            if len(missing_features) > self.scd41_thresholds['max_missing_features']:
                recommendations.append(f"Sensor {sensor_id} has too many missing features, check sensor connectivity")
        
        # Validate CO2 concentration ranges
        if 'co2_concentration' in sensor_data:
            co2 = sensor_data['co2_concentration']
            if not (self.scd41_thresholds['min_co2'] <= co2 <= self.scd41_thresholds['max_co2']):
                issues.append(f"SCD41 sensor {sensor_id} CO2 concentration out of range: {co2} ppm")
                recommendations.append(f"Check calibration of SCD41 sensor {sensor_id}")
        
        # Validate CO2 delta
        if 'gas_delta' in sensor_data:
            delta = sensor_data['gas_delta']
            if abs(delta) > self.scd41_thresholds['max_delta']:
                issues.append(f"SCD41 sensor {sensor_id} CO2 delta too high: {delta} ppm/minute")
                recommendations.append(f"Check for rapid CO2 changes or sensor issues in {sensor_id}")
        
        # Validate timestamp
        if 'timestamp' in sensor_data:
            try:
                timestamp = datetime.fromisoformat(sensor_data['timestamp'].replace('Z', '+00:00'))
                time_diff = datetime.now() - timestamp
                if abs(time_diff.total_seconds()) > 300:  # 5 minutes
                    issues.append(f"SCD41 sensor {sensor_id} data is stale (>5 minutes old)")
                    recommendations.append(f"Check connectivity of SCD41 sensor {sensor_id}")
            except ValueError:
                issues.append(f"SCD41 sensor {sensor_id} has invalid timestamp format")
        
        return issues, recommendations
    
    def _determine_quality_level(self, issues: List[str], flir_validation: ValidationResult, 
                                scd41_validation: ValidationResult) -> DataQualityLevel:
        """
        Determine overall data quality level.
        
        Args:
            issues: List of validation issues
            flir_validation: FLIR validation result
            scd41_validation: SCD41 validation result
            
        Returns:
            DataQualityLevel representing the overall quality
        """
        # Check for critical issues that should result in unusable data
        critical_issues = [
            "out of range",
            "stale",
            "invalid timestamp"
        ]
        
        critical_count = sum(1 for issue in issues if any(crit in issue.lower() for crit in critical_issues))
        
        if len(issues) == 0:
            return DataQualityLevel.EXCELLENT
        elif critical_count > 0:
            # If there are critical issues, classify as poor or unusable
            if critical_count >= 2:
                return DataQualityLevel.UNUSABLE
            else:
                return DataQualityLevel.POOR
        elif len(issues) <= 2:
            return DataQualityLevel.GOOD
        elif len(issues) <= 5:
            return DataQualityLevel.ACCEPTABLE
        elif len(issues) <= 10:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.UNUSABLE
    
    def _determine_flir_quality(self, issues: List[str]) -> DataQualityLevel:
        """Determine FLIR data quality level."""
        if len(issues) == 0:
            return DataQualityLevel.EXCELLENT
        elif len(issues) <= 1:
            return DataQualityLevel.GOOD
        elif len(issues) <= 3:
            return DataQualityLevel.ACCEPTABLE
        elif len(issues) <= 5:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.UNUSABLE
    
    def _determine_scd41_quality(self, issues: List[str]) -> DataQualityLevel:
        """Determine SCD41 data quality level."""
        if len(issues) == 0:
            return DataQualityLevel.EXCELLENT
        elif len(issues) <= 1:
            return DataQualityLevel.GOOD
        elif len(issues) <= 2:
            return DataQualityLevel.ACCEPTABLE
        elif len(issues) <= 3:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.UNUSABLE
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Dictionary containing validation statistics
        """
        return self.validation_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'quality_distribution': {
                level.value: 0 for level in DataQualityLevel
            }
        }


# Convenience function for creating a data validator
def create_iot_data_validator(config: Optional[Dict[str, Any]] = None) -> IoTDataValidator:
    """
    Create an IoT data validator with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured IoTDataValidator instance
    """
    default_config = {
        'flir': {
            'min_temperature': -40.0,
            'max_temperature': 330.0,
            'max_missing_features': 3,
            'min_hot_area_pct': 0.0,
            'max_hot_area_pct': 100.0,
            'max_temperature_std': 50.0,
            'min_data_points': 10,
        },
        'scd41': {
            'min_co2': 400.0,
            'max_co2': 40000.0,
            'max_missing_features': 1,
            'max_delta': 1000.0,
            'min_data_points': 5,
        }
    }
    
    if config:
        # Merge configs
        for key, value in config.items():
            if key in default_config:
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return IoTDataValidator(default_config)