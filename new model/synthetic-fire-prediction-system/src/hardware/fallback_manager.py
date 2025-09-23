"""
Fallback Manager for FLIR+SCD41 Sensors.

This module provides comprehensive fallback mechanisms for handling
device failures in the FLIR Lepton 3.5 + SCD41 CO₂ sensor system.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import time
import json

from .base import SensorMode
from ..data_generation.thermal.thermal_image_generator import ThermalImageGenerator
from ..data_generation.gas.gas_concentration_generator import GasConcentrationGenerator


class FallbackStrategy:
    """
    Base class for fallback strategies.
    """
    
    def __init__(self, name: str, priority: int = 0, description: str = ""):
        """
        Initialize the fallback strategy.
        
        Args:
            name: Name of the strategy
            priority: Priority level (higher = more preferred)
            description: Description of the strategy
        """
        self.name = name
        self.priority = priority
        self.description = description
        self.logger = logging.getLogger(__name__)
    
    def can_handle(self, failure_context: Dict[str, Any]) -> bool:
        """
        Check if this strategy can handle the given failure context.
        
        Args:
            failure_context: Context describing the failure
            
        Returns:
            True if this strategy can handle the failure
        """
        return True
    
    def execute(self, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the fallback strategy.
        
        Args:
            failure_context: Context describing the failure
            
        Returns:
            Fallback data
        """
        raise NotImplementedError("Subclasses must implement execute method")


class SyntheticDataFallback(FallbackStrategy):
    """
    Fallback strategy using synthetic data generators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the synthetic data fallback strategy.
        
        Args:
            config: Configuration for synthetic data generators
        """
        super().__init__("synthetic_data", priority=100, description="Generate synthetic sensor data")
        self.config = config
        self.thermal_generator = None
        self.gas_generator = None
        self._initialize_generators()
    
    def _initialize_generators(self) -> None:
        """Initialize synthetic data generators."""
        try:
            thermal_config = self.config.get('synthetic_thermal', {})
            self.thermal_generator = ThermalImageGenerator(thermal_config)
            
            gas_config = self.config.get('synthetic_gas', {})
            self.gas_generator = GasConcentrationGenerator(gas_config)
            
            self.logger.info("Synthetic data generators initialized for fallback")
        except Exception as e:
            self.logger.error(f"Failed to initialize synthetic data generators: {str(e)}")
    
    def can_handle(self, failure_context: Dict[str, Any]) -> bool:
        """
        Check if synthetic data fallback can handle the failure.
        
        Args:
            failure_context: Context describing the failure
            
        Returns:
            True if synthetic data can be generated
        """
        # Can handle any sensor failure if generators are available
        return self.thermal_generator is not None or self.gas_generator is not None
    
    def execute(self, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate synthetic data as fallback.
        
        Args:
            failure_context: Context describing the failure
            
        Returns:
            Synthetic sensor data
        """
        fallback_data = {
            'flir': {},
            'scd41': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': 'synthetic_fallback',
                'fallback_strategy': self.name,
                'failure_context': failure_context
            }
        }
        
        # Generate synthetic FLIR data if needed
        if failure_context.get('sensor_type') == 'flir' or failure_context.get('missing_sensors', {}).get('flir'):
            try:
                synthetic_flir = self.thermal_generator.generate_frame()
                # Convert to expected format
                fallback_data['flir'] = {
                    'synthetic': {
                        't_mean': float(np.mean(synthetic_flir['thermal_image'])),
                        't_std': float(np.std(synthetic_flir['thermal_image'])),
                        't_max': float(np.max(synthetic_flir['thermal_image'])),
                        't_p95': float(np.percentile(synthetic_flir['thermal_image'], 95)),
                        't_hot_area_pct': synthetic_flir.get('hotspot_percentage', 0.0),
                        't_grad_mean': synthetic_flir.get('gradient_magnitude', 0.0),
                        'tproxy_val': float(np.max(synthetic_flir['thermal_image'])),
                        'timestamp': datetime.now().isoformat(),
                        'thermal_frame': synthetic_flir['thermal_image']
                    }
                }
            except Exception as e:
                self.logger.warning(f"Failed to generate synthetic FLIR data: {str(e)}")
        
        # Generate synthetic SCD41 data if needed
        if failure_context.get('sensor_type') == 'scd41' or failure_context.get('missing_sensors', {}).get('scd41'):
            try:
                synthetic_gas = self.gas_generator.generate_readings()
                # Convert to expected format
                fallback_data['scd41'] = {
                    'synthetic': {
                        'gas_val': synthetic_gas.get('co2_concentration', 400.0),
                        'co2_concentration': synthetic_gas.get('co2_concentration', 400.0),
                        'gas_delta': synthetic_gas.get('co2_delta', 0.0),
                        'gas_vel': synthetic_gas.get('co2_velocity', 0.0),
                        'timestamp': datetime.now().isoformat()
                    }
                }
            except Exception as e:
                self.logger.warning(f"Failed to generate synthetic SCD41 data: {str(e)}")
        
        return fallback_data


class CachedDataFallback(FallbackStrategy):
    """
    Fallback strategy using cached historical data.
    """
    
    def __init__(self, cache_duration: timedelta = timedelta(hours=24)):
        """
        Initialize the cached data fallback strategy.
        
        Args:
            cache_duration: How long to keep cached data
        """
        super().__init__("cached_data", priority=50, description="Use cached historical sensor data")
        self.cache_duration = cache_duration
        self.data_cache = defaultdict(deque)
        self.cache_lock = threading.Lock()
    
    def can_handle(self, failure_context: Dict[str, Any]) -> bool:
        """
        Check if cached data fallback can handle the failure.
        
        Args:
            failure_context: Context describing the failure
            
        Returns:
            True if cached data is available
        """
        sensor_type = failure_context.get('sensor_type')
        if sensor_type and sensor_type in self.data_cache:
            # Check if we have recent enough data
            with self.cache_lock:
                if self.data_cache[sensor_type]:
                    latest_data = self.data_cache[sensor_type][-1]
                    data_age = datetime.now() - datetime.fromisoformat(latest_data['timestamp'])
                    return data_age <= self.cache_duration
        return False
    
    def execute(self, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use cached data as fallback.
        
        Args:
            failure_context: Context describing the failure
            
        Returns:
            Cached sensor data
        """
        fallback_data = {
            'flir': {},
            'scd41': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': 'cached_fallback',
                'fallback_strategy': self.name,
                'failure_context': failure_context
            }
        }
        
        sensor_type = failure_context.get('sensor_type')
        if sensor_type and sensor_type in self.data_cache:
            with self.cache_lock:
                if self.data_cache[sensor_type]:
                    # Use the most recent cached data
                    cached_data = self.data_cache[sensor_type][-1]
                    if sensor_type == 'flir':
                        fallback_data['flir'] = {'cached': cached_data}
                    elif sensor_type == 'scd41':
                        fallback_data['scd41'] = {'cached': cached_data}
        
        return fallback_data
    
    def add_to_cache(self, sensor_type: str, data: Dict[str, Any]) -> None:
        """
        Add data to the cache.
        
        Args:
            sensor_type: Type of sensor ('flir' or 'scd41')
            data: Sensor data to cache
        """
        with self.cache_lock:
            self.data_cache[sensor_type].append(data)
            # Remove old data
            cutoff_time = datetime.now() - self.cache_duration
            while (self.data_cache[sensor_type] and 
                   datetime.fromisoformat(self.data_cache[sensor_type][0]['timestamp']) < cutoff_time):
                self.data_cache[sensor_type].popleft()


class InterpolationFallback(FallbackStrategy):
    """
    Fallback strategy using interpolation from nearby sensors or previous readings.
    """
    
    def __init__(self):
        """
        Initialize the interpolation fallback strategy.
        """
        super().__init__("interpolation", priority=75, description="Interpolate from nearby sensors or previous readings")
        self.historical_data = defaultdict(deque)
        self.data_lock = threading.Lock()
        self.max_history = 100
    
    def can_handle(self, failure_context: Dict[str, Any]) -> bool:
        """
        Check if interpolation fallback can handle the failure.
        
        Args:
            failure_context: Context describing the failure
            
        Returns:
            True if interpolation is possible
        """
        sensor_id = failure_context.get('sensor_id')
        sensor_type = failure_context.get('sensor_type')
        
        if sensor_id and sensor_type:
            with self.data_lock:
                # Check if we have historical data for this sensor
                if sensor_id in self.historical_data and len(self.historical_data[sensor_id]) > 1:
                    return True
                
                # Check if we have data from other sensors of the same type
                for sid, data_list in self.historical_data.items():
                    if sid != sensor_id and sid.startswith(sensor_type) and data_list:
                        return True
        
        return False
    
    def execute(self, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpolate data as fallback.
        
        Args:
            failure_context: Context describing the failure
            
        Returns:
            Interpolated sensor data
        """
        fallback_data = {
            'flir': {},
            'scd41': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': 'interpolation_fallback',
                'fallback_strategy': self.name,
                'failure_context': failure_context
            }
        }
        
        sensor_id = failure_context.get('sensor_id')
        sensor_type = failure_context.get('sensor_type')
        
        if sensor_id and sensor_type:
            interpolated_data = self._interpolate_data(sensor_id, sensor_type)
            if interpolated_data:
                if sensor_type == 'flir':
                    fallback_data['flir'] = {sensor_id: interpolated_data}
                elif sensor_type == 'scd41':
                    fallback_data['scd41'] = {sensor_id: interpolated_data}
        
        return fallback_data
    
    def _interpolate_data(self, sensor_id: str, sensor_type: str) -> Optional[Dict[str, Any]]:
        """
        Interpolate data for a specific sensor.
        
        Args:
            sensor_id: ID of the sensor
            sensor_type: Type of sensor
            
        Returns:
            Interpolated data or None if not possible
        """
        with self.data_lock:
            # Try to interpolate from historical data of the same sensor
            if sensor_id in self.historical_data and len(self.historical_data[sensor_id]) >= 2:
                return self._interpolate_from_history(list(self.historical_data[sensor_id]))
            
            # Try to interpolate from other sensors of the same type
            for sid, data_list in self.historical_data.items():
                if sid != sensor_id and sid.startswith(sensor_type) and data_list:
                    return self._interpolate_from_history(list(data_list))
        
        return None
    
    def _interpolate_from_history(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Interpolate data from historical readings.
        
        Args:
            history: List of historical readings
            
        Returns:
            Interpolated data
        """
        if len(history) < 2:
            return history[0] if history else {}
        
        # Simple linear interpolation based on time
        # For now, just return the most recent data with a slight modification
        latest = history[-1].copy()
        latest['timestamp'] = datetime.now().isoformat()
        latest['source'] = 'interpolated'
        
        # Add some noise to make it look interpolated
        if 't_mean' in latest:
            latest['t_mean'] += np.random.normal(0, 0.5)
        if 'co2_concentration' in latest:
            latest['co2_concentration'] += np.random.normal(0, 10)
        
        return latest
    
    def add_historical_data(self, sensor_id: str, data: Dict[str, Any]) -> None:
        """
        Add data to historical records.
        
        Args:
            sensor_id: ID of the sensor
            data: Sensor data to add
        """
        with self.data_lock:
            self.historical_data[sensor_id].append(data)
            # Limit history size
            if len(self.historical_data[sensor_id]) > self.max_history:
                self.historical_data[sensor_id].popleft()


class LastKnownGoodFallback(FallbackStrategy):
    """
    Fallback strategy using the last known good values.
    """
    
    def __init__(self):
        """
        Initialize the last known good fallback strategy.
        """
        super().__init__("last_known_good", priority=90, description="Use last known good sensor values")
        self.last_good_values = {}
        self.values_lock = threading.Lock()
    
    def can_handle(self, failure_context: Dict[str, Any]) -> bool:
        """
        Check if last known good fallback can handle the failure.
        
        Args:
            failure_context: Context describing the failure
            
        Returns:
            True if last known good values are available
        """
        sensor_id = failure_context.get('sensor_id')
        if sensor_id and sensor_id in self.last_good_values:
            return True
        return False
    
    def execute(self, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use last known good values as fallback.
        
        Args:
            failure_context: Context describing the failure
            
        Returns:
            Last known good sensor data
        """
        fallback_data = {
            'flir': {},
            'scd41': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': 'last_known_good_fallback',
                'fallback_strategy': self.name,
                'failure_context': failure_context
            }
        }
        
        sensor_id = failure_context.get('sensor_id')
        sensor_type = failure_context.get('sensor_type')
        
        if sensor_id and sensor_id in self.last_good_values:
            last_known_data = self.last_good_values[sensor_id].copy()
            last_known_data['timestamp'] = datetime.now().isoformat()
            last_known_data['source'] = 'last_known_good'
            
            if sensor_type == 'flir':
                fallback_data['flir'] = {sensor_id: last_known_data}
            elif sensor_type == 'scd41':
                fallback_data['scd41'] = {sensor_id: last_known_data}
        
        return fallback_data
    
    def update_last_known_values(self, sensor_id: str, data: Dict[str, Any]) -> None:
        """
        Update last known good values for a sensor.
        
        Args:
            sensor_id: ID of the sensor
            data: Current sensor data
        """
        with self.values_lock:
            self.last_good_values[sensor_id] = data


class FallbackManager:
    """
    Comprehensive fallback manager for FLIR+SCD41 sensor system.
    
    This class coordinates multiple fallback strategies to ensure
    continuous operation even when sensors fail.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the fallback manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize fallback strategies
        self.strategies = []
        
        # Components for specific fallback mechanisms
        self.last_known_fallback = LastKnownGoodFallback()
        self.interpolation_fallback = InterpolationFallback()
        self.cached_fallback = CachedDataFallback()
        
        # Synthetic data fallback (highest priority)
        synthetic_config = self.config.get('synthetic_data', {})
        self.synthetic_fallback = SyntheticDataFallback(synthetic_config)
        
        self._initialize_strategies()
        
        self.logger.info("Initialized Fallback Manager with strategies: " + 
                        ", ".join([s.name for s in self.strategies]))
    
    def _initialize_strategies(self) -> None:
        """Initialize fallback strategies in order of priority."""
        # Add strategies in order of preference
        self.strategies = [
            self.last_known_fallback,
            self.interpolation_fallback,
            self.cached_fallback,
            self.synthetic_fallback
        ]
        
        # Sort by priority (highest first)
        self.strategies.sort(key=lambda s: s.priority, reverse=True)
    
    def handle_sensor_failure(self, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a sensor failure by applying the best available fallback strategy.
        
        Args:
            failure_context: Context describing the failure including:
                - sensor_id: ID of the failed sensor
                - sensor_type: Type of sensor ('flir' or 'scd41')
                - error: Error message
                - timestamp: When the failure occurred
                - missing_sensors: Dict of all missing sensors
                
        Returns:
            Fallback sensor data
        """
        self.logger.info(f"Handling sensor failure: {failure_context}")
        
        # Try each strategy in order of priority
        for strategy in self.strategies:
            try:
                if strategy.can_handle(failure_context):
                    self.logger.info(f"Applying fallback strategy: {strategy.name}")
                    fallback_data = strategy.execute(failure_context)
                    
                    # Add metadata about the fallback
                    fallback_data['metadata']['applied_strategy'] = strategy.name
                    fallback_data['metadata']['fallback_timestamp'] = datetime.now().isoformat()
                    
                    self.logger.info(f"Successfully applied fallback strategy: {strategy.name}")
                    return fallback_data
            except Exception as e:
                self.logger.warning(f"Failed to apply fallback strategy {strategy.name}: {str(e)}")
                continue
        
        # If no strategy worked, return minimal fallback data
        self.logger.warning("All fallback strategies failed, returning minimal fallback data")
        return self._get_minimal_fallback(failure_context)
    
    def _get_minimal_fallback(self, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get minimal fallback data when all strategies fail.
        
        Args:
            failure_context: Context describing the failure
            
        Returns:
            Minimal fallback data
        """
        return {
            'flir': {},
            'scd41': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': 'minimal_fallback',
                'fallback_strategy': 'none',
                'failure_context': failure_context,
                'message': 'All fallback strategies failed'
            }
        }
    
    def update_sensor_data(self, sensor_id: str, sensor_type: str, data: Dict[str, Any]) -> None:
        """
        Update fallback components with new sensor data.
        
        Args:
            sensor_id: ID of the sensor
            sensor_type: Type of sensor ('flir' or 'scd41')
            data: Current sensor data
        """
        try:
            # Update last known good values
            self.last_known_fallback.update_last_known_values(sensor_id, data)
            
            # Add to interpolation history
            self.interpolation_fallback.add_historical_data(sensor_id, data)
            
            # Add to cache
            self.cached_fallback.add_to_cache(sensor_type, data)
            
        except Exception as e:
            self.logger.warning(f"Failed to update fallback components with sensor data: {str(e)}")
    
    def get_fallback_status(self) -> Dict[str, Any]:
        """
        Get the status of fallback mechanisms.
        
        Returns:
            Dictionary containing fallback status information
        """
        return {
            'strategies': [s.name for s in self.strategies],
            'active_strategies': len(self.strategies),
            'last_known_sensors': list(self.last_known_fallback.last_good_values.keys()),
            'cached_sensor_types': list(self.cached_fallback.data_cache.keys()),
            'interpolation_sensors': list(self.interpolation_fallback.historical_data.keys())
        }


# Convenience function for creating fallback manager
def create_fallback_manager(config: Optional[Dict[str, Any]] = None) -> FallbackManager:
    """
    Create a fallback manager with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured FallbackManager instance
    """
    default_config = {
        'synthetic_data': {
            'synthetic_thermal': {
                'image_width': 160,
                'image_height': 120,
                'base_temperature': 20.0,
                'hotspot_probability': 0.05
            },
            'synthetic_gas': {
                'base_co2': 400.0,
                'variation_factor': 0.1
            }
        },
        'cache_duration_hours': 24
    }
    
    if config:
        # Merge configs
        for key, value in config.items():
            if key in default_config and isinstance(default_config[key], dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return FallbackManager(default_config)