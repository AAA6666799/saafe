"""
Sensor Manager for Hardware Abstraction Layer.

This module provides a unified interface for managing both synthetic and real sensors,
enabling seamless integration between simulation and production environments.
"""

import logging
import numpy as np
import pandas as pd
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Type
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import threading
import queue
import json

from .base import SensorInterface, ThermalSensorInterface, GasSensorInterface, EnvironmentalSensorInterface


class SensorMode:
    """Constants for sensor operation modes."""
    SYNTHETIC = "synthetic"
    REAL = "real"
    HYBRID = "hybrid"
    SIMULATION = "simulation"


class SensorManager:
    """
    Comprehensive sensor manager for hardware abstraction.
    
    This manager provides a unified interface for both synthetic and real sensors,
    enabling seamless switching between simulation and production modes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sensor manager.
        
        Args:
            config: Configuration dictionary containing sensor settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Operating mode (synthetic, real, or hybrid)
        self.mode = config.get('mode', SensorMode.SYNTHETIC)
        self.auto_fallback = config.get('auto_fallback', True)
        
        # Sensor registries
        self.thermal_sensors = {}
        self.gas_sensors = {}
        self.environmental_sensors = {}
        self.all_sensors = {}
        
        # Data collection and buffering
        self.data_buffer = deque(maxlen=config.get('buffer_size', 1000))
        self.collection_interval = config.get('collection_interval', 1.0)  # seconds
        self.is_collecting = False
        self.collection_thread = None
        self.stop_event = threading.Event()
        
        # Health monitoring
        self.sensor_health = {}
        self.failed_sensors = set()
        self.retry_attempts = defaultdict(int)
        self.max_retry_attempts = config.get('max_retry_attempts', 3)
        
        # Performance metrics
        self.collection_count = 0
        self.error_count = 0
        self.last_collection_time = None
        self.average_collection_time = 0.0
        
        # Synthetic data generators (for fallback and simulation)
        self.synthetic_generators = {}
        
        self.logger.info(f"Initialized SensorManager in {self.mode} mode")
    
    def register_thermal_sensor(self, sensor_id: str, sensor: ThermalSensorInterface) -> bool:
        """
        Register a thermal sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            sensor: Thermal sensor instance
            
        Returns:
            True if registration successful
        """
        try:
            self.thermal_sensors[sensor_id] = sensor
            self.all_sensors[sensor_id] = sensor
            self.sensor_health[sensor_id] = {
                'status': 'registered',
                'last_reading': None,
                'error_count': 0,
                'last_error': None
            }
            
            self.logger.info(f"Registered thermal sensor: {sensor_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register thermal sensor {sensor_id}: {str(e)}")
            return False
    
    def register_gas_sensor(self, sensor_id: str, sensor: GasSensorInterface) -> bool:
        """
        Register a gas sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            sensor: Gas sensor instance
            
        Returns:
            True if registration successful
        """
        try:
            self.gas_sensors[sensor_id] = sensor
            self.all_sensors[sensor_id] = sensor
            self.sensor_health[sensor_id] = {
                'status': 'registered',
                'last_reading': None,
                'error_count': 0,
                'last_error': None
            }
            
            self.logger.info(f"Registered gas sensor: {sensor_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register gas sensor {sensor_id}: {str(e)}")
            return False
    
    def register_environmental_sensor(self, sensor_id: str, sensor: EnvironmentalSensorInterface) -> bool:
        """
        Register an environmental sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            sensor: Environmental sensor instance
            
        Returns:
            True if registration successful
        """
        try:
            self.environmental_sensors[sensor_id] = sensor
            self.all_sensors[sensor_id] = sensor
            self.sensor_health[sensor_id] = {
                'status': 'registered',
                'last_reading': None,
                'error_count': 0,
                'last_error': None
            }
            
            self.logger.info(f"Registered environmental sensor: {sensor_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register environmental sensor {sensor_id}: {str(e)}")
            return False
    
    def initialize_sensors(self) -> Dict[str, bool]:
        """
        Initialize all registered sensors.
        
        Returns:
            Dictionary mapping sensor IDs to initialization success status
        """
        initialization_results = {}
        
        for sensor_id, sensor in self.all_sensors.items():
            try:
                self.logger.info(f"Initializing sensor: {sensor_id}")
                
                # Connect to sensor
                if sensor.connect():
                    self.sensor_health[sensor_id]['status'] = 'connected'
                    initialization_results[sensor_id] = True
                    self.logger.info(f"Successfully initialized sensor: {sensor_id}")
                else:
                    self.sensor_health[sensor_id]['status'] = 'connection_failed'
                    initialization_results[sensor_id] = False
                    self.logger.error(f"Failed to connect to sensor: {sensor_id}")
                
            except Exception as e:
                self.sensor_health[sensor_id]['status'] = 'initialization_error'
                self.sensor_health[sensor_id]['last_error'] = str(e)
                initialization_results[sensor_id] = False
                self.logger.error(f"Error initializing sensor {sensor_id}: {str(e)}")
        
        # Setup synthetic fallbacks if needed
        if self.auto_fallback and self.mode in [SensorMode.HYBRID, SensorMode.REAL]:
            self._setup_synthetic_fallbacks()
        
        successful_init = sum(1 for success in initialization_results.values() if success)
        total_sensors = len(initialization_results)
        
        self.logger.info(f"Sensor initialization complete: {successful_init}/{total_sensors} successful")
        
        return initialization_results
    
    def start_data_collection(self) -> bool:
        """
        Start continuous data collection from all sensors.
        
        Returns:
            True if data collection started successfully
        """
        if self.is_collecting:
            self.logger.warning("Data collection already running")
            return True
        
        try:
            self.is_collecting = True
            self.stop_event.clear()
            
            # Start collection thread
            self.collection_thread = threading.Thread(target=self._data_collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            
            self.logger.info("Started continuous data collection")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start data collection: {str(e)}")
            self.is_collecting = False
            return False
    
    def stop_data_collection(self) -> None:
        """Stop continuous data collection."""
        try:
            self.logger.info("Stopping data collection...")
            
            self.is_collecting = False
            self.stop_event.set()
            
            # Wait for collection thread to finish
            if self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=5.0)
            
            self.logger.info("Data collection stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping data collection: {str(e)}")
    
    def read_all_sensors(self) -> Dict[str, Any]:
        """
        Read data from all sensors synchronously.
        
        Returns:
            Dictionary containing data from all sensor types
        """
        collection_start = time.time()
        
        try:
            # Collect thermal data
            thermal_data = self._collect_thermal_data()
            
            # Collect gas data
            gas_data = self._collect_gas_data()
            
            # Collect environmental data
            environmental_data = self._collect_environmental_data()
            
            # Compile sensor readings
            sensor_data = {
                'thermal': thermal_data,
                'gas': gas_data,
                'environmental': environmental_data,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'collection_time_ms': (time.time() - collection_start) * 1000,
                    'mode': self.mode,
                    'sensor_count': {
                        'thermal': len(self.thermal_sensors),
                        'gas': len(self.gas_sensors),
                        'environmental': len(self.environmental_sensors)
                    },
                    'health_status': self._get_overall_health_status()
                }
            }
            
            # Update performance metrics
            self._update_performance_metrics(collection_start)
            
            return sensor_data
            
        except Exception as e:
            self.logger.error(f"Error reading sensors: {str(e)}")
            self.error_count += 1
            return self._get_fallback_data()
    
    def get_sensor_health(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of all sensors.
        
        Returns:
            Dictionary containing health information for all sensors
        """
        return {
            'overall_status': self._get_overall_health_status(),
            'mode': self.mode,
            'sensors': self.sensor_health.copy(),
            'failed_sensors': list(self.failed_sensors),
            'performance_metrics': {
                'collection_count': self.collection_count,
                'error_count': self.error_count,
                'error_rate': self.error_count / max(1, self.collection_count),
                'average_collection_time': self.average_collection_time,
                'last_collection': self.last_collection_time
            },
            'retry_attempts': dict(self.retry_attempts)
        }
    
    def switch_mode(self, new_mode: str) -> bool:
        """
        Switch sensor operating mode.
        
        Args:
            new_mode: New operating mode (synthetic, real, or hybrid)
            
        Returns:
            True if mode switch successful
        """
        if new_mode not in [SensorMode.SYNTHETIC, SensorMode.REAL, SensorMode.HYBRID, SensorMode.SIMULATION]:
            self.logger.error(f"Invalid sensor mode: {new_mode}")
            return False
        
        try:
            old_mode = self.mode
            self.mode = new_mode
            
            self.logger.info(f"Switched sensor mode from {old_mode} to {new_mode}")
            
            # Reconfigure sensors for new mode
            if new_mode == SensorMode.SYNTHETIC:
                self._setup_synthetic_fallbacks()
            elif new_mode == SensorMode.REAL and old_mode == SensorMode.SYNTHETIC:
                # Attempt to reconnect to real sensors
                self.initialize_sensors()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch mode to {new_mode}: {str(e)}")
            self.mode = old_mode  # Revert on failure
            return False
    
    def calibrate_sensors(self, sensor_ids: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Calibrate specified sensors or all sensors.
        
        Args:
            sensor_ids: Optional list of sensor IDs to calibrate
            
        Returns:
            Dictionary mapping sensor IDs to calibration success status
        """
        if sensor_ids is None:
            sensor_ids = list(self.all_sensors.keys())
        
        calibration_results = {}
        
        for sensor_id in sensor_ids:
            if sensor_id not in self.all_sensors:
                calibration_results[sensor_id] = False
                continue
            
            try:
                sensor = self.all_sensors[sensor_id]
                success = sensor.calibrate()
                calibration_results[sensor_id] = success
                
                if success:
                    self.logger.info(f"Successfully calibrated sensor: {sensor_id}")
                else:
                    self.logger.warning(f"Calibration failed for sensor: {sensor_id}")
                    
            except Exception as e:
                self.logger.error(f"Error calibrating sensor {sensor_id}: {str(e)}")
                calibration_results[sensor_id] = False
        
        return calibration_results
    
    def _data_collection_loop(self) -> None:
        """Main data collection loop running in separate thread."""
        self.logger.info("Starting data collection loop")
        
        while self.is_collecting and not self.stop_event.is_set():
            try:
                # Collect sensor data
                sensor_data = self.read_all_sensors()
                
                # Add to buffer
                self.data_buffer.append(sensor_data)
                
                # Wait for next collection
                self.stop_event.wait(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in data collection loop: {str(e)}")
                self.stop_event.wait(1.0)  # Wait longer on error
        
        self.logger.info("Data collection loop stopped")
    
    def _collect_thermal_data(self) -> Dict[str, Any]:
        """Collect data from all thermal sensors."""
        thermal_data = {}
        
        for sensor_id, sensor in self.thermal_sensors.items():
            try:
                if sensor_id in self.failed_sensors:
                    # Use synthetic fallback
                    thermal_data[sensor_id] = self._get_synthetic_thermal_data(sensor_id)
                else:
                    # Try to read from real sensor
                    reading = sensor.read()
                    thermal_data[sensor_id] = reading
                    
                    # Update health status
                    self.sensor_health[sensor_id]['last_reading'] = datetime.now().isoformat()
                    self.sensor_health[sensor_id]['status'] = 'active'
                
            except Exception as e:
                self.logger.warning(f"Error reading thermal sensor {sensor_id}: {str(e)}")
                self._handle_sensor_error(sensor_id, e)
                
                # Use fallback data
                thermal_data[sensor_id] = self._get_synthetic_thermal_data(sensor_id)
        
        return thermal_data
    
    def _collect_gas_data(self) -> Dict[str, Any]:
        """Collect data from all gas sensors."""
        gas_data = {}
        
        for sensor_id, sensor in self.gas_sensors.items():
            try:
                if sensor_id in self.failed_sensors:
                    # Use synthetic fallback
                    gas_data[sensor_id] = self._get_synthetic_gas_data(sensor_id)
                else:
                    # Try to read from real sensor
                    reading = sensor.read()
                    gas_data[sensor_id] = reading
                    
                    # Update health status
                    self.sensor_health[sensor_id]['last_reading'] = datetime.now().isoformat()
                    self.sensor_health[sensor_id]['status'] = 'active'
                
            except Exception as e:
                self.logger.warning(f"Error reading gas sensor {sensor_id}: {str(e)}")
                self._handle_sensor_error(sensor_id, e)
                
                # Use fallback data
                gas_data[sensor_id] = self._get_synthetic_gas_data(sensor_id)
        
        return gas_data
    
    def _collect_environmental_data(self) -> Dict[str, Any]:
        """Collect data from all environmental sensors."""
        environmental_data = {}
        
        for sensor_id, sensor in self.environmental_sensors.items():
            try:
                if sensor_id in self.failed_sensors:
                    # Use synthetic fallback
                    environmental_data[sensor_id] = self._get_synthetic_environmental_data(sensor_id)
                else:
                    # Try to read from real sensor
                    reading = sensor.read()
                    environmental_data[sensor_id] = reading
                    
                    # Update health status
                    self.sensor_health[sensor_id]['last_reading'] = datetime.now().isoformat()
                    self.sensor_health[sensor_id]['status'] = 'active'
                
            except Exception as e:
                self.logger.warning(f"Error reading environmental sensor {sensor_id}: {str(e)}")
                self._handle_sensor_error(sensor_id, e)
                
                # Use fallback data
                environmental_data[sensor_id] = self._get_synthetic_environmental_data(sensor_id)
        
        return environmental_data
    
    def _handle_sensor_error(self, sensor_id: str, error: Exception) -> None:
        """Handle sensor reading errors with retry logic."""
        # Update health status
        self.sensor_health[sensor_id]['error_count'] += 1
        self.sensor_health[sensor_id]['last_error'] = str(error)
        self.sensor_health[sensor_id]['status'] = 'error'
        
        # Increment retry attempts
        self.retry_attempts[sensor_id] += 1
        
        # Mark as failed if too many attempts
        if self.retry_attempts[sensor_id] >= self.max_retry_attempts:
            self.failed_sensors.add(sensor_id)
            self.logger.error(f"Sensor {sensor_id} marked as failed after {self.max_retry_attempts} attempts")
        else:
            self.logger.warning(f"Sensor {sensor_id} error (attempt {self.retry_attempts[sensor_id]}): {str(error)}")
    
    def _setup_synthetic_fallbacks(self) -> None:
        """Setup synthetic data generators as fallbacks."""
        try:
            # Import synthetic generators
            from ..data_generation.thermal.thermal_image_generator import ThermalImageGenerator
            from ..data_generation.gas.gas_concentration_generator import GasConcentrationGenerator
            from ..data_generation.environmental.environmental_data_generator import EnvironmentalDataGenerator
            
            # Setup thermal generator
            thermal_config = self.config.get('synthetic_thermal', {})
            self.synthetic_generators['thermal'] = ThermalImageGenerator(thermal_config)
            
            # Setup gas generator
            gas_config = self.config.get('synthetic_gas', {})
            self.synthetic_generators['gas'] = GasConcentrationGenerator(gas_config)
            
            # Setup environmental generator
            env_config = self.config.get('synthetic_environmental', {})
            self.synthetic_generators['environmental'] = EnvironmentalDataGenerator(env_config)
            
            self.logger.info("Synthetic fallback generators initialized")
            
        except ImportError as e:
            self.logger.warning(f"Could not import synthetic generators: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error setting up synthetic fallbacks: {str(e)}")
    
    def _get_synthetic_thermal_data(self, sensor_id: str) -> Dict[str, Any]:
        """Generate synthetic thermal data as fallback."""
        try:
            if 'thermal' in self.synthetic_generators:
                generator = self.synthetic_generators['thermal']
                synthetic_data = generator.generate_frame()
                return synthetic_data
        except Exception as e:
            self.logger.warning(f"Error generating synthetic thermal data: {str(e)}")
        
        # Basic fallback data
        return {
            'temperature_max': 22.0 + np.random.normal(0, 2),
            'temperature_avg': 20.0 + np.random.normal(0, 1),
            'temperature_min': 18.0 + np.random.normal(0, 1),
            'hotspot_count': 0,
            'thermal_image': np.random.normal(20, 2, (32, 32)),
            'timestamp': datetime.now().isoformat(),
            'source': 'synthetic_fallback'
        }
    
    def _get_synthetic_gas_data(self, sensor_id: str) -> Dict[str, Any]:
        """Generate synthetic gas data as fallback."""
        try:
            if 'gas' in self.synthetic_generators:
                generator = self.synthetic_generators['gas']
                synthetic_data = generator.generate_readings()
                return synthetic_data
        except Exception as e:
            self.logger.warning(f"Error generating synthetic gas data: {str(e)}")
        
        # Basic fallback data
        return {
            'co_concentration': 5.0 + np.random.normal(0, 1),
            'co2_concentration': 400.0 + np.random.normal(0, 20),
            'smoke_density': 10.0 + np.random.normal(0, 2),
            'voc_total': 200.0 + np.random.normal(0, 50),
            'timestamp': datetime.now().isoformat(),
            'source': 'synthetic_fallback'
        }
    
    def _get_synthetic_environmental_data(self, sensor_id: str) -> Dict[str, Any]:
        """Generate synthetic environmental data as fallback."""
        try:
            if 'environmental' in self.synthetic_generators:
                generator = self.synthetic_generators['environmental']
                synthetic_data = generator.generate_readings()
                return synthetic_data
        except Exception as e:
            self.logger.warning(f"Error generating synthetic environmental data: {str(e)}")
        
        # Basic fallback data
        return {
            'temperature': 22.0 + np.random.normal(0, 1),
            'humidity': 50.0 + np.random.normal(0, 5),
            'pressure': 1013.0 + np.random.normal(0, 2),
            'wind_speed': 1.0 + np.random.normal(0, 0.2),
            'timestamp': datetime.now().isoformat(),
            'source': 'synthetic_fallback'
        }
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """Get complete fallback data when all else fails."""
        return {
            'thermal': self._get_synthetic_thermal_data('fallback'),
            'gas': self._get_synthetic_gas_data('fallback'),
            'environmental': self._get_synthetic_environmental_data('fallback'),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'mode': 'emergency_fallback',
                'source': 'synthetic_emergency'
            }
        }
    
    def _get_overall_health_status(self) -> str:
        """Calculate overall health status of the sensor system."""
        if not self.sensor_health:
            return 'no_sensors'
        
        active_sensors = sum(1 for health in self.sensor_health.values() 
                           if health['status'] == 'active')
        total_sensors = len(self.sensor_health)
        
        if active_sensors == total_sensors:
            return 'healthy'
        elif active_sensors > total_sensors * 0.7:
            return 'degraded'
        elif active_sensors > 0:
            return 'critical'
        else:
            return 'failed'
    
    def _update_performance_metrics(self, collection_start: float) -> None:
        """Update performance tracking metrics."""
        collection_time = time.time() - collection_start
        self.collection_count += 1
        self.last_collection_time = datetime.now().isoformat()
        
        # Update average collection time (exponential moving average)
        if self.average_collection_time == 0:
            self.average_collection_time = collection_time
        else:
            self.average_collection_time = 0.9 * self.average_collection_time + 0.1 * collection_time
    
    def get_recent_data(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent sensor data from buffer.
        
        Args:
            count: Number of recent data points to retrieve
            
        Returns:
            List of recent sensor data dictionaries
        """
        return list(self.data_buffer)[-count:] if self.data_buffer else []
    
    def export_data(self, filepath: str, format_type: str = 'json') -> bool:
        """
        Export collected sensor data to file.
        
        Args:
            filepath: Path to export file
            format_type: Export format ('json' or 'csv')
            
        Returns:
            True if export successful
        """
        try:
            data_to_export = list(self.data_buffer)
            
            if format_type == 'json':
                with open(filepath, 'w') as f:
                    json.dump(data_to_export, f, indent=2, default=str)
            
            elif format_type == 'csv':
                # Flatten data for CSV export
                flattened_data = []
                for data_point in data_to_export:
                    flat_point = {}
                    for sensor_type, sensor_data in data_point.items():
                        if isinstance(sensor_data, dict):
                            for key, value in sensor_data.items():
                                flat_point[f"{sensor_type}_{key}"] = value
                        else:
                            flat_point[sensor_type] = sensor_data
                    flattened_data.append(flat_point)
                
                df = pd.DataFrame(flattened_data)
                df.to_csv(filepath, index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            self.logger.info(f"Exported {len(data_to_export)} data points to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export data: {str(e)}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the sensor manager and disconnect all sensors."""
        try:
            self.logger.info("Shutting down sensor manager...")
            
            # Stop data collection
            self.stop_data_collection()
            
            # Disconnect all sensors
            for sensor_id, sensor in self.all_sensors.items():
                try:
                    sensor.disconnect()
                    self.logger.debug(f"Disconnected sensor: {sensor_id}")
                except Exception as e:
                    self.logger.warning(f"Error disconnecting sensor {sensor_id}: {str(e)}")
            
            self.logger.info("Sensor manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")


# Convenience function for creating a sensor manager
def create_sensor_manager(config: Optional[Dict[str, Any]] = None) -> SensorManager:
    """
    Create a sensor manager with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured SensorManager instance
    """
    default_config = {
        'mode': SensorMode.SYNTHETIC,
        'auto_fallback': True,
        'buffer_size': 1000,
        'collection_interval': 1.0,
        'max_retry_attempts': 3,
        'synthetic_thermal': {
            'image_width': 32,
            'image_height': 32,
            'base_temperature': 20.0
        },
        'synthetic_gas': {
            'base_co': 5.0,
            'base_co2': 400.0,
            'base_smoke': 10.0
        },
        'synthetic_environmental': {
            'base_temperature': 22.0,
            'base_humidity': 50.0,
            'base_pressure': 1013.0
        }
    }
    
    if config:
        default_config.update(config)
    
    return SensorManager(default_config)