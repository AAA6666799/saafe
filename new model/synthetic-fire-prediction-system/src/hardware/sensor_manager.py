"""
FLIR Lepton 3.5 + SCD41 Sensor Manager.

This module provides a hardware abstraction layer for FLIR thermal imaging
and SCD41 CO₂ sensors, supporting both real hardware and synthetic
data generation for testing and development.
"""

import logging
import numpy as np
import time
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque, defaultdict
import json

from .base import SensorMode
from .specific.flir_lepton35_interface import FLIRLepton35Interface
from .specific.scd41_interface import SCD41Interface
from .specific.synthetic_flir_interface import SyntheticFLIRInterface
from .specific.synthetic_scd41_interface import SyntheticSCD41Interface

# Import MQTT handler
try:
    from .mqtt_handler import MqttHandler, create_mqtt_handler
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    MqttHandler = None
    create_mqtt_handler = None

# Import data validator
try:
    from ..data_quality.iot_data_validator import IoTDataValidator, create_iot_data_validator, DataQualityLevel
    DATA_VALIDATION_AVAILABLE = True
except ImportError:
    DATA_VALIDATION_AVAILABLE = False
    IoTDataValidator = None
    create_iot_data_validator = None
    DataQualityLevel = None

# Import device health monitor
try:
    from .device_health_monitor import DeviceHealthMonitor, create_device_health_monitor
    DEVICE_HEALTH_MONITORING_AVAILABLE = True
except ImportError:
    DEVICE_HEALTH_MONITORING_AVAILABLE = False
    DeviceHealthMonitor = None
    create_device_health_monitor = None

# Import fallback manager
try:
    from .fallback_manager import FallbackManager, create_fallback_manager
    FALLBACK_MANAGER_AVAILABLE = True
except ImportError:
    FALLBACK_MANAGER_AVAILABLE = False
    FallbackManager = None
    create_fallback_manager = None

# Import data logger
try:
    from ..data_logging.sensor_data_logger import SensorDataLogger, create_sensor_data_logger
    DATA_LOGGING_AVAILABLE = True
except ImportError:
    DATA_LOGGING_AVAILABLE = False
    SensorDataLogger = None
    create_sensor_data_logger = None


class SensorManager:
    """
    FLIR Lepton 3.5 + SCD41 sensor manager for hardware abstraction.
    
    This manager provides a unified interface for FLIR thermal imaging
    and SCD41 CO₂ sensors, supporting both real hardware and synthetic
    data generation for testing and development.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FLIR + SCD41 sensor manager.
        
        Args:
            config: Configuration dictionary containing sensor settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Operating mode (synthetic, real, or hybrid)
        self.mode = config.get('mode', SensorMode.SYNTHETIC)
        self.auto_fallback = config.get('auto_fallback', True)
        
        # FLIR Lepton 3.5 + SCD41 specific registries
        self.flir_sensors = {}  # FLIR Lepton 3.5 thermal sensors
        self.scd41_sensors = {}  # SCD41 CO₂ sensors
        self.all_sensors = {}   # Combined registry
        
        # MQTT communication for IoT devices
        self.mqtt_client = None
        self.mqtt_config = config.get('mqtt', {
            'broker': 'localhost',
            'port': 1883,
            'topics': {
                'flir': 'sensors/flir/+/data',
                'scd41': 'sensors/scd41/+/data'
            }
        })
        self.mqtt_enabled = self.mqtt_config.get('enabled', False)
        
        # Data collection and buffering
        self.is_collecting = False
        self.collection_thread = None
        self.stop_event = threading.Event()
        self.data_buffer = deque(maxlen=config.get('buffer_size', 1000))
        self.collection_interval = config.get('collection_interval', 1.0)
        
        # Performance tracking
        self.collection_count = 0
        self.error_count = 0
        self.average_collection_time = 0.0
        self.last_collection_time = None
        self.flir_frame_count = 0
        self.scd41_reading_count = 0
        
        # Health monitoring
        self.sensor_health = defaultdict(lambda: {
            'status': 'unknown',
            'last_reading': None,
            'error_count': 0,
            'last_error': None,
            'frames_captured': 0,
            'readings_taken': 0
        })
        self.failed_sensors = set()
        self.retry_attempts = defaultdict(int)
        self.max_retry_attempts = config.get('max_retry_attempts', 3)
        
        # Device health monitoring
        self.device_health_monitor = None
        self.enable_device_health_monitoring = config.get('enable_device_health_monitoring', False)
        if self.enable_device_health_monitoring and DEVICE_HEALTH_MONITORING_AVAILABLE:
            try:
                self.device_health_monitor = create_device_health_monitor(config.get('device_health_config', {}))
                self.logger.info("Device health monitoring enabled")
            except Exception as e:
                self.logger.error(f"Failed to initialize device health monitor: {str(e)}")
        
        # Fallback management
        self.fallback_manager = None
        self.enable_fallback_management = config.get('enable_fallback_management', True)
        if self.enable_fallback_management and FALLBACK_MANAGER_AVAILABLE:
            try:
                self.fallback_manager = create_fallback_manager(config.get('fallback_config', {}))
                self.logger.info("Fallback management enabled")
            except Exception as e:
                self.logger.error(f"Failed to initialize fallback manager: {str(e)}")
        
        # Data logging
        self.data_logger = None
        self.enable_data_logging = config.get('enable_data_logging', False)
        if self.enable_data_logging and DATA_LOGGING_AVAILABLE:
            try:
                self.data_logger = create_sensor_data_logger(config.get('data_logging_config', {}))
                self.logger.info("Data logging enabled")
            except Exception as e:
                self.logger.error(f"Failed to initialize data logger: {str(e)}")
        
        # Synthetic data generators for fallback
        self.synthetic_generators = {}
        
        # Data validation
        self.data_validator = None
        self.enable_data_validation = config.get('enable_data_validation', False)
        if self.enable_data_validation and DATA_VALIDATION_AVAILABLE:
            try:
                self.data_validator = create_iot_data_validator(config.get('data_validation_config', {}))
                self.logger.info("Data validation enabled")
            except Exception as e:
                self.logger.error(f"Failed to initialize data validator: {str(e)}")
        
        self.logger.info("Initialized FLIR + SCD41 Sensor Manager")
        self.logger.info(f"Operating mode: {self.mode}")
        self.logger.info(f"MQTT enabled: {self.mqtt_enabled}")
        self.logger.info(f"Data validation enabled: {self.enable_data_validation}")
        self.logger.info(f"Device health monitoring enabled: {self.enable_device_health_monitoring}")
        self.logger.info(f"Fallback management enabled: {self.enable_fallback_management}")
        self.logger.info(f"Data logging enabled: {self.enable_data_logging}")
    
    def initialize_sensors(self) -> Dict[str, bool]:
        """
        Initialize all FLIR and SCD41 sensors based on configuration.
        
        Returns:
            Dictionary mapping sensor IDs to initialization success status
        """
        self.logger.info("Initializing FLIR + SCD41 sensors...")
        
        initialization_results = {}
        
        # Initialize FLIR Lepton 3.5 thermal sensors
        flir_configs = self.config.get('flir_sensors', {})
        for sensor_id, sensor_config in flir_configs.items():
            try:
                if self.mode == SensorMode.SYNTHETIC:
                    sensor = SyntheticFLIRInterface(sensor_config)
                else:
                    sensor = FLIRLepton35Interface(sensor_config)
                
                if sensor.connect():
                    self.flir_sensors[sensor_id] = sensor
                    self.all_sensors[sensor_id] = sensor
                    self.sensor_health[sensor_id]['status'] = 'initialized'
                    initialization_results[sensor_id] = True
                    
                    # Register device with health monitor
                    if self.device_health_monitor:
                        self.device_health_monitor.register_device(sensor_id, 'flir', sensor_config)
                    
                    self.logger.info(f"Successfully initialized FLIR sensor: {sensor_id}")
                else:
                    self.sensor_health[sensor_id]['status'] = 'connection_failed'
                    initialization_results[sensor_id] = False
                    self.logger.error(f"Failed to connect to FLIR sensor: {sensor_id}")
                
            except Exception as e:
                self.sensor_health[sensor_id]['status'] = 'initialization_error'
                self.sensor_health[sensor_id]['last_error'] = str(e)
                initialization_results[sensor_id] = False
                self.logger.error(f"Error initializing FLIR sensor {sensor_id}: {str(e)}")
        
        # Initialize SCD41 CO₂ sensors
        scd41_configs = self.config.get('scd41_sensors', {})
        for sensor_id, sensor_config in scd41_configs.items():
            try:
                if self.mode == SensorMode.SYNTHETIC:
                    sensor = SyntheticSCD41Interface(sensor_config)
                else:
                    sensor = SCD41Interface(sensor_config)
                
                if sensor.connect():
                    self.scd41_sensors[sensor_id] = sensor
                    self.all_sensors[sensor_id] = sensor
                    self.sensor_health[sensor_id]['status'] = 'initialized'
                    initialization_results[sensor_id] = True
                    
                    # Register device with health monitor
                    if self.device_health_monitor:
                        self.device_health_monitor.register_device(sensor_id, 'scd41', sensor_config)
                    
                    self.logger.info(f"Successfully initialized SCD41 sensor: {sensor_id}")
                else:
                    self.sensor_health[sensor_id]['status'] = 'connection_failed'
                    initialization_results[sensor_id] = False
                    self.logger.error(f"Failed to connect to SCD41 sensor: {sensor_id}")
                
            except Exception as e:
                self.sensor_health[sensor_id]['status'] = 'initialization_error'
                self.sensor_health[sensor_id]['last_error'] = str(e)
                initialization_results[sensor_id] = False
                self.logger.error(f"Error initializing SCD41 sensor {sensor_id}: {str(e)}")
        
        # Initialize MQTT client if enabled
        if self.mqtt_enabled and MQTT_AVAILABLE:
            try:
                self.mqtt_client = create_mqtt_handler(self.mqtt_config)
                
                # Set callbacks
                self.mqtt_client.set_data_callback(self._on_mqtt_data_received)
                self.mqtt_client.set_connection_callback(self._on_mqtt_connection_change)
                self.mqtt_client.set_error_callback(self._on_mqtt_error)
                
                # Connect to broker
                if self.mqtt_client.connect():
                    self.logger.info("Successfully connected to MQTT broker")
                else:
                    self.logger.error("Failed to connect to MQTT broker")
                    
            except Exception as e:
                self.logger.error(f"Error initializing MQTT client: {str(e)}")
                self.mqtt_client = None
        
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
            
            # Stop MQTT client if active
            if self.mqtt_client:
                self.mqtt_client.disconnect()
            
            # Wait for collection thread to finish
            if self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=5.0)
            
            self.logger.info("Data collection stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping data collection: {str(e)}")
    
    def read_all_sensors(self) -> Dict[str, Any]:
        """
        Read data from all FLIR and SCD41 sensors synchronously.
        
        Returns:
            Dictionary containing data from FLIR and SCD41 sensors
        """
        collection_start = time.time()
        
        try:
            # Collect FLIR thermal data
            flir_data = self._collect_flir_data()
            
            # Collect SCD41 gas data
            scd41_data = self._collect_scd41_data()
            
            # Compile sensor readings in the expected format
            sensor_data = {
                'flir': flir_data,
                'scd41': scd41_data,
                # Backward compatibility aliases
                'thermal': flir_data,
                'gas': scd41_data,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'collection_time_ms': (time.time() - collection_start) * 1000,
                    'mode': self.mode,
                    'sensor_count': {
                        'flir': len(self.flir_sensors),
                        'scd41': len(self.scd41_sensors),
                        'total': len(self.all_sensors)
                    },
                    'health_status': self._get_overall_health_status(),
                    'system_type': 'flir_scd41'
                }
            }
            
            # Update performance metrics
            self._update_performance_metrics(collection_start)
            
            return sensor_data
            
        except Exception as e:
            self.logger.error(f"Error reading FLIR + SCD41 sensors: {str(e)}")
            self.error_count += 1
            return self._get_fallback_data()
    
    def get_sensor_health(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of all sensors.
        
        Returns:
            Dictionary containing health information for all sensors
        """
        # Get device health information if available
        device_health = {}
        if self.device_health_monitor:
            device_health = self.device_health_monitor.get_overall_health()
        
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
            'retry_attempts': dict(self.retry_attempts),
            'device_health': device_health
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
                
                # Log data if logging is enabled
                if self.data_logger:
                    try:
                        self.data_logger.log_sensor_data(sensor_data)
                    except Exception as e:
                        self.logger.warning(f"Failed to log sensor data: {str(e)}")
                
                # Wait for next collection
                self.stop_event.wait(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in data collection loop: {str(e)}")
                self.stop_event.wait(1.0)  # Wait longer on error
        
        self.logger.info("Data collection loop stopped")
    
    def _collect_flir_data(self) -> Dict[str, Any]:
        """Collect data from all FLIR Lepton 3.5 thermal sensors."""
        flir_data = {}
        failed_flir_sensors = []
        
        for sensor_id, sensor in self.flir_sensors.items():
            try:
                start_time = time.time()
                
                if sensor_id in self.failed_sensors:
                    failed_flir_sensors.append(sensor_id)
                    continue
                
                # Try to read from real FLIR sensor
                reading = sensor.read()
                
                # Record response time
                response_time = time.time() - start_time
                if self.device_health_monitor:
                    self.device_health_monitor.record_response_time(sensor_id, response_time)
                
                # Record reading with health monitor
                if self.device_health_monitor:
                    self.device_health_monitor.record_device_reading(sensor_id, reading)
                
                # Update fallback manager with good data
                if self.fallback_manager:
                    self.fallback_manager.update_sensor_data(sensor_id, 'flir', reading)
                
                # Ensure the reading has the expected format
                if isinstance(reading, dict):
                    # Add FLIR-specific metadata
                    reading['sensor_type'] = 'flir_lepton35'
                    reading['resolution'] = (160, 120)  # FLIR Lepton 3.5 resolution
                    if 'thermal_frame' not in reading and 'frame' in reading:
                        reading['thermal_frame'] = reading['frame']
                    flir_data[sensor_id] = reading
                else:
                    # Handle raw numpy array from sensor
                    flir_data[sensor_id] = {
                        'thermal_frame': reading,
                        'timestamp': datetime.now().isoformat(),
                        'sensor_type': 'flir_lepton35',
                        'resolution': reading.shape if hasattr(reading, 'shape') else (160, 120),
                        'temperature_stats': {
                            'min': float(np.min(reading)) if hasattr(reading, 'shape') else 20.0,
                            'max': float(np.max(reading)) if hasattr(reading, 'shape') else 25.0,
                            'mean': float(np.mean(reading)) if hasattr(reading, 'shape') else 22.0
                        }
                    }
                
                # Update health status
                self.sensor_health[sensor_id]['last_reading'] = datetime.now().isoformat()
                self.sensor_health[sensor_id]['status'] = 'active'
                self.sensor_health[sensor_id]['frames_captured'] += 1
                self.flir_frame_count += 1
                
            except Exception as e:
                self.logger.warning(f"Error reading FLIR sensor {sensor_id}: {str(e)}")
                self._handle_sensor_error(sensor_id, e)
                
                # Record error with health monitor
                if self.device_health_monitor:
                    self.device_health_monitor.record_device_error(sensor_id, str(e), "reading_error")
                
                failed_flir_sensors.append(sensor_id)
        
        # Handle failed FLIR sensors with fallback
        if failed_flir_sensors and self.fallback_manager:
            for sensor_id in failed_flir_sensors:
                failure_context = {
                    'sensor_id': sensor_id,
                    'sensor_type': 'flir',
                    'error': f'Failed to read from FLIR sensor {sensor_id}',
                    'timestamp': datetime.now().isoformat(),
                    'missing_sensors': {'flir': [sensor_id]}
                }
                
                fallback_data = self.fallback_manager.handle_sensor_failure(failure_context)
                if fallback_data and 'flir' in fallback_data and sensor_id in fallback_data['flir']:
                    flir_data[sensor_id] = fallback_data['flir'][sensor_id]
                    self.logger.info(f"Applied fallback data for failed FLIR sensor {sensor_id}")
                else:
                    # Use synthetic fallback as last resort
                    flir_data[sensor_id] = self._get_synthetic_flir_data(sensor_id)
        
        return flir_data
    
    def _collect_scd41_data(self) -> Dict[str, Any]:
        """Collect data from all SCD41 CO₂ sensors."""
        scd41_data = {}
        failed_scd41_sensors = []
        
        for sensor_id, sensor in self.scd41_sensors.items():
            try:
                start_time = time.time()
                
                if sensor_id in self.failed_sensors:
                    failed_scd41_sensors.append(sensor_id)
                    continue
                
                # Try to read from real SCD41 sensor
                reading = sensor.read()
                
                # Record response time
                response_time = time.time() - start_time
                if self.device_health_monitor:
                    self.device_health_monitor.record_response_time(sensor_id, response_time)
                
                # Record reading with health monitor
                if self.device_health_monitor:
                    self.device_health_monitor.record_device_reading(sensor_id, reading)
                
                # Update fallback manager with good data
                if self.fallback_manager:
                    self.fallback_manager.update_sensor_data(sensor_id, 'scd41', reading)
                
                # Ensure the reading has the expected format
                if isinstance(reading, dict):
                    # Add SCD41-specific metadata
                    reading['sensor_type'] = 'scd41_co2'
                    reading['measurement_range'] = (400, 40000)
                    # Ensure multiple naming conventions for compatibility
                    if 'co2_concentration' in reading:
                        reading['gas_reading'] = reading['co2_concentration']
                        reading['value'] = reading['co2_concentration']
                    scd41_data[sensor_id] = reading
                else:
                    # Handle raw CO₂ value from sensor
                    co2_value = float(reading)
                    scd41_data[sensor_id] = {
                        'co2_concentration': co2_value,
                        'gas_reading': co2_value,
                        'value': co2_value,
                        'timestamp': datetime.now().isoformat(),
                        'sensor_type': 'scd41_co2',
                        'measurement_range': (400, 40000),
                        'sensor_temp': 20.0,  # Default if not provided
                        'sensor_humidity': 45.0  # Default if not provided
                    }
                
                # Update health status
                self.sensor_health[sensor_id]['last_reading'] = datetime.now().isoformat()
                self.sensor_health[sensor_id]['status'] = 'active'
                self.sensor_health[sensor_id]['readings_taken'] += 1
                self.scd41_reading_count += 1
                
            except Exception as e:
                self.logger.warning(f"Error reading SCD41 sensor {sensor_id}: {str(e)}")
                self._handle_sensor_error(sensor_id, e)
                
                # Record error with health monitor
                if self.device_health_monitor:
                    self.device_health_monitor.record_device_error(sensor_id, str(e), "reading_error")
                
                failed_scd41_sensors.append(sensor_id)
        
        # Handle failed SCD41 sensors with fallback
        if failed_scd41_sensors and self.fallback_manager:
            for sensor_id in failed_scd41_sensors:
                failure_context = {
                    'sensor_id': sensor_id,
                    'sensor_type': 'scd41',
                    'error': f'Failed to read from SCD41 sensor {sensor_id}',
                    'timestamp': datetime.now().isoformat(),
                    'missing_sensors': {'scd41': [sensor_id]}
                }
                
                fallback_data = self.fallback_manager.handle_sensor_failure(failure_context)
                if fallback_data and 'scd41' in fallback_data and sensor_id in fallback_data['scd41']:
                    scd41_data[sensor_id] = fallback_data['scd41'][sensor_id]
                    self.logger.info(f"Applied fallback data for failed SCD41 sensor {sensor_id}")
                else:
                    # Use synthetic fallback as last resort
                    scd41_data[sensor_id] = self._get_synthetic_scd41_data(sensor_id)
        
        return scd41_data
    
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
    
    def _get_synthetic_scd41_data(self, sensor_id: str) -> Dict[str, Any]:
        """Generate synthetic SCD41 data as fallback."""
        try:
            if 'gas' in self.synthetic_generators:
                generator = self.synthetic_generators['gas']
                synthetic_data = generator.generate_readings()
                # Ensure SCD41-specific format
                if 'co2_concentration' in synthetic_data:
                    synthetic_data['gas_val'] = synthetic_data['co2_concentration']
                    synthetic_data['gas_delta'] = synthetic_data.get('co2_delta', 0.0)
                    synthetic_data['gas_vel'] = synthetic_data.get('co2_velocity', 0.0)
                return synthetic_data
        except Exception as e:
            self.logger.warning(f"Error generating synthetic SCD41 data: {str(e)}")
        
        # Basic fallback data in SCD41 format
        co2_value = 400.0 + np.random.normal(0, 20)
        return {
            'gas_val': co2_value,
            'gas_delta': np.random.normal(0, 5),
            'gas_vel': np.random.normal(0, 2),
            'co2_concentration': co2_value,
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
        fallback_data = {
            'flir': {},
            'scd41': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'mode': 'emergency_fallback',
                'source': 'synthetic_emergency'
            }
        }
        
        # Try to use fallback manager if available
        if self.fallback_manager:
            failure_context = {
                'error': 'Complete system failure - all sensors unavailable',
                'timestamp': datetime.now().isoformat(),
                'missing_sensors': {
                    'flir': list(self.flir_sensors.keys()),
                    'scd41': list(self.scd41_sensors.keys())
                }
            }
            
            try:
                managed_fallback = self.fallback_manager.handle_sensor_failure(failure_context)
                if managed_fallback:
                    # Merge with our structure
                    fallback_data.update(managed_fallback)
                    return fallback_data
            except Exception as e:
                self.logger.warning(f"Fallback manager failed: {str(e)}")
        
        # Fallback to synthetic data generators
        for sensor_id in self.flir_sensors.keys():
            fallback_data['flir'][sensor_id] = self._get_synthetic_flir_data(sensor_id)
        
        for sensor_id in self.scd41_sensors.keys():
            fallback_data['scd41'][sensor_id] = self._get_synthetic_scd41_data(sensor_id)
        
        return fallback_data
    
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
            
            # Stop device health monitoring
            if self.device_health_monitor:
                self.device_health_monitor.stop_monitoring()
            
            # Shutdown data logger
            if self.data_logger:
                self.data_logger.shutdown()
            
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
    
    def _on_mqtt_data_received(self, data: Dict[str, Any]) -> None:
        """
        Handle MQTT data received callback.
        
        Args:
            data: Sensor data received via MQTT
        """
        self.logger.debug(f"MQTT data received: {data}")
        
        # Log data if logging is enabled
        if self.data_logger:
            try:
                self.data_logger.log_sensor_data(data)
            except Exception as e:
                self.logger.warning(f"Failed to log MQTT sensor data: {str(e)}")
        
        # Validate data if validation is enabled
        if self.data_validator and self.enable_data_validation:
            try:
                validation_result = self.data_validator.validate_sensor_data(data)
                
                if not validation_result.is_valid:
                    self.logger.warning(f"Data validation failed: {validation_result.issues}")
                    if validation_result.recommendations:
                        self.logger.info(f"Recommendations: {validation_result.recommendations}")
                
                # Log quality level
                self.logger.info(f"Data quality level: {validation_result.quality_level.value}")
                
                # Only add to buffer if data quality is acceptable or better
                if validation_result.quality_level in [DataQualityLevel.EXCELLENT, DataQualityLevel.GOOD, DataQualityLevel.ACCEPTABLE]:
                    # Add to data buffer
                    self.data_buffer.append(data)
                    self.logger.debug("Data added to buffer")
                else:
                    self.logger.warning("Poor quality data rejected and not added to buffer")
                    return  # Don't add to buffer
                    
            except Exception as e:
                self.logger.error(f"Error during data validation: {str(e)}")
                # Still add data to buffer in case of validation errors to avoid data loss
                self.data_buffer.append(data)
                self.logger.debug("Data added to buffer despite validation error")
        else:
            # Add to data buffer
            self.data_buffer.append(data)
            self.logger.debug("Data added to buffer (validation disabled)")
    
    def _on_mqtt_connection_change(self, connected: bool) -> None:
        """
        Handle MQTT connection status change callback.
        
        Args:
            connected: True if connected, False if disconnected
        """
        if connected:
            self.logger.info("MQTT client connected")
        else:
            self.logger.info("MQTT client disconnected")
    
    def _on_mqtt_error(self, error: str) -> None:
        """
        Handle MQTT error callback.
        
        Args:
            error: Error message
        """
        self.logger.error(f"MQTT error: {error}")


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