"""
MQTT Handler for FLIR+SCD41 IoT Sensor Data.

This module provides MQTT communication capabilities for receiving
data from FLIR Lepton 3.5 thermal cameras and SCD41 CO₂ sensors.
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Try to import MQTT library
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    # Create a mock MQTT client for development
    class MockMQTTClient:
        def __init__(self, *args, **kwargs):
            pass
        
        def connect(self, *args, **kwargs):
            return 0
        
        def subscribe(self, *args, **kwargs):
            pass
        
        def loop_start(self):
            pass
        
        def loop_stop(self):
            pass
        
        def disconnect(self):
            pass
    
    mqtt = type('mqtt', (), {'Client': MockMQTTClient})
    MQTT_AVAILABLE = False

logger = logging.getLogger(__name__)


class MqttHandler:
    """
    MQTT handler for FLIR+SCD41 sensor data ingestion.
    
    This class handles MQTT communication with IoT sensors,
    receiving and processing data from FLIR thermal cameras
    and SCD41 CO₂ sensors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MQTT handler.
        
        Args:
            config: Configuration dictionary containing MQTT settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # MQTT configuration
        self.broker = config.get('broker', 'localhost')
        self.port = config.get('port', 1883)
        self.username = config.get('username')
        self.password = config.get('password')
        self.client_id = config.get('client_id', f"fire_detection_client_{int(time.time())}")
        
        # Topic configuration for FLIR and SCD41 sensors
        self.topics = config.get('topics', {
            'flir': 'sensors/flir/+/data',
            'scd41': 'sensors/scd41/+/data'
        })
        
        # Connection state
        self.connected = False
        self.client = None
        self.stop_event = threading.Event()
        
        # Data storage
        self.latest_flir_data = {}
        self.latest_scd41_data = {}
        self.data_lock = threading.Lock()
        
        # Callbacks
        self.data_callback = None
        self.connection_callback = None
        self.error_callback = None
        
        # Initialize MQTT client if available
        if MQTT_AVAILABLE:
            self._initialize_client()
        else:
            self.logger.warning("MQTT library not available. Using mock client.")
    
    def _initialize_client(self) -> None:
        """Initialize the MQTT client."""
        try:
            self.client = mqtt.Client(client_id=self.client_id)
            
            # Set authentication if provided
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            self.client.on_log = self._on_log
            
            self.logger.info(f"MQTT client initialized for broker {self.broker}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MQTT client: {str(e)}")
            self.client = None
    
    def _on_connect(self, client, userdata, flags, rc) -> None:
        """Handle MQTT connection event."""
        if rc == 0:
            self.connected = True
            self.logger.info(f"Connected to MQTT broker {self.broker}:{self.port}")
            
            # Subscribe to topics
            for sensor_type, topic in self.topics.items():
                client.subscribe(topic)
                self.logger.info(f"Subscribed to topic: {topic}")
            
            # Call connection callback if provided
            if self.connection_callback:
                self.connection_callback(True)
        else:
            self.logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
            if self.connection_callback:
                self.connection_callback(False)
    
    def _on_message(self, client, userdata, msg) -> None:
        """Handle incoming MQTT messages."""
        try:
            # Parse message payload
            payload = msg.payload.decode('utf-8')
            data = json.loads(payload)
            
            # Extract sensor ID from topic
            topic_parts = msg.topic.split('/')
            if len(topic_parts) >= 3:
                sensor_id = topic_parts[2]
            else:
                sensor_id = "unknown"
            
            # Process data based on topic
            if 'flir' in msg.topic:
                with self.data_lock:
                    self.latest_flir_data[sensor_id] = data
                    self.logger.debug(f"Received FLIR data from {sensor_id}: {data}")
            
            elif 'scd41' in msg.topic:
                with self.data_lock:
                    self.latest_scd41_data[sensor_id] = data
                    self.logger.debug(f"Received SCD41 data from {sensor_id}: {data}")
            
            # Call data callback if provided
            if self.data_callback:
                # Format data in the expected structure
                formatted_data = {
                    'flir': self.latest_flir_data.copy(),
                    'scd41': self.latest_scd41_data.copy(),
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'source': 'mqtt',
                        'sensor_count': {
                            'flir': len(self.latest_flir_data),
                            'scd41': len(self.latest_scd41_data)
                        }
                    }
                }
                self.data_callback(formatted_data)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse MQTT message payload: {str(e)}")
            if self.error_callback:
                self.error_callback(f"JSON decode error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {str(e)}")
            if self.error_callback:
                self.error_callback(f"Message processing error: {str(e)}")
    
    def _on_disconnect(self, client, userdata, rc) -> None:
        """Handle MQTT disconnection event."""
        self.connected = False
        self.logger.info("Disconnected from MQTT broker")
        
        if self.connection_callback:
            self.connection_callback(False)
    
    def _on_log(self, client, userdata, level, buf) -> None:
        """Handle MQTT client logs."""
        if level == mqtt.MQTT_LOG_ERR:
            self.logger.error(f"MQTT Error: {buf}")
        elif level == mqtt.MQTT_LOG_WARNING:
            self.logger.warning(f"MQTT Warning: {buf}")
        else:
            self.logger.debug(f"MQTT Log: {buf}")
    
    def connect(self) -> bool:
        """
        Connect to the MQTT broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not MQTT_AVAILABLE:
            self.logger.warning("MQTT library not available. Cannot connect.")
            return False
        
        if not self.client:
            self.logger.error("MQTT client not initialized")
            return False
        
        try:
            self.logger.info(f"Connecting to MQTT broker {self.broker}:{self.port}")
            result = self.client.connect(self.broker, self.port, 60)
            
            if result == 0:
                self.client.loop_start()
                return True
            else:
                self.logger.error(f"Failed to connect to MQTT broker. Result code: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to MQTT broker: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if self.client and self.connected:
            try:
                self.client.loop_stop()
                self.client.disconnect()
                self.connected = False
                self.logger.info("Disconnected from MQTT broker")
            except Exception as e:
                self.logger.error(f"Error disconnecting from MQTT broker: {str(e)}")
    
    def set_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Set callback function for received data.
        
        Args:
            callback: Function to call when data is received
        """
        self.data_callback = callback
    
    def set_connection_callback(self, callback: Callable[[bool], None]) -> None:
        """
        Set callback function for connection status changes.
        
        Args:
            callback: Function to call when connection status changes
        """
        self.connection_callback = callback
    
    def set_error_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set callback function for errors.
        
        Args:
            callback: Function to call when errors occur
        """
        self.error_callback = callback
    
    def get_latest_data(self) -> Dict[str, Any]:
        """
        Get the latest sensor data.
        
        Returns:
            Dictionary containing latest FLIR and SCD41 data
        """
        with self.data_lock:
            return {
                'flir': self.latest_flir_data.copy(),
                'scd41': self.latest_scd41_data.copy(),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'source': 'mqtt',
                    'sensor_count': {
                        'flir': len(self.latest_flir_data),
                        'scd41': len(self.latest_scd41_data)
                    }
                }
            }
    
    def is_connected(self) -> bool:
        """
        Check if connected to MQTT broker.
        
        Returns:
            True if connected, False otherwise
        """
        return self.connected


# Convenience function for creating MQTT handler
def create_mqtt_handler(config: Optional[Dict[str, Any]] = None) -> MqttHandler:
    """
    Create an MQTT handler with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured MqttHandler instance
    """
    default_config = {
        'broker': 'localhost',
        'port': 1883,
        'topics': {
            'flir': 'sensors/flir/+/data',
            'scd41': 'sensors/scd41/+/data'
        }
    }
    
    if config:
        default_config.update(config)
    
    return MqttHandler(default_config)