# MQTT Implementation Summary for FLIR+SCD41 Sensors

## Overview

This document provides a comprehensive summary of the MQTT implementation for FLIR Lepton 3.5 thermal cameras and SCD41 CO₂ sensors in the Saafe Fire Detection System.

## Implementation Details

### 1. MQTT Handler Module

The MQTT functionality is implemented in `/src/hardware/mqtt_handler.py` with the following key components:

- **MqttHandler Class**: Main class that handles MQTT communication
- **create_mqtt_handler Function**: Convenience function for creating MQTT handlers
- **Mock Implementation**: Fallback implementation when the paho-mqtt library is not available

### 2. Key Features

- **Topic Subscription**: Subscribes to `sensors/flir/+/data` and `sensors/scd41/+/data` topics
- **Data Processing**: Parses incoming JSON messages and extracts sensor data
- **Sensor Identification**: Extracts sensor IDs from MQTT topic structure
- **Data Storage**: Maintains latest data from all sensors in memory
- **Callback System**: Provides callbacks for data reception, connection status, and errors
- **Thread Safety**: Uses locks to ensure thread-safe data access

### 3. Data Format

The system expects MQTT messages in the following JSON format:

#### FLIR Lepton 3.5 Data
```json
{
  "t_mean": 25.5,
  "t_std": 2.3,
  "t_max": 35.2,
  "t_p95": 32.1,
  "t_hot_area_pct": 5.2,
  "t_hot_largest_blob_pct": 2.8,
  "t_grad_mean": 1.2,
  "t_grad_std": 0.8,
  "t_diff_mean": 0.3,
  "t_diff_std": 0.1,
  "flow_mag_mean": 0.5,
  "flow_mag_std": 0.2,
  "tproxy_val": 35.2,
  "tproxy_delta": 2.1,
  "tproxy_vel": 1.0,
  "timestamp": "2025-08-28T10:30:45.123456",
  "device_type": "flir_lepton_3_5"
}
```

#### SCD41 CO₂ Data
```json
{
  "gas_val": 450.0,
  "gas_delta": 10.0,
  "gas_vel": 10.0,
  "co2_concentration": 450.0,
  "sensor_temp": 23.5,
  "sensor_humidity": 48.2,
  "timestamp": "2025-08-28T10:30:45.123456",
  "device_type": "sensirion_scd41"
}
```

### 4. Integration with Sensor Manager

The MQTT handler is integrated with the SensorManager in `/src/hardware/sensor_manager.py`:

- **Initialization**: MQTT client is created and configured during sensor initialization
- **Connection Management**: Automatic connection to MQTT broker with error handling
- **Data Callback**: Sensor data received via MQTT is added to the data buffer
- **Health Monitoring**: Connection status and errors are tracked

### 5. Configuration

MQTT can be enabled in the sensor manager configuration:

```python
config = {
    'mqtt': {
        'enabled': True,
        'broker': 'localhost',  # MQTT broker address
        'port': 1883,           # MQTT broker port
        'topics': {
            'flir': 'sensors/flir/+/data',
            'scd41': 'sensors/scd41/+/data'
        },
        'username': 'optional_username',
        'password': 'optional_password'
    }
}
```

## Usage Examples

### 1. Basic MQTT Setup

```python
from src.hardware.mqtt_handler import create_mqtt_handler

# Configuration
config = {
    'broker': 'mqtt.saafe.ai',
    'port': 1883,
    'topics': {
        'flir': 'sensors/flir/+/data',
        'scd41': 'sensors/scd41/+/data'
    }
}

# Create handler
mqtt_handler = create_mqtt_handler(config)

# Set callbacks
mqtt_handler.set_data_callback(on_data_received)
mqtt_handler.set_connection_callback(on_connection_change)
mqtt_handler.set_error_callback(on_error)

# Connect
if mqtt_handler.connect():
    print("Connected to MQTT broker")
```

### 2. Integration with Sensor Manager

```python
from src.hardware.sensor_manager import SensorManager, SensorMode

# Configuration with MQTT enabled
config = {
    'mode': SensorMode.REAL,
    'mqtt': {
        'enabled': True,
        'broker': 'mqtt.saafe.ai',
        'port': 1883
    }
}

# Create and initialize sensor manager
sensor_manager = SensorManager(config)
sensor_manager.initialize_sensors()
sensor_manager.start_data_collection()
```

## Testing

Several test scripts are provided to verify the MQTT implementation:

1. `test_mqtt_simple.py` - Basic import and creation test
2. `test_mqtt_comprehensive.py` - Comprehensive functionality test
3. `test_mqtt_publisher.py` - MQTT publisher simulation
4. `test_mqtt_integration.py` - Full integration test
5. `test_sensor_manager_mqtt.py` - Sensor manager integration test
6. `demo_mqtt_data_ingestion.py` - Complete end-to-end demonstration

## Deployment Considerations

### 1. Network Requirements

- MQTT broker must be accessible from the deployment environment
- Firewall must allow connections on the MQTT port (typically 1883)
- Reliable network connectivity for continuous data streaming

### 2. Security

- Use TLS/SSL for encrypted MQTT connections in production
- Implement proper authentication with username/password
- Consider using client certificates for mutual authentication

### 3. Scalability

- The implementation supports multiple sensors per type
- Data is stored in memory with configurable buffer sizes
- Thread-safe design allows for concurrent access

### 4. Error Handling

- Automatic reconnection attempts on broker disconnection
- Graceful degradation when MQTT is unavailable
- Comprehensive error logging and callback mechanisms

## Troubleshooting

### Common Issues

1. **MQTT Library Not Found**: Install paho-mqtt: `pip install paho-mqtt`
2. **Connection Refused**: Check broker address, port, and network connectivity
3. **Authentication Failed**: Verify username and password
4. **No Data Received**: Check topic subscriptions and publisher configuration

### Debugging Tips

1. Enable debug logging to see detailed MQTT communication
2. Use the test scripts to verify basic functionality
3. Check broker logs for connection and subscription information
4. Verify that publishers are sending data in the expected format

## Future Enhancements

1. **Quality of Service (QoS)**: Implement different QoS levels for critical data
2. **Last Will and Testament**: Add LWT messages for graceful failure detection
3. **Retained Messages**: Support for retained messages for initial state
4. **TLS/SSL Support**: Enhanced security with encrypted connections
5. **Message Persistence**: Store messages locally during broker outages