# FLIR+SCD41 Fire Detection System User Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Hardware Configuration](#hardware-configuration)
4. [System Operation](#system-operation)
5. [Monitoring and Alerts](#monitoring-and-alerts)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance](#maintenance)

## System Overview

The FLIR+SCD41 Fire Detection System combines thermal imaging from the FLIR Lepton 3.5 camera with CO₂ sensing from the Sensirion SCD41 sensor to provide advanced fire detection capabilities. The system features:

- **Enhanced Feature Engineering**: 10 advanced feature extraction techniques
- **Advanced Fusion Model**: Attention-based sensor integration
- **Dynamic Weighting**: Adaptive ensemble weights based on conditions
- **Temporal Modeling**: LSTM and Transformer-based sequence analysis
- **Active Learning**: Continuous improvement through feedback
- **Edge Case Optimization**: Robust handling of challenging scenarios

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Required Python packages (see `requirements.txt`)
- FLIR Lepton 3.5 thermal camera
- Sensirion SCD41 CO₂ sensor
- MQTT broker (for sensor communication)

### Installation Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd synthetic-fire-prediction-system
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Configuration Files

The system uses several configuration files located in the `config/` directory:

- `base_config.yaml`: Main system configuration
- `sensors.yaml`: Sensor-specific settings
- `models.yaml`: Model configuration parameters
- `alerts.yaml`: Alert threshold settings

## Hardware Configuration

### FLIR Lepton 3.5 Setup

1. **Connect the camera:**
   - Connect FLIR Lepton 3.5 to your system via the appropriate interface (SPI, USB adapter, etc.)
   - Ensure proper power supply (3.3V)

2. **Configure thermal sensor settings:**
```yaml
# config/sensors.yaml
flir_lepton35:
  enabled: true
  interface: "spi"  # or "usb"
  calibration_file: "calibration/flir_calib.json"
  update_rate: 9  # Hz
  features:
    - t_mean
    - t_std
    - t_max
    - t_p95
    - t_hot_area_pct
    - t_hot_largest_blob_pct
    - t_grad_mean
    - t_grad_std
    - t_diff_mean
    - t_diff_std
    - flow_mag_mean
    - flow_mag_std
    - tproxy_val
    - tproxy_delta
    - tproxy_vel
```

### SCD41 CO₂ Sensor Setup

1. **Connect the sensor:**
   - Connect SCD41 via I2C interface
   - Ensure proper power supply (3.3V)

2. **Configure gas sensor settings:**
```yaml
# config/sensors.yaml
scd41_co2:
  enabled: true
  interface: "i2c"
  i2c_address: "0x62"
  update_rate: 1  # Hz (SCD41 limitation)
  features:
    - gas_val
    - gas_delta
    - gas_vel
```

### MQTT Configuration

The system communicates with sensors via MQTT:

```yaml
# config/base_config.yaml
mqtt:
  broker: "localhost"
  port: 1883
  username: "fire_sensor"
  password: "sensor_password"
  topics:
    flir: "sensors/flir/lepton35"
    scd41: "sensors/gas/scd41"
```

## System Operation

### Starting the System

1. **Start MQTT broker:**
```bash
# If using Mosquitto
mosquitto -c /etc/mosquitto/mosquitto.conf
```

2. **Start the fire detection system:**
```bash
python main.py --config config/base_config.yaml
```

### System Modes

The system can operate in different modes:

```bash
# Normal operation mode
python main.py --mode normal

# Testing mode with synthetic data
python main.py --mode test

# Calibration mode
python main.py --mode calibrate

# Benchmark mode
python main.py --mode benchmark
```

### Command Line Options

```bash
usage: main.py [-h] [--config CONFIG] [--mode MODE] [--verbose] [--log-file LOG_FILE]

FLIR+SCD41 Fire Detection System

optional arguments:
  -h, --help           show this help message and exit
  --config CONFIG      Path to configuration file
  --mode MODE          Operation mode (normal, test, calibrate, benchmark)
  --verbose            Enable verbose logging
  --log-file LOG_FILE  Path to log file
```

## Monitoring and Alerts

### Dashboard Access

The system provides a web-based dashboard for monitoring:

1. **Access the dashboard:**
   - Open browser to `http://localhost:8080`
   - Login with credentials from configuration

2. **Dashboard features:**
   - Real-time sensor data visualization
   - Fire detection alerts
   - System performance metrics
   - Historical data analysis

### Alert Types

The system generates different types of alerts:

1. **Fire Detection Alert**
   - Triggered when fire is detected with high confidence
   - Severity levels: Low, Medium, High, Critical
   - Automatic notification to configured endpoints

2. **System Health Alert**
   - Triggered for hardware or software issues
   - Examples: Sensor failure, connectivity issues, low resources

3. **Performance Alert**
   - Triggered when system performance degrades
   - Examples: High false positive rate, slow processing

### Alert Configuration

Alert thresholds and notifications are configured in `config/alerts.yaml`:

```yaml
# Fire detection thresholds
fire_detection:
  confidence_threshold: 0.8
  critical_threshold: 0.95
  warning_threshold: 0.7

# System health thresholds
system_health:
  cpu_usage_threshold: 80
  memory_usage_threshold: 85
  disk_usage_threshold: 90

# Notification settings
notifications:
  email:
    enabled: true
    smtp_server: "smtp.example.com"
    recipients:
      - "admin@example.com"
  sms:
    enabled: false
    provider: "twilio"
  webhook:
    enabled: true
    url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
```

## Performance Tuning

### Model Performance Monitoring

The system continuously monitors model performance:

1. **AUC Score Tracking:**
   - Target: > 0.90
   - Alert threshold: < 0.85

2. **False Positive Rate:**
   - Target: < 10%
   - Alert threshold: > 15%

3. **Detection Time:**
   - Target: < 30 seconds
   - Alert threshold: > 45 seconds

### Dynamic Weighting Adjustment

The ensemble weights are automatically adjusted based on performance:

```python
# View current weights
python scripts/view_weights.py

# Manually adjust weights
python scripts/adjust_weights.py --thermal-weight 0.6 --gas-weight 0.4
```

### Feature Selection Optimization

The system uses dynamic feature selection based on input patterns:

```python
# View selected features
python scripts/view_features.py --timestamp "2023-01-01T12:00:00"

# Force feature reselection
python scripts/optimize_features.py --force
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Sensor Connection Issues

**Problem:** Sensors not detected or reporting errors

**Solution:**
```bash
# Check sensor connectivity
python scripts/diagnose_sensors.py

# Restart sensor services
sudo systemctl restart flir-sensor
sudo systemctl restart scd41-sensor
```

#### 2. High False Positive Rate

**Problem:** Too many false alarms

**Solution:**
```bash
# Check false positive logs
tail -f logs/false_positives.log

# Adjust discrimination thresholds
python scripts/adjust_discrimination.py --threshold 0.85
```

#### 3. Slow Processing Performance

**Problem:** System processing slower than expected

**Solution:**
```bash
# Check system resources
python scripts/monitor_resources.py

# Optimize model inference
python scripts/optimize_inference.py --quantize
```

### Log Analysis

System logs are located in the `logs/` directory:

```bash
# View main application logs
tail -f logs/application.log

# View error logs
tail -f logs/error.log

# View performance logs
tail -f logs/performance.log
```

### Diagnostic Tools

Several diagnostic tools are available:

```bash
# System health check
python scripts/health_check.py

# Sensor diagnostics
python scripts/diagnose_sensors.py

# Model performance analysis
python scripts/analyze_performance.py

# Network connectivity test
python scripts/test_connectivity.py
```

## Maintenance

### Regular Maintenance Tasks

1. **Daily:**
   - Check system logs for errors
   - Verify sensor connectivity
   - Monitor alert rates

2. **Weekly:**
   - Review performance metrics
   - Update model with new data
   - Clean sensor lenses

3. **Monthly:**
   - Calibrate sensors
   - Review false positive patterns
   - Update system software

### Model Updates

The system supports automatic model updates:

```bash
# Check for model updates
python scripts/check_updates.py

# Apply model updates
python scripts/update_models.py

# Rollback to previous version
python scripts/rollback_models.py --version v1.2.3
```

### Backup and Recovery

Regular backups are essential:

```bash
# Create system backup
python scripts/backup_system.py --output backup_$(date +%Y%m%d).tar.gz

# Restore from backup
python scripts/restore_system.py --input backup_20230101.tar.gz
```

### Software Updates

Keep the system updated with the latest improvements:

```bash
# Check for software updates
python scripts/check_software_updates.py

# Apply updates
git pull
pip install -r requirements.txt
```

## Advanced Configuration

### Custom Feature Engineering

Add custom feature extraction modules:

```python
# Create new feature extractor in src/feature_engineering/extractors/custom/
class CustomFeatureExtractor:
    def __init__(self, config):
        self.config = config
    
    def extract_features(self, data):
        # Implement custom feature extraction
        return features
```

### Alert Customization

Customize alert handling:

```python
# Create custom alert handler in src/alerts/custom/
class CustomAlertHandler:
    def handle_alert(self, alert_data):
        # Implement custom alert handling
        pass
```

### Integration with External Systems

The system can integrate with external fire suppression systems:

```yaml
# config/integration.yaml
fire_suppression:
  enabled: true
  system: "autonomous"
  activation_threshold: 0.95
  safety_delay: 30  # seconds
```

## Support and Contact

For technical support, contact:
- **Email:** support@saafe.com
- **Phone:** +1-800-FIRE-AID
- **Website:** https://saafe.com/support

## Version Information

- **System Version:** 2.1.0
- **Model Version:** FLIR-SCD41-v2.1
- **Last Updated:** 2023-01-01

---

*This user guide was last updated on {datetime.now().strftime('%Y-%m-%d')}*
*For the latest documentation, visit: https://saafe.com/docs*