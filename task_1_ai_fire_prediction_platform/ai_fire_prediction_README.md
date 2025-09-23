# AI Fire Prediction Platform

## Overview

The AI Fire Prediction Platform is an advanced AI-powered platform designed to detect and predict fire hazards using multi-sensor data fusion. This system processes data from thermal cameras, gas sensors, and environmental monitors to provide early fire detection capabilities.

## Features

### Core Components

1. **Multi-Sensor Data Processing**
   - Thermal camera data analysis (384x288 resolution)
   - Gas sensor readings (CO, CO2, smoke particles)
   - Environmental monitoring (temperature, humidity, pressure)

2. **Advanced AI Models**
   - Ensemble learning with multiple model types
   - Real-time prediction capabilities
   - Confidence scoring and uncertainty quantification

3. **Feature Engineering**
   - Thermal signature extraction
   - Gas concentration analysis
   - Environmental pattern recognition
   - Multi-modal data fusion

4. **Hardware Abstraction**
   - S3 storage interface for real sensor data
   - Modular hardware integration framework
   - Scalable sensor deployment architecture

5. **Alerting System** 🔴
   - Hysteresis-based alert level management
   - Multi-level alert hierarchy (Normal → Mild → Elevated → Critical)
   - Notification system (Email, SMS, Webhook)
   - Alert history and statistics tracking

### System Architecture

```
ai_fire_prediction_platform/
├── alerting/           # Alert generation and notification system
├── agents/             # Autonomous system agents
├── core/               # Core system components and configuration
├── feature_engineering/ # Data processing and feature extraction
├── hardware/           # Hardware abstraction and interfaces
├── models/             # AI models and ensemble system
├── synthetic_data/     # Synthetic data generation
└── system/             # System management and orchestration
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai_fire_prediction_platform

# Install dependencies
pip install -r synthetic_fire_requirements.txt

# Install the package
pip install -e .
```

## Usage

### Running the System

```bash
# Start the main system
python -m ai_fire_prediction_platform.main --run-time 300

# Run with custom configuration
python -m ai_fire_prediction_platform.main --config-dir ./config --log-level DEBUG
```

### Running the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run dashboard.py

# Or use the convenience script
python run_dashboard.py
```

### Training Models

```bash
# Train the system with synthetic data
python train_system.py

# Train with real S3 data
python train_with_s3_data.py
```

## Alerting System

The system includes a comprehensive alerting mechanism:

### Alert Levels
- **Normal** (🟢): System operating normally
- **Mild Anomaly** (🟡): Minor environmental variations detected
- **Elevated Risk** (🟠): Multiple sensors showing concerning patterns
- **Critical Fire Alert** (🔴): Immediate action required

### Hysteresis Logic
To prevent false alarms, the system implements hysteresis with configurable thresholds:
- Prevents rapid oscillation between alert levels
- Requires sustained conditions to change alert levels
- Configurable margins for sensitivity adjustment

### Notifications
- Email alerts for elevated and critical alerts
- SMS notifications for critical alerts
- Webhook integration for system-to-system communication

## Testing

```bash
# Run core system tests
python -m pytest test_system.py

# Run dashboard tests
python -m pytest test_dashboard.py

# Run alerting system tests
python test_alerting.py

# Run all tests
python -m pytest
```

## Configuration

The system can be configured through JSON configuration files:
- `config_system.json`: Main system configuration
- Custom threshold settings for alerting
- Model parameters and training configurations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue on the GitHub repository or contact the development team.