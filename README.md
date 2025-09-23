# Synthetic Fire Prediction System

A comprehensive AI-powered fire detection platform that uses synthetic data generation to develop, train, and validate the complete system before hardware deployment.

## Overview

The Synthetic Fire Prediction System follows a modular, agent-based approach with clear separation between data generation, feature processing, model inference, and response coordination. The system operates in two phases:

1. **Synthetic Development Phase**: Using generated data for system development and validation
2. **Hardware Integration Phase**: Seamless transition to real sensors

## Features

- **Synthetic Data Generation**: Realistic thermal, gas, and environmental sensor data
- **Feature Engineering**: 18+ meaningful features from multi-sensor data
- **Advanced ML Models**: Temporal models, baseline models, and ensemble systems
- **Multi-Agent System**: Specialized agents for monitoring, analysis, response, and learning
- **Hardware Abstraction**: Seamless transition from synthetic to real hardware
- **Comprehensive Testing**: Automated testing with synthetic data validation
- **AWS Dashboard**: Real-time monitoring of sensor data and fire detection scores
- **Alerting System**: Multi-level alert hierarchy with hysteresis-based transitions

## Project Structure

```
synthetic_fire_system/
├── core/                   # Core interfaces and configuration
├── synthetic_data/         # Data generation components
├── feature_engineering/    # Feature extraction and fusion
├── models/                 # ML models and ensemble systems
├── agents/                 # Multi-agent system components
├── hardware/               # Hardware abstraction layer
├── system/                 # System management components
├── alerting/               # Alerting system components
└── __init__.py

tests/
├── unit/                  # Unit tests
├── integration/           # Integration tests
└── performance/           # Performance tests

config/                    # Configuration files
```

## Installation

```bash
# Clone the repository
git clone https://github.com/firesafety/synthetic-fire-prediction-system.git
cd synthetic-fire-prediction-system

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from synthetic_fire_system.core.config import config_manager
from synthetic_fire_system.system.manager import SystemManager

# Initialize system with default configuration
system = SystemManager(config_manager)

# Start the system
system.start()

# The system will begin generating synthetic data and training models
```

## AWS Dashboard

The system includes a real-time dashboard for monitoring sensor data and fire detection scores from your AWS deployment:

```bash
# Run the AWS dashboard
python run_aws_dashboard.py
```

Access the dashboard at http://localhost:8501 to view:
- Real-time sensor readings (temperature, PM2.5, CO₂, audio levels)
- Fire detection risk scores
- System component status (S3, Lambda, SageMaker)
- Historical data trends
- Alert status and notifications

See `AWS_DASHBOARD_README.md` for detailed documentation.

### Deploying to AWS

To deploy the dashboard to AWS, use our deployment scripts:

```bash
# For ECS Fargate deployment
./deploy_dashboard.sh

# For Elastic Beanstalk deployment (simpler option)
./deploy_dashboard_simple.sh
```

**Note on JSON/YAML Errors**: If you encounter JSON or YAML parsing errors during deployment, this is likely due to hidden macOS metadata files. Use our clean deployment package:
```bash
python3 create_clean_deployment.py
```

### Testing AWS Connections

To verify that your AWS credentials and connections are properly configured:

```bash
python test_aws_connection.py
```

This script will test connections to all required AWS services:
- S3 bucket access
- Lambda function access
- SageMaker endpoint access
- CloudWatch metrics access

## Configuration

The system uses a comprehensive configuration management system. Configuration files are located in the `config/` directory:

- `system.json`: Main system configuration
- `synthetic_data.json`: Synthetic data generation settings
- `model.json`: Machine learning model configuration
- `agent.json`: Multi-agent system settings

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=synthetic_fire_system

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

### Code Quality

```bash
# Format code
black synthetic_fire_system/

# Lint code
flake8 synthetic_fire_system/

# Type checking
mypy synthetic_fire_system/
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, SciPy, Pandas
- Scikit-learn, XGBoost
- OpenCV, Matplotlib
- FastAPI, Uvicorn
- MLflow, DVC

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## Documentation

Comprehensive documentation is available in the `docs/` directory and can be built using Sphinx:

```bash
cd docs/
make html
```

## Support

For support and questions, please open an issue on GitHub or contact the development team.