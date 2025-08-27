# Synthetic Fire Prediction System

A comprehensive system for synthetic data generation, feature engineering, model training, and agent-based fire prediction aligned with AWS architecture.

## Project Overview

The Synthetic Fire Prediction System is designed to generate realistic synthetic data for fire detection scenarios, extract meaningful features, train machine learning models, and implement an agent-based system for monitoring, analysis, response, and learning. The system is built with AWS integration in mind, allowing for scalable deployment and operation.

## Directory Structure

```
synthetic-fire-prediction-system/
├── src/
│   ├── data_generation/        # Synthetic data generation components
│   │   ├── thermal/            # Thermal image generation
│   │   ├── gas/                # Gas concentration generation
│   │   ├── environmental/      # Environmental data generation
│   │   └── scenarios/          # Scenario generation
│   ├── feature_engineering/    # Feature extraction components
│   │   ├── extractors/         # Feature extractors
│   │   │   ├── thermal/        # Thermal feature extraction
│   │   │   ├── gas/            # Gas feature extraction
│   │   │   └── environmental/  # Environmental feature extraction
│   │   └── fusion/             # Feature fusion
│   ├── models/                 # Machine learning models
│   │   ├── baseline/           # Baseline models (Random Forest, XGBoost)
│   │   ├── temporal/           # Temporal models (LSTM, GRU)
│   │   └── ensemble/           # Model ensemble
│   ├── agents/                 # Agent system components
│   │   ├── monitoring/         # Monitoring agent
│   │   ├── analysis/           # Analysis agent
│   │   ├── response/           # Response agent
│   │   └── learning/           # Learning agent
│   ├── aws/                    # AWS service integration
│   │   ├── s3/                 # S3 integration
│   │   ├── sagemaker/          # SageMaker integration
│   │   ├── lambda/             # Lambda integration
│   │   └── cloudwatch/         # CloudWatch integration
│   ├── config/                 # Configuration management
│   │   ├── environments/       # Environment-specific configurations
│   │   └── secrets/            # Secret management
│   ├── utils/                  # Utility functions
│   │   ├── logging/            # Logging utilities
│   │   └── validation/         # Validation utilities
│   └── hardware/               # Hardware abstraction layer
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── system/                 # System tests
├── notebooks/                  # Jupyter notebooks for analysis and visualization
├── docs/                       # Documentation
└── scripts/                    # Utility scripts
```

## Core Components

### Data Generation

- **Thermal Data Generation**: Creates synthetic thermal images with configurable hotspots and realistic noise patterns.
- **Gas Data Generation**: Simulates gas concentration readings for multiple gas types with realistic sensor characteristics.
- **Environmental Data Generation**: Generates temperature, humidity, pressure, and VOC data with daily and seasonal variations.
- **Scenario Generation**: Creates complete fire scenarios combining all sensor types with temporal evolution.

### Feature Engineering

- **Thermal Feature Extraction**: Extracts temperature statistics, hotspot detection, and motion patterns from thermal images.
- **Gas Feature Extraction**: Processes gas concentration readings, calculates slopes, and detects anomalies.
- **Environmental Feature Extraction**: Extracts environmental context and calculates derived metrics like dew point.
- **Feature Fusion**: Combines features from multiple sensors to create comprehensive risk assessments.

### Models

- **Baseline Models**: Traditional machine learning models like Random Forest and XGBoost.
- **Temporal Models**: Deep learning models like LSTM and GRU for temporal pattern recognition.
- **Model Ensemble**: Combines multiple models for improved prediction accuracy and robustness.

### Agent System

- **Monitoring Agent**: Continuously monitors data streams for anomalies and sensor health.
- **Analysis Agent**: Performs in-depth analysis of detected patterns and calculates confidence levels.
- **Response Agent**: Determines appropriate response levels and generates alerts and recommendations.
- **Learning Agent**: Tracks system performance and recommends improvements based on outcomes.

### AWS Integration

- **S3 Integration**: Manages data storage and retrieval from Amazon S3.
- **SageMaker Integration**: Handles model training and deployment using Amazon SageMaker.
- **Lambda Integration**: Implements serverless functions for event-driven processing.
- **CloudWatch Integration**: Manages monitoring, logging, and alerting through Amazon CloudWatch.

### Configuration Management

- **Environment-Specific Configurations**: Separate configurations for development, testing, and production.
- **Secret Management**: Secure handling of sensitive information like API keys and credentials.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- AWS account with appropriate permissions (for AWS integration)
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/synthetic-fire-prediction-system.git
   cd synthetic-fire-prediction-system
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure the system:
   - Copy `src/config/environments/base_config.yaml` to create custom configurations
   - Set up AWS credentials if using AWS integration

### Running the System

The system can be run in different modes:

- **Run Mode**: Normal operation mode
  ```
  python main.py --mode run
  ```

- **Test Mode**: Run system tests
  ```
  python main.py --mode test
  ```

- **Generate Mode**: Generate synthetic data
  ```
  python main.py --mode generate
  ```

- **Train Mode**: Train machine learning models
  ```
  python main.py --mode train
  ```

You can specify the environment and configuration path:
```
python main.py --mode run --env dev --config path/to/config
```

## Configuration

The system uses YAML configuration files located in `src/config/environments/`:

- `base_config.yaml`: Base configuration with default values
- `dev_config.yaml`: Development environment configuration
- `prod_config.yaml`: Production environment configuration

Environment-specific configurations override values from the base configuration.

## AWS Integration

The system is designed to integrate with AWS services:

- **Amazon S3**: Used for data storage and model artifacts
- **Amazon SageMaker**: Used for model training and deployment
- **AWS Lambda**: Used for event-driven processing and alerts
- **Amazon CloudWatch**: Used for monitoring, logging, and alerting

AWS credentials and region settings are configured in the configuration files.

## Testing

The system includes a comprehensive testing framework:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **System Tests**: Test the entire system end-to-end

Run tests using:
```
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.