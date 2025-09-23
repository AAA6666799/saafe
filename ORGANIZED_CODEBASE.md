# Saafe Fire Detection System - Organized Codebase Structure

This document outlines the organized structure of the Saafe Fire Detection System codebase for handover to the company.

## 1. Project Overview

The Saafe Fire Detection System is an AI-powered fire detection and prevention platform that uses synthetic data generation to develop, train, and validate the complete system before hardware deployment. It features:

- Synthetic data generation for thermal, gas, and environmental sensors
- Multi-agent system architecture for monitoring, analysis, response, and learning
- Hardware abstraction layer for easy integration with real devices
- Advanced ML model training using transformer-based architectures
- Ensemble modeling combining temporal, baseline, and deep learning models

## 2. Organized Directory Structure

```
saafe-fire-detection/
├── README.md                           # Project overview and quick start guide
├── LICENSE                             # License information
├── CHANGELOG.md                        # Version history and changes
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package installation configuration
├── main.py                             # Main application entry point
├── start_saafe.sh                      # Startup script
│
├── docs/                              # Documentation
│   ├── architecture.md                # System architecture documentation
│   ├── deployment.md                  # Deployment guide
│   ├── api_reference.md               # API documentation
│   └── user_manual.md                 # User guide
│
├── src/                               # Source code (main application)
│   ├── core/                          # Core business logic
│   │   ├── data_models.py             # Data structures and models
│   │   ├── scenario_manager.py        # Scenario generation and management
│   │   ├── fire_detection_pipeline.py # Main detection pipeline
│   │   └── alert_engine.py            # Alert generation and management
│   │
│   ├── models/                        # Machine learning models
│   │   ├── transformer.py             # Spatio-temporal transformer model
│   │   ├── anti_hallucination.py      # False positive prevention
│   │   ├── model_manager.py           # Model loading and management
│   │   └── model_loader.py            # Model persistence
│   │
│   ├── agents/                        # Multi-agent system
│   │   ├── monitoring/                # System monitoring agents
│   │   ├── analysis/                  # Data analysis agents
│   │   ├── response/                  # Alert response agents
│   │   ├── learning/                  # Continuous learning agents
│   │   └── coordination/              # Agent coordination system
│   │
│   ├── hardware/                      # Hardware abstraction layer
│   │   ├── sensor_manager.py          # Sensor management
│   │   ├── real_sensors.py            # Real sensor implementations
│   │   └── base.py                    # Hardware interfaces
│   │
│   ├── services/                      # Supporting services
│   │   ├── notification_manager.py    # Alert notification system
│   │   ├── export_service.py          # Data export functionality
│   │   └── performance_monitor.py     # System performance tracking
│   │
│   ├── ui/                            # User interface components
│   │   ├── dashboard.py               # Main dashboard interface
│   │   ├── sensor_components.py       # Sensor data visualization
│   │   ├── ai_analysis_components.py  # AI analysis panels
│   │   └── alert_components.py        # Alert status displays
│   │
│   └── utils/                         # Utility functions
│       ├── error_handler.py           # Error management
│       └── fallback_manager.py        # System resilience
│
├── config/                            # Configuration files
│   ├── app_config.json                # Main application configuration
│   ├── system_config.json             # System-level settings
│   └── agent_config.json              # Agent-specific configurations
│
├── tests/                             # Test suite
│   ├── unit/                          # Unit tests
│   ├── integration/                   # Integration tests
│   └── performance/                   # Performance benchmarks
│
├── scripts/                           # Utility scripts
│   ├── deploy.sh                      # Deployment script
│   ├── train_models.py                # Model training script
│   └── run_tests.py                   # Test execution script
│
├── models/                            # Trained model artifacts
│   ├── transformer_model.pth          # Main transformer model
│   ├── anti_hallucination.pkl         # Anti-hallucination model
│   └── model_metadata.json            # Model version information
│
├── data/                              # Data files
│   ├── synthetic_datasets/            # Generated synthetic data
│   └── sample_data/                   # Example real-world data
│
└── deployment/                        # Deployment configurations
    ├── docker/                        # Docker configurations
    ├── kubernetes/                    # Kubernetes manifests
    ├── aws/                           # AWS deployment templates
    └── systemd/                       # Systemd service files
```

## 3. Key Components

### 3.1 Core System (`src/core/`)

The core module contains the main business logic of the Saafe system:

- **Data Models**: Standardized data structures for sensor readings, scenarios, and predictions
- **Scenario Manager**: Generates synthetic scenarios for training and testing
- **Fire Detection Pipeline**: Processes sensor data through the complete detection workflow
- **Alert Engine**: Manages alert generation and escalation

### 3.2 Machine Learning Models (`src/models/`)

The models module implements the AI components:

- **Spatio-Temporal Transformer**: Deep learning model that processes temporal and spatial sensor data
- **Anti-Hallucination Engine**: Prevents false positives during cooking scenarios
- **Model Manager**: Handles model loading, saving, and version management
- **Ensemble System**: Combines multiple models for improved accuracy

### 3.3 Multi-Agent System (`src/agents/`)

The agents module implements a multi-agent architecture:

- **Monitoring Agents**: Track system health and sensor data quality
- **Analysis Agents**: Process sensor data using ML models
- **Response Agents**: Generate alerts and coordinate responses
- **Learning Agents**: Track performance and recommend improvements
- **Coordination System**: Manages communication between agents

### 3.4 Hardware Abstraction (`src/hardware/`)

The hardware module enables seamless transition from synthetic to real sensors:

- **Sensor Manager**: Unified interface for sensor operations
- **Real Sensors**: Implementations for actual hardware devices
- **Interfaces**: Abstract classes defining sensor contracts

### 3.5 User Interface (`src/ui/`)

The UI module provides the dashboard interface:

- **Main Dashboard**: Central interface for monitoring and control
- **Sensor Components**: Real-time sensor data visualization
- **AI Analysis**: Model predictions and confidence scores
- **Alert Components**: Alert status and management

## 4. Deployment Options

The system supports multiple deployment scenarios:

1. **Local Development**: Using Docker Compose for development
2. **Cloud Deployment**: AWS ECS with Fargate for production
3. **Enterprise Deployment**: Kubernetes for large-scale installations
4. **IoT Deployment**: Lightweight version for edge devices

## 5. Testing and Quality Assurance

The codebase includes comprehensive testing:

- **Unit Tests**: Component-level testing
- **Integration Tests**: End-to-end workflow validation
- **Performance Tests**: Benchmarking and optimization
- **Security Tests**: Vulnerability scanning and mitigation

## 6. Documentation

Complete documentation is available in the `docs/` directory:

- **Architecture Guide**: Detailed system design
- **Deployment Manual**: Step-by-step deployment instructions
- **API Reference**: Technical API documentation
- **User Manual**: End-user operation guide