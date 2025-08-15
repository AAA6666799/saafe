# Saafe Fire Detection System - File Manifest

## Overview
This document provides a comprehensive inventory of all files in the Saafe Fire Detection System repository, with detailed descriptions of their purpose, functionality, and relationships.

---

## 📋 Root Level Files

### Core Application Files

#### `main.py`
**Purpose**: Primary application entry point and launcher  
**Type**: Python executable  
**Dependencies**: `saafe_mvp` package, `streamlit`  
**Description**: Handles application initialization, environment detection (PyInstaller vs source), and launches the Streamlit web interface. Includes error handling for missing dependencies and provides fallback demo mode for testing environments.

#### `app.py`
**Purpose**: Streamlit web application entry point  
**Type**: Python module  
**Dependencies**: `saafe_mvp.ui.dashboard`  
**Description**: Minimal wrapper that imports and executes the main dashboard function. Designed for direct Streamlit execution with `streamlit run app.py`.

#### `requirements.txt`
**Purpose**: Python package dependencies specification  
**Type**: pip requirements file  
**Description**: Comprehensive list of all Python packages required for the application, including ML libraries (PyTorch, scikit-learn), web framework (Streamlit), notification services (Twilio, SendGrid), and development tools.

### Configuration Files

#### `.gitignore`
**Purpose**: Git version control exclusion rules  
**Type**: Git configuration  
**Description**: Comprehensive exclusion patterns for Python artifacts, virtual environments, IDE files, model files, logs, temporary files, and system-specific files. Follows enterprise standards for Python projects.

#### `LICENSE`
**Purpose**: Software license agreement  
**Type**: Legal document  
**Description**: MIT License granting permissions for use, modification, and distribution while limiting liability and requiring attribution.

### Container Configuration

#### `Dockerfile`
**Purpose**: Docker container build instructions  
**Type**: Docker configuration  
**Description**: Multi-stage Docker build configuration optimized for production deployment with security hardening, non-root user execution, and minimal attack surface.

#### `docker-compose.yml`
**Purpose**: Multi-container application orchestration  
**Type**: Docker Compose configuration  
**Description**: Defines services for the main application, Redis cache, Prometheus monitoring, and Grafana dashboards with proper networking and volume management.

#### `docker-compose-codeartifact.yml`
**Purpose**: AWS CodeArtifact integration for container builds  
**Type**: Docker Compose configuration  
**Description**: Specialized configuration for building and deploying containers using AWS CodeArtifact for private package management.

#### `Dockerfile-codeartifact`
**Purpose**: CodeArtifact-optimized container build  
**Type**: Docker configuration  
**Description**: Docker build configuration specifically designed for AWS CodeArtifact environments with proper authentication and package resolution.

### Documentation Files

#### `README.md`
**Purpose**: Primary project documentation and overview  
**Type**: Markdown documentation  
**Description**: Comprehensive project introduction with architecture overview, quick start guide, feature descriptions, and links to detailed documentation.

#### `PROJECT_OVERVIEW.md`
**Purpose**: Executive-level project summary  
**Type**: Markdown documentation  
**Description**: High-level project overview including business context, architecture philosophy, technology stack, and strategic roadmap for stakeholders and decision-makers.

#### `ARCHITECTURE.md`
**Purpose**: Technical architecture documentation  
**Type**: Markdown documentation  
**Description**: Detailed technical architecture including system design, component interactions, data flow, security architecture, and scalability considerations.

#### `DEPLOYMENT_GUIDE.md`
**Purpose**: Production deployment procedures  
**Type**: Markdown documentation  
**Description**: Comprehensive deployment guide covering environment setup, container orchestration, cloud deployment, monitoring configuration, and operational procedures.

#### `SECURITY.md`
**Purpose**: Security framework documentation  
**Type**: Markdown documentation  
**Description**: Complete security framework including authentication, authorization, encryption, compliance standards, threat detection, and incident response procedures.

#### `FILE_MANIFEST.md`
**Purpose**: Complete file inventory and documentation  
**Type**: Markdown documentation  
**Description**: This document - comprehensive catalog of all repository files with descriptions, purposes, and relationships.

### Installation and Setup Files

#### `INSTALLATION_GUIDE.md`
**Purpose**: Installation instructions and system requirements  
**Type**: Markdown documentation  
**Description**: Step-by-step installation guide covering system requirements, dependency installation, configuration setup, and initial deployment procedures.

#### `start_saafe.sh`
**Purpose**: Application startup script  
**Type**: Shell script  
**Dependencies**: Python environment, Streamlit  
**Description**: Automated startup script that activates the virtual environment, validates dependencies, and launches the Streamlit application with proper configuration.

### AWS Integration Files

#### `AWS_Deployment_Guide.md`
**Purpose**: AWS-specific deployment documentation  
**Type**: Markdown documentation  
**Description**: Detailed guide for deploying Saafe on AWS infrastructure including service configuration, security groups, IAM roles, and best practices.

#### `setup_codeartifact.py`
**Purpose**: AWS CodeArtifact configuration automation  
**Type**: Python script  
**Dependencies**: `boto3`, AWS CLI  
**Description**: Automated setup script for AWS CodeArtifact repositories, including domain creation, repository configuration, and authentication setup.

#### `upload_to_aws.py`
**Purpose**: AWS deployment automation  
**Type**: Python script  
**Dependencies**: `boto3`, AWS services  
**Description**: Comprehensive deployment script for uploading application artifacts to AWS services including S3, CodeCommit, and CodeArtifact.

#### `cleanup_for_aws.py`
**Purpose**: AWS resource cleanup automation  
**Type**: Python script  
**Dependencies**: `boto3`  
**Description**: Automated cleanup script for removing AWS resources, temporary files, and deployment artifacts to maintain clean environments and control costs.

#### `codecommit_setup_instructions.md`
**Purpose**: AWS CodeCommit setup guide  
**Type**: Markdown documentation  
**Description**: Step-by-step instructions for setting up AWS CodeCommit repositories, configuring Git credentials, and establishing CI/CD pipelines.

### Workflow and Automation Files

#### `codeartifact_workflow.sh`
**Purpose**: CodeArtifact workflow automation  
**Type**: Shell script  
**Dependencies**: AWS CLI, Docker  
**Description**: Automated workflow for building, testing, and publishing packages to AWS CodeArtifact with proper authentication and versioning.

#### `requirements-codeartifact.txt`
**Purpose**: CodeArtifact-specific Python dependencies  
**Type**: pip requirements file  
**Description**: Modified requirements file configured for AWS CodeArtifact package resolution with private repository URLs and authentication.

#### `upload_codebase.sh`
**Purpose**: Codebase upload automation  
**Type**: Shell script  
**Dependencies**: Git, AWS CLI  
**Description**: Automated script for uploading the complete codebase to version control systems with proper branching, tagging, and metadata.

#### `upload_summary.md`
**Purpose**: Upload process documentation  
**Type**: Markdown documentation  
**Description**: Summary of upload procedures, version history, and deployment status for tracking and audit purposes.

### Utility and Helper Files

#### `rename_to_saafe.py`
**Purpose**: Project renaming utility  
**Type**: Python script  
**Description**: Utility script for renaming project references, file paths, and configuration values when migrating from development names to production "Saafe" branding.

#### `test_results.json`
**Purpose**: Test execution results  
**Type**: JSON data file  
**Description**: Structured test results including pass/fail status, coverage metrics, performance benchmarks, and security scan results for CI/CD pipeline integration.

### Archive and Backup Files

#### `saafe_codebase_20250815_211725.zip`
**Purpose**: Timestamped codebase archive  
**Type**: ZIP archive  
**Description**: Complete codebase snapshot with timestamp for backup, distribution, or deployment purposes. Contains all source code, configuration, and documentation.

---

## 📁 Directory Structure

### `/saafe_mvp/` - Core Application Package

#### Core Business Logic (`/saafe_mvp/core/`)

##### `__init__.py`
**Purpose**: Package initialization  
**Type**: Python package file  
**Description**: Initializes the core package and exposes key classes and functions for import by other modules.

##### `data_models.py`
**Purpose**: Data structure definitions  
**Type**: Python module  
**Dependencies**: `pydantic`, `datetime`  
**Description**: Defines data models for sensor readings, alerts, user configurations, and system states using Pydantic for validation and serialization.

##### `data_stream.py`
**Purpose**: Real-time data stream processing  
**Type**: Python module  
**Dependencies**: `asyncio`, `queue`, `threading`  
**Description**: Handles real-time sensor data ingestion, buffering, validation, and distribution to processing components with fault tolerance and backpressure handling.

##### `fire_detection_pipeline.py`
**Purpose**: Main fire detection processing engine  
**Type**: Python module  
**Dependencies**: `saafe_mvp.models`, `saafe_mvp.core.data_stream`  
**Description**: Core fire detection logic combining sensor data analysis, AI model inference, and risk assessment to generate fire alerts with configurable thresholds.

##### `alert_engine.py`
**Purpose**: Intelligent alerting system  
**Type**: Python module  
**Dependencies**: `saafe_mvp.services.notification_manager`  
**Description**: Manages alert generation, prioritization, escalation, and delivery across multiple channels with deduplication and rate limiting.

##### `scenario_manager.py`
**Purpose**: Risk scenario management  
**Type**: Python module  
**Dependencies**: `saafe_mvp.core.data_models`  
**Description**: Manages different fire risk scenarios (normal, cooking, fire) with dynamic threshold adjustment and context-aware detection logic.

##### `scenario_generator.py`
**Purpose**: Scenario simulation and generation  
**Type**: Python module  
**Dependencies**: `random`, `numpy`  
**Description**: Generates realistic sensor data scenarios for testing, demonstration, and training purposes with configurable parameters and realistic variations.

##### `normal_scenario.py`
**Purpose**: Normal operating condition simulation  
**Type**: Python module  
**Dependencies**: `saafe_mvp.core.scenario_generator`  
**Description**: Implements normal environmental condition simulation with typical temperature, humidity, and air quality variations.

##### `cooking_scenario.py`
**Purpose**: Cooking activity simulation  
**Type**: Python module  
**Dependencies**: `saafe_mvp.core.scenario_generator`  
**Description**: Simulates cooking-related sensor changes that should not trigger fire alerts, helping train the anti-hallucination system.

##### `fire_scenario.py`
**Purpose**: Fire event simulation  
**Type**: Python module  
**Dependencies**: `saafe_mvp.core.scenario_generator`  
**Description**: Generates realistic fire progression scenarios with escalating sensor readings for testing detection accuracy and response times.

#### AI/ML Models (`/saafe_mvp/models/`)

##### `__init__.py`
**Purpose**: Models package initialization  
**Type**: Python package file  
**Description**: Initializes the models package and provides convenient imports for model classes and utilities.

##### `model_manager.py`
**Purpose**: AI model lifecycle management  
**Type**: Python module  
**Dependencies**: `torch`, `joblib`, `json`  
**Description**: Manages model loading, versioning, performance monitoring, and hot-swapping with fallback mechanisms and health checks.

##### `model_loader.py`
**Purpose**: Model loading utilities  
**Type**: Python module  
**Dependencies**: `torch`, `pickle`  
**Description**: Handles secure model loading from various formats (PyTorch, pickle, ONNX) with validation, integrity checks, and error handling.

##### `transformer.py`
**Purpose**: Deep learning transformer model  
**Type**: Python module  
**Dependencies**: `torch`, `torch.nn`  
**Description**: Custom transformer architecture optimized for time-series sensor data analysis with attention mechanisms for fire pattern recognition.

##### `anti_hallucination.py`
**Purpose**: False positive prevention system  
**Type**: Python module  
**Dependencies**: `scikit-learn`, `numpy`  
**Description**: Proprietary anti-hallucination system using ensemble methods and confidence scoring to reduce false fire alerts.

#### User Interface (`/saafe_mvp/ui/`)

##### `__init__.py`
**Purpose**: UI package initialization  
**Type**: Python package file  
**Description**: Initializes the UI package and sets up Streamlit configuration and theming.

##### `dashboard.py`
**Purpose**: Main dashboard orchestration  
**Type**: Python module  
**Dependencies**: `streamlit`, all UI components  
**Description**: Main dashboard that orchestrates all UI components, handles navigation, state management, and real-time updates.

##### `sensor_components.py`
**Purpose**: Sensor data visualization  
**Type**: Python module  
**Dependencies**: `streamlit`, `plotly`, `pandas`  
**Description**: Real-time sensor data visualization components including gauges, time series charts, and status indicators.

##### `alert_components.py`
**Purpose**: Alert management interface  
**Type**: Python module  
**Dependencies**: `streamlit`, `saafe_mvp.core.alert_engine`  
**Description**: Alert display, acknowledgment, and management interface with filtering, sorting, and bulk operations.

##### `ai_analysis_components.py`
**Purpose**: AI model insights display  
**Type**: Python module  
**Dependencies**: `streamlit`, `plotly`  
**Description**: Displays AI model predictions, confidence scores, feature importance, and model performance metrics.

##### `settings_components.py`
**Purpose**: Configuration interface  
**Type**: Python module  
**Dependencies**: `streamlit`, `json`  
**Description**: User interface for system configuration including thresholds, notification settings, and user preferences.

##### `settings_page.py`
**Purpose**: Settings page implementation  
**Type**: Python module  
**Dependencies**: `streamlit`, `saafe_mvp.ui.settings_components`  
**Description**: Dedicated settings page with tabbed interface for different configuration categories.

##### `export_components.py`
**Purpose**: Data export functionality  
**Type**: Python module  
**Dependencies**: `streamlit`, `saafe_mvp.services.export_service`  
**Description**: User interface for data export including format selection, date ranges, and download management.

#### Services (`/saafe_mvp/services/`)

##### `__init__.py`
**Purpose**: Services package initialization  
**Type**: Python package file  
**Description**: Initializes the services package and provides service registry and dependency injection.

##### `notification_manager.py`
**Purpose**: Multi-channel notification orchestration  
**Type**: Python module  
**Dependencies**: All notification services  
**Description**: Orchestrates notifications across multiple channels (SMS, email, push) with delivery confirmation, retry logic, and failure handling.

##### `email_service.py`
**Purpose**: Email notification service  
**Type**: Python module  
**Dependencies**: `sendgrid`, `email`  
**Description**: SendGrid integration for email notifications with template management, attachment support, and delivery tracking.

##### `sms_service.py`
**Purpose**: SMS notification service  
**Type**: Python module  
**Dependencies**: `twilio`  
**Description**: Twilio integration for SMS notifications with international support, delivery confirmation, and cost optimization.

##### `push_notification_service.py`
**Purpose**: Push notification service  
**Type**: Python module  
**Dependencies**: `firebase-admin` (future)  
**Description**: Push notification service for mobile applications with device registration, targeting, and analytics.

##### `export_service.py`
**Purpose**: Report generation and export  
**Type**: Python module  
**Dependencies**: `reportlab`, `pandas`, `matplotlib`  
**Description**: Generates reports in multiple formats (PDF, CSV, Excel) with charts, tables, and customizable templates.

##### `performance_monitor.py`
**Purpose**: System performance monitoring  
**Type**: Python module  
**Dependencies**: `psutil`, `GPUtil`, `prometheus_client`  
**Description**: Monitors system performance metrics including CPU, memory, GPU usage, and application-specific metrics.

##### `session_manager.py`
**Purpose**: User session management  
**Type**: Python module  
**Dependencies**: `streamlit`  
**Description**: Manages user sessions, state persistence, and multi-user support with security controls and timeout handling.

#### Utilities (`/saafe_mvp/utils/`)

##### `__init__.py`
**Purpose**: Utils package initialization  
**Type**: Python package file  
**Description**: Initializes the utilities package and provides common utility functions.

##### `error_handler.py`
**Purpose**: Comprehensive error management  
**Type**: Python module  
**Dependencies**: `logging`, `traceback`  
**Description**: Centralized error handling with logging, user-friendly error messages, recovery procedures, and error reporting.

##### `fallback_manager.py`
**Purpose**: System resilience and fallback mechanisms  
**Type**: Python module  
**Dependencies**: `saafe_mvp.core`  
**Description**: Implements fallback mechanisms for service failures, degraded mode operations, and graceful degradation strategies.

### `/config/` - Configuration Management

#### `app_config.json`
**Purpose**: Application configuration  
**Type**: JSON configuration file  
**Description**: Main application configuration including detection thresholds, service endpoints, feature flags, and system parameters.

#### `user_config.json`
**Purpose**: User preferences and settings  
**Type**: JSON configuration file  
**Description**: User-specific configuration including notification preferences, dashboard layout, alert settings, and personalization options.

### `/models/` - Trained AI Models

#### `transformer_model.pth`
**Purpose**: PyTorch transformer model  
**Type**: Binary model file  
**Size**: ~50MB  
**Description**: Trained transformer model for fire detection with optimized weights for production inference.

#### `anti_hallucination.pkl`
**Purpose**: Anti-hallucination classifier  
**Type**: Pickle model file  
**Size**: ~5MB  
**Description**: Scikit-learn model for false positive reduction trained on historical alert data.

#### `model_metadata.json`
**Purpose**: Model versioning and metadata  
**Type**: JSON metadata file  
**Description**: Model version information, training metrics, performance benchmarks, and deployment history.

### `/docs/` - Documentation

#### `USER_MANUAL.md`
**Purpose**: End-user documentation  
**Type**: Markdown documentation  
**Description**: Comprehensive user manual covering system operation, dashboard usage, alert management, and troubleshooting.

#### `TECHNICAL_DOCUMENTATION.md`
**Purpose**: Technical specifications  
**Type**: Markdown documentation  
**Description**: Detailed technical documentation including API specifications, data models, algorithms, and integration guides.

#### `TROUBLESHOOTING_GUIDE.md`
**Purpose**: Issue resolution guide  
**Type**: Markdown documentation  
**Description**: Common issues, diagnostic procedures, resolution steps, and escalation procedures for technical support.

#### `DEMO_SCRIPT.md`
**Purpose**: Demonstration procedures  
**Type**: Markdown documentation  
**Description**: Step-by-step demonstration script for showcasing system capabilities, features, and use cases.

### `/saafe_for_codecommit/` - CodeCommit Integration

This directory contains a complete copy of the application configured specifically for AWS CodeCommit deployment with modified paths, configurations, and deployment scripts.

### `/saafe_env/` - Virtual Environment

Python virtual environment containing all installed dependencies and packages. This directory is typically excluded from version control but may be included for deployment purposes.

---

## 🔗 File Relationships and Dependencies

### Core Application Flow
```
main.py → app.py → saafe_mvp.ui.dashboard → [All UI Components]
                                        ↓
                              saafe_mvp.core.fire_detection_pipeline
                                        ↓
                              saafe_mvp.models.* + saafe_mvp.services.*
```

### Configuration Dependencies
```
app_config.json ← [All Core Components]
user_config.json ← [UI Components]
model_metadata.json ← [Model Components]
```

### Service Integration
```
notification_manager.py → email_service.py
                       → sms_service.py
                       → push_notification_service.py
```

### Model Pipeline
```
model_loader.py → transformer.py + anti_hallucination.py
                ↓
model_manager.py → fire_detection_pipeline.py
```

---

## 📊 File Statistics

### Code Metrics
- **Total Files**: 150+
- **Python Files**: 45+
- **Configuration Files**: 15+
- **Documentation Files**: 20+
- **Script Files**: 10+

### Size Distribution
- **Large Files (>1MB)**: Model files, archives
- **Medium Files (100KB-1MB)**: Documentation, complex modules
- **Small Files (<100KB)**: Most Python modules, configuration files

### Maintenance Priority
- **Critical**: Core detection pipeline, model files, security configurations
- **High**: UI components, service integrations, deployment scripts
- **Medium**: Documentation, utility functions, test files
- **Low**: Archive files, temporary files, development artifacts

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Classification**: Technical Documentation  
**Owner**: Engineering Team