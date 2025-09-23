# FLIR+SCD41 Fire Detection System Architecture

This document outlines the architecture of the FLIR+SCD41 Fire Detection System, specifically designed for the FLIR Lepton 3.5 thermal camera and Sensirion SCD41 CO₂ sensor combination.

## System Overview

The FLIR+SCD41 Fire Detection System is a specialized fire detection solution that leverages the FLIR Lepton 3.5 thermal imaging camera and Sensirion SCD41 CO₂ sensor to provide accurate, real-time fire detection. The system processes 18 distinct features (15 thermal + 3 gas) to achieve high accuracy with minimal false positives.

## Component Architecture

### Hardware Components

#### FLIR Lepton 3.5 Thermal Camera
- **Resolution**: 160×120 pixels
- **Frame Rate**: 9 Hz
- **Temperature Range**: -10°C to +150°C
- **Features Processed**: 15 thermal features including:
  - `t_mean`, `t_std`: Average temperature and variation
  - `t_max`, `t_p95`: Hottest pixel and 95th percentile
  - `t_hot_area_pct`, `t_hot_largest_blob_pct`: Hot area percentages
  - `t_grad_mean`, `t_grad_std`: Gradient sharpness
  - `t_diff_mean`, `t_diff_std`: Frame-to-frame changes
  - `flow_mag_mean`, `flow_mag_std`: Optical flow (motion)
  - `tproxy_val`, `tproxy_delta`, `tproxy_vel`: Hotspot proxy values

#### Sensirion SCD41 CO₂ Sensor
- **Measurement Range**: 400 to 40,000 ppm
- **Sampling Rate**: Every 5 seconds
- **Features Processed**: 3 gas features including:
  - `gas_val`: Current CO₂ concentration (ppm)
  - `gas_delta`: Change from previous reading
  - `gas_vel`: Rate of change (same as delta)

### Data Processing Pipeline

#### 1. Feature Extraction
The system extracts 18 distinct features from the raw sensor data:

**Thermal Feature Extractor** (`FlirThermalExtractor`):
- Validates incoming thermal data format
- Computes derived features (ratios, activity metrics)
- Calculates fire indicators based on temperature thresholds
- Generates quality metrics for data validation

**Gas Feature Extractor** (`Scd41GasExtractor`):
- Validates incoming gas data format
- Computes derived features (concentration levels, changes)
- Calculates fire indicators based on CO₂ thresholds
- Generates quality metrics for data validation

#### 2. Feature Fusion
The feature fusion engine combines the 15 thermal features and 3 gas features into a single 18-feature vector for model processing.

#### 3. Machine Learning Models
The system implements multiple ML models trained specifically on the 18-feature input:

**Baseline Models**:
- Random Forest classifier with 200 estimators
- XGBoost classifier with optimized parameters

**Temporal Models**:
- LSTM neural network with attention mechanism
- 18 input features, 64 hidden units, 2 layers

**Ensemble System**:
- Weighted voting ensemble combining all models
- Confidence scoring based on model agreement
- Uncertainty estimation for robust predictions

#### 4. Multi-Agent System
The system employs a multi-agent architecture for comprehensive fire detection:

**Monitoring Agent**:
- Continuous data stream monitoring
- Anomaly detection in real-time sensor data
- Sensor health monitoring and data quality assessment

**Analysis Agent**:
- In-depth pattern analysis with historical correlation
- Confidence level calculation for fire assessments
- Detailed risk assessment generation

**Response Agent**:
- Response level determination based on risk assessment
- Alert distribution with severity-based routing
- Escalation protocols for critical situations

**Learning Agent**:
- Performance tracking with metrics collection
- Error pattern analysis for system improvement
- Model retraining recommendations

### AWS Integration

#### S3 Integration
- Storage of training datasets and model artifacts
- Data backup and archival capabilities
- Secure access control through IAM policies

#### SageMaker Integration
- Model training on scalable compute instances
- Automated hyperparameter tuning
- Model deployment and endpoint management

#### IoT Core Integration
- Device connectivity for real FLIR+SCD41 sensors
- MQTT message routing and processing
- Device shadow management

#### CloudWatch Integration
- System monitoring and alerting
- Performance metrics collection
- Log aggregation and analysis

## Data Flow

1. **Sensor Data Ingestion**: Raw data from FLIR Lepton 3.5 and SCD41 sensors
2. **Feature Extraction**: Processing of 15 thermal + 3 gas features
3. **Feature Validation**: Quality checking and data validation
4. **Model Inference**: ML models process 18-feature input vector
5. **Ensemble Decision**: Combined prediction with confidence scoring
6. **Agent Analysis**: Multi-agent system processes results
7. **Response Generation**: Appropriate alerts and actions
8. **Feedback Loop**: Learning agent updates system performance

## System Performance

### Accuracy Metrics
- **Fire Detection Accuracy**: >95%
- **False Positive Rate**: <2%
- **False Negative Rate**: <1%
- **Average Response Time**: <500ms

### Feature Importance
1. `t_max` (Max temperature) - 15% importance
2. `gas_val` (CO₂ concentration) - 14% importance
3. `t_hot_area_pct` (Hot area percentage) - 12% importance
4. `t_grad_mean` (Temperature gradient) - 10% importance
5. `gas_delta` (CO₂ change rate) - 9% importance
6. Other features contribute 5% or less each

## Deployment Architecture

### Edge Deployment
- Raspberry Pi or similar edge device
- Real-time processing with minimal latency
- Local decision making with cloud backup

### Cloud Deployment
- AWS infrastructure for training and analytics
- Scalable processing for large deployments
- Centralized management and monitoring

### Hybrid Deployment
- Edge processing for real-time decisions
- Cloud processing for model training and analytics
- Seamless integration between edge and cloud

## Security Considerations

### Authentication
- AWS IAM for service authentication
- Device certificates for sensor authentication
- Role-based access control for system components

### Data Encryption
- TLS encryption for data in transit
- AES-256 encryption for data at rest
- Secure key management through AWS KMS

### Access Control
- Fine-grained permissions through IAM policies
- Network isolation through VPCs
- Regular security audits and compliance checks

## Scalability Considerations

### Horizontal Scaling
- Multiple sensor pairs can be deployed
- Load balancing across processing nodes
- Distributed processing for large installations

### Vertical Scaling
- Increased compute resources for complex scenarios
- Memory optimization for large datasets
- GPU acceleration for deep learning models

### Auto Scaling
- Automatic scaling based on workload
- Cost optimization through resource management
- Performance monitoring and adjustment

## Monitoring and Logging

### System Metrics
- Processing latency and throughput
- Model accuracy and confidence scores
- Sensor health and data quality metrics

### Alerting System
- Real-time alerts for fire detection
- System health notifications
- Performance degradation warnings

### Log Management
- Centralized log aggregation
- Structured logging for analysis
- Long-term storage for compliance

## Maintenance and Updates

### Model Updates
- Continuous learning from new data
- Periodic retraining with updated datasets
- A/B testing for model improvements

### System Updates
- Over-the-air updates for edge devices
- Backward compatibility for older versions
- Rollback capabilities for failed updates

### Performance Optimization
- Regular performance benchmarking
- Resource utilization monitoring
- Optimization recommendations from learning agent