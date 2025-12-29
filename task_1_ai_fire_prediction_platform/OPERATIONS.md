# Saafe Fire Detection System - Operations Guide

This document provides step-by-step instructions for operating the Saafe Fire Detection System.

## System Overview

The Saafe Fire Detection System processes real sensor data from AWS S3 to provide real-time fire risk assessment. The system consists of:

1. **Data Ingestion**: Reads live sensor data from AWS S3
2. **Feature Processing**: Extracts meaningful features from sensor data
3. **Risk Assessment**: Uses ML models to predict fire risk
4. **Dashboard**: Provides real-time visualization of sensor data and risk assessment

## Features

- **Real S3 Data Integration**: Processes live thermal, gas, and environmental sensor data from AWS S3
- **Feature Engineering**: Comprehensive feature extraction from multi-sensor data
- **Advanced ML Models**: Temporal models, baseline models, and ensemble systems
- **Interactive Dashboard**: Real-time visualization of sensor data and fire risk
- **Data Provenance**: Clear proof of AWS S3 and IoT device data sources
- **Comprehensive Testing**: Automated testing with real data validation

## Prerequisites

Before operating the system, ensure you have:

1. **AWS Credentials**: Valid AWS credentials with access to the `data-collector-of-first-device` S3 bucket
2. **Python 3.8+**: Installed on the system
3. **Required Packages**: Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
4. **Network Access**: Ability to connect to AWS services

## System Components

### 1. Core System Manager
- **Location**: `synthetic_fire_system/system/manager.py`
- **Function**: Coordinates all system components
- **Usage**:
  ```bash
  python -m synthetic_fire_system.main
  ```

### 2. Dashboard
- **Location**: `dashboard.py`
- **Function**: Provides real-time visualization of sensor data
- **Usage**:
  ```bash
  python run_dashboard.py
  ```
- **Access**: Open browser to `http://localhost:8505`

### 3. Training System
- **Location**: `train_with_s3_data.py`
- **Function**: Trains ML models using historical S3 data
- **Usage**:
  ```bash
  python train_with_s3_data.py
  ```

### 4. Data Verification
- **Location**: `verify_data_source.py`
- **Function**: Verifies data is coming from AWS S3 and IoT devices
- **Usage**:
  ```bash
  python verify_data_source.py
  ```

## Step-by-Step Operations

### Starting the System

1. **Verify AWS Credentials**:
   ```bash
   aws sts get-caller-identity
   ```
   This should return your AWS account information.

2. **Start the Core System**:
   ```bash
   cd task_1_synthetic_fire_system
   python -m synthetic_fire_system.main
   ```

3. **Start the Dashboard** (in a separate terminal):
   ```bash
   cd task_1_synthetic_fire_system
   python run_dashboard.py
   ```

4. **Access the Dashboard**:
   Open your browser and navigate to `http://localhost:8505`

### Monitoring System Status

1. **Check System Logs**: 
   - The main system will output status information to the console
   - Look for "System started successfully" message

2. **Dashboard Indicators**:
   - Green indicators: Normal operation
   - Yellow indicators: Elevated risk
   - Red indicators: High risk/fire detected

3. **Data Flow Verification**:
   - Dashboard should show updated sensor data every few seconds
   - Thermal images should show real temperature readings
   - Gas sensors should show current gas concentrations
   - Data provenance section shows AWS S3 connection and timestamps

### Training ML Models

1. **Run Training Script**:
   ```bash
   python train_with_s3_data.py
   ```

2. **Monitor Training Progress**:
   - Script will show progress as it collects data samples
   - Training metrics will be displayed upon completion

3. **Model Output**:
   - Trained model will be saved as `trained_fire_model_s3.pkl`

### Verifying Data Authenticity

1. **Run Data Verification Script**:
   ```bash
   python verify_data_source.py
   ```

2. **Check Dashboard Provenance**:
   - View "Data Provenance" section on dashboard
   - Expand "Data Verification Details" for more information
   - Check "Recent S3 Files" to see actual file names and timestamps

3. **Independent Verification**:
   - Stakeholders can run `verify_data_source.py` to independently verify data sources
   - The script shows actual S3 file names, timestamps, and structure

### Troubleshooting

#### Common Issues and Solutions

1. **AWS Connection Failed**:
   - **Symptom**: "Failed to connect to S3" error
   - **Solution**: 
     - Verify AWS credentials: `aws sts get-caller-identity`
     - Check network connectivity
     - Ensure IAM permissions for S3 access

2. **No Data Displayed**:
   - **Symptom**: Dashboard shows "No sensor data available"
   - **Solution**:
     - Verify IoT sensors are active and sending data
     - Check S3 bucket for recent data files
     - Restart the system to refresh connections

3. **Dashboard Not Loading**:
   - **Symptom**: Browser cannot connect to `http://localhost:8505`
   - **Solution**:
     - Verify dashboard script is running
     - Check if port 8505 is available: `lsof -i :8505`
     - Try accessing via Network URL if Local URL fails

4. **Training Script Fails**:
   - **Symptom**: Errors during model training
   - **Solution**:
     - Ensure sufficient data samples are available in S3
     - Check system memory and resources
     - Verify sklearn and other dependencies are properly installed

### Maintenance Procedures

#### Regular Maintenance

1. **Daily**:
   - Check system logs for errors
   - Verify dashboard is displaying current data
   - Confirm AWS credentials are still valid

2. **Weekly**:
   - Review model performance metrics
   - Update ML models with new training data
   - Check system resource usage

3. **Monthly**:
   - Rotate log files
   - Update system dependencies
   - Review and update this operations guide

#### Emergency Procedures

1. **System Failure**:
   - Stop all running processes
   - Check system logs for error messages
   - Restart system components one by one
   - Contact system administrator if issues persist

2. **High Risk Alert**:
   - Immediately verify sensor readings
   - Check for actual fire conditions
   - Follow established emergency protocols
   - Document incident for future analysis

## System Monitoring

### Key Metrics to Monitor

1. **Data Ingestion**:
   - Frequency of data updates
   - Completeness of sensor data
   - Connection status to S3

2. **Model Performance**:
   - Fire risk prediction accuracy
   - False positive/negative rates
   - Model confidence scores

3. **System Health**:
   - CPU and memory usage
   - Network connectivity
   - Disk space availability

### Alerting Thresholds

1. **Critical Alerts**:
   - Fire risk score > 70
   - System offline for > 5 minutes
   - AWS connection failure

2. **Warning Alerts**:
   - Fire risk score 40-70
   - Data staleness > 2 minutes
   - High system resource usage

3. **Informational Alerts**:
   - System startup/shutdown
   - Model retraining completion
   - Configuration changes

## Backup and Recovery

### Backup Procedures

1. **Model Backups**:
   - Copy `trained_fire_model_s3.pkl` to secure location
   - Document model version and training date

2. **Configuration Backups**:
   - Backup configuration files in `config/` directory
   - Document any custom settings

### Recovery Procedures

1. **System Recovery**:
   - Restore configuration files from backup
   - Reinstall dependencies if needed
   - Restart system components

2. **Model Recovery**:
   - Restore trained model from backup
   - Retrain model if backup unavailable

## Contact Information

For system support and maintenance:
- **Primary Contact**: [ch.ajay1707@gmail.com](mailto:ch.ajay1707@gmail.com)
- **AWS Support**: AWS Console Support Center
- **Emergency**: Follow established fire safety protocols

## Revision History

- **Version 1.0**: Initial release
- **Version 1.1**: Updated dashboard port to 8505
- **Version 1.2**: Added data verification features and stakeholder proof