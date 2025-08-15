# Saafe Fire Detection MVP - User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Getting Started](#getting-started)
5. [Using the Dashboard](#using-the-dashboard)
6. [Scenario Testing](#scenario-testing)
7. [Mobile Notifications](#mobile-notifications)
8. [Settings and Configuration](#settings-and-configuration)
9. [Export and Reporting](#export-and-reporting)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

## Introduction

Welcome to Saafe Fire Detection MVP, an intelligent fire detection system that demonstrates advanced AI-powered fire risk assessment without requiring physical sensors. This software-only solution simulates realistic sensor environments and provides real-time fire risk analysis through sophisticated machine learning models.

### Key Features
- **Real-time Fire Detection**: AI-powered analysis of simulated sensor data
- **Anti-Hallucination Technology**: Prevents false alarms during cooking activities
- **Multiple Scenarios**: Test normal, cooking, and fire emergency situations
- **Mobile Alerts**: SMS, email, and push notifications for critical alerts
- **Professional Dashboard**: Clean, intuitive interface with real-time visualizations
- **Export Capabilities**: Generate reports and export data for analysis
- **Offline Operation**: Works completely offline without internet connectivity

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10, macOS 10.14, or Linux (Ubuntu 18.04+)
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **CPU**: Intel i5 or AMD equivalent (2.0GHz+)
- **Python**: 3.8 or higher (if running from source)

### Recommended Requirements
- **RAM**: 8GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)
- **CPU**: Intel i7 or AMD Ryzen 5 (3.0GHz+)

## Installation

### Option 1: Standalone Executable (Recommended)
1. Download the appropriate executable for your operating system:
   - Windows: `saafe-mvp-windows.exe`
   - macOS: `saafe-mvp-macos.app`
   - Linux: `saafe-mvp-linux`

2. Run the installer and follow the setup wizard
3. Launch Saafe from your applications menu or desktop shortcut

### Option 2: From Source Code
1. Clone or download the source code
2. Install Python 3.8+ if not already installed
3. Open terminal/command prompt in the project directory
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   python app.py
   ```

## Getting Started

### First Launch
1. **Launch the Application**: Double-click the Saafe icon or run from terminal
2. **Setup Wizard**: On first launch, you'll see a setup wizard:
   - Choose your preferred theme (Light/Dark)
   - Configure notification preferences
   - Test your system performance
3. **Dashboard**: After setup, the main dashboard will appear

### Quick Start Guide
1. **Select a Scenario**: Click one of the three scenario buttons:
   - Normal Environment (stable conditions)
   - Cooking Activity (elevated PM2.5/CO₂)
   - Fire Emergency (critical conditions)

2. **Monitor the Dashboard**: Watch real-time sensor readings and AI analysis

3. **Observe Alerts**: See how the system responds to different risk levels

4. **Test Notifications**: Configure and test mobile alerts in Settings

## Using the Dashboard

### Main Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│  🔥 Saafe                           ⚙️  📊  ❓          │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Normal Environment  Cooking Activity  Fire Emergency   │ │
│  │       ●                   ○                ○           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │    22°C      12 μg/m³     400 ppm      35 dB          │ │
│  │  Temperature   PM2.5       CO₂       Audio            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Risk Score: 15    Alert Level: Normal                 │ │
│  │  Confidence: 94%   Processing: 23ms                    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  🟢 All Systems Normal                                  │ │
│  │  📱 Mobile alerts configured                           │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Dashboard Components

#### 1. Header Bar
- **Saafe Logo**: Application branding
- **Settings (⚙️)**: Access configuration options
- **Charts (📊)**: View detailed sensor trends
- **Help (❓)**: Access help and documentation

#### 2. Scenario Controls
- **Normal Environment**: Simulates stable baseline conditions
- **Cooking Activity**: Elevated PM2.5 and CO₂ without fire risk
- **Fire Emergency**: Rapid temperature increases and fire indicators
- **Active Indicator**: Shows which scenario is currently running

#### 3. Sensor Display Panel
- **Temperature**: Current temperature reading in Celsius
- **PM2.5**: Particulate matter concentration (μg/m³)
- **CO₂**: Carbon dioxide levels (ppm)
- **Audio**: Sound level in decibels
- **Color Coding**: Green (normal), Yellow (elevated), Red (critical)

#### 4. AI Analysis Panel
- **Risk Score**: 0-100 scale fire risk assessment
- **Alert Level**: Normal, Mild Anomaly, Elevated Risk, or Critical
- **Confidence**: AI model confidence percentage
- **Processing Time**: Model inference speed in milliseconds

#### 5. System Status Panel
- **System Status**: Overall system health indicator
- **Mobile Alerts**: Notification configuration status
- **Recent Events**: Log of recent alerts and system messages

## Scenario Testing

### Normal Environment Scenario
**Purpose**: Demonstrates stable baseline conditions
- Temperature: 20-25°C
- PM2.5: 5-15 μg/m³
- CO₂: 350-450 ppm
- Audio: 30-40 dB
- **Expected Risk Score**: 0-30 (Normal)

### Cooking Activity Scenario
**Purpose**: Shows elevated readings without fire alerts
- Temperature: 25-35°C
- PM2.5: 20-60 μg/m³ (elevated from cooking)
- CO₂: 450-600 ppm (elevated from cooking)
- Audio: 35-50 dB
- **Expected Risk Score**: 30-50 (Mild Anomaly)
- **Anti-Hallucination**: System detects cooking patterns and prevents false fire alarms

### Fire Emergency Scenario
**Purpose**: Simulates actual fire conditions
- Temperature: 40-80°C (rapid increase)
- PM2.5: 80-200 μg/m³ (smoke particles)
- CO₂: 600-1200 ppm (combustion products)
- Audio: 50-70 dB (fire sounds)
- **Expected Risk Score**: 85-100 (Critical)
- **Mobile Alerts**: Automatic notifications sent to configured devices

### Running Scenarios
1. Click the desired scenario button
2. Watch the sensor readings update in real-time
3. Observe how the AI model responds to different patterns
4. Note the alert level changes and system messages
5. Click another scenario or the same button to stop

## Mobile Notifications

### Notification Types
- **SMS**: Text messages to configured phone numbers
- **Email**: HTML emails with alert details
- **Push Notifications**: Browser-based notifications

### Configuration Steps
1. **Open Settings**: Click the ⚙️ icon in the header
2. **Navigate to Notifications**: Select the notifications tab
3. **Enable Services**: Toggle on desired notification types
4. **Add Contacts**:
   - SMS: Enter phone numbers in international format (+1234567890)
   - Email: Enter valid email addresses
5. **Set Alert Levels**: Choose minimum alert level for each service
6. **Test Notifications**: Use the "Test All" button to verify setup

### Alert Levels and Notifications
- **Normal (0-30)**: No notifications sent
- **Mild Anomaly (31-50)**: Email and push notifications (if enabled)
- **Elevated Risk (51-85)**: All notification types (if enabled)
- **Critical (86-100)**: All notification types sent immediately

### Notification Content
Critical fire alerts include:
- Alert level and risk score
- Timestamp and location
- Sensor readings summary
- Recommended actions
- System confidence level

## Settings and Configuration

### Accessing Settings
Click the ⚙️ (Settings) icon in the dashboard header to open the configuration panel.

### Notification Settings
- **SMS Configuration**:
  - Enable/disable SMS alerts
  - Add/remove phone numbers
  - Set minimum alert level for SMS
  - Test SMS delivery

- **Email Configuration**:
  - Enable/disable email alerts
  - Add/remove email addresses
  - Set minimum alert level for email
  - Test email delivery

- **Push Notifications**:
  - Enable/disable browser notifications
  - Grant notification permissions
  - Test push notifications

### Alert Thresholds
Customize when alerts are triggered:
- **Normal**: 0-30 (default)
- **Mild Anomaly**: 31-50 (default)
- **Elevated Risk**: 51-85 (default)
- **Critical**: 86-100 (default)

### System Preferences
- **Update Frequency**: How often sensor data updates (1-10 seconds)
- **Theme**: Light or dark mode
- **Performance Mode**:
  - Fast Response: Prioritizes speed
  - Balanced: Default setting
  - High Accuracy: Prioritizes precision
- **Model Path**: Location of AI model files (advanced users)

### Saving Settings
Click "Save" to apply changes. Some settings may require restarting the application.

## Export and Reporting

### Session Data Export
The system automatically tracks all sensor readings, predictions, and alerts during operation.

### Export Formats
1. **CSV**: Raw data for spreadsheet analysis
2. **JSON**: Structured data for programmatic use
3. **PDF**: Professional reports with charts and summaries

### Exporting Data
1. **Access Export**: Click the 📊 (Charts) icon in the header
2. **Select Time Range**: Choose the period to export
3. **Choose Format**: Select CSV, JSON, or PDF
4. **Download**: Click "Export" to generate and download the file

### Report Contents
PDF reports include:
- Executive summary
- Sensor data trends
- Alert history
- System performance metrics
- Risk assessment analysis
- Recommendations

### Scheduled Exports
Configure automatic exports in Settings:
- Enable auto-export
- Set export frequency (hourly, daily, weekly)
- Choose export formats
- Specify save location

## Troubleshooting

### Common Issues

#### Application Won't Start
**Symptoms**: Error message on launch or application crashes immediately
**Solutions**:
1. Check system requirements
2. Run as administrator (Windows) or with sudo (Linux)
3. Check antivirus software isn't blocking the application
4. Reinstall the application
5. Check the error log in the application directory

#### No Sensor Data Showing
**Symptoms**: Dashboard shows no readings or all zeros
**Solutions**:
1. Select a scenario by clicking one of the scenario buttons
2. Wait 2-3 seconds for data generation to start
3. Check if the scenario is actually running (active indicator)
4. Restart the application
5. Check system resources (CPU/memory usage)

#### Mobile Notifications Not Working
**Symptoms**: No SMS, email, or push notifications received
**Solutions**:
1. Verify notification settings are enabled
2. Check contact information is correct
3. Use the "Test" buttons to verify each service
4. Check spam/junk folders for emails
5. Ensure browser permissions are granted for push notifications
6. Verify internet connection for SMS/email services

#### Slow Performance
**Symptoms**: High processing times, laggy interface
**Solutions**:
1. Close other applications to free up memory
2. Switch to "Fast Response" performance mode
3. Reduce update frequency in settings
4. Check if GPU acceleration is available
5. Restart the application
6. Check system resources in Task Manager

#### AI Model Errors
**Symptoms**: Error messages about model loading or prediction failures
**Solutions**:
1. The system should automatically use fallback models
2. Check model files exist in the models directory
3. Verify sufficient disk space
4. Restart the application to reload models
5. Check the error log for specific model issues

### Error Logs
Error logs are saved in:
- Windows: `%APPDATA%/Saafe/logs/`
- macOS: `~/Library/Application Support/Saafe/logs/`
- Linux: `~/.saafe/logs/`

### Getting Help
1. Check this user manual
2. Review the FAQ section below
3. Check error logs for specific issues
4. Contact support with log files and system information

## FAQ

### General Questions

**Q: Do I need internet connectivity to use Saafe?**
A: No, Saafe works completely offline. Internet is only needed for SMS and email notifications.

**Q: Can I use Saafe with real sensors?**
A: This MVP version uses simulated data. Future versions will support real sensor integration.

**Q: How accurate is the fire detection?**
A: The AI models are trained on realistic fire patterns and achieve high accuracy in distinguishing between normal conditions, cooking activities, and actual fires.

**Q: Can I run multiple scenarios simultaneously?**
A: No, only one scenario can be active at a time to ensure clear testing results.

### Technical Questions

**Q: What AI models does Saafe use?**
A: Saafe uses Spatio-Temporal Transformer models with ensemble voting and anti-hallucination logic.

**Q: How much system resources does Saafe use?**
A: Typically 200-500MB RAM and minimal CPU usage. GPU acceleration is used when available.

**Q: Can I customize the alert thresholds?**
A: Yes, alert thresholds can be customized in the Settings panel.

**Q: How is cooking activity detected?**
A: The anti-hallucination system uses pattern recognition to identify cooking signatures (elevated PM2.5/CO₂ without rapid temperature increases).

### Notification Questions

**Q: Why am I not receiving SMS notifications?**
A: Check that SMS is enabled, phone numbers are in international format, and the alert level meets your configured threshold.

**Q: Can I customize notification messages?**
A: Currently, notification messages are standardized for clarity and consistency.

**Q: How quickly are critical alerts sent?**
A: Critical alerts are sent immediately when detected, typically within 1-2 seconds.

### Data and Privacy Questions

**Q: What data does Saafe collect?**
A: Saafe only processes simulated sensor data locally. No personal data is transmitted externally.

**Q: Where is my data stored?**
A: All data is stored locally on your device. Session data can be exported for your own analysis.

**Q: Can I delete historical data?**
A: Yes, you can clear session history in the Settings panel or by deleting export files.

---

## Support

For additional support or questions not covered in this manual:

1. **Check Error Logs**: Review application logs for specific error messages
2. **System Information**: Note your operating system, RAM, and any error messages
3. **Screenshots**: Capture screenshots of any issues for better support
4. **Contact Information**: Reach out to the development team with detailed information

---

*Saafe Fire Detection MVP - Intelligent Fire Safety Through Advanced AI*

**Version**: 1.0.0  
15 August 2025 
**Document Version**: 1.0