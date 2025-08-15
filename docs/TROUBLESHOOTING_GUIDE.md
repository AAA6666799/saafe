# Saafe Fire Detection MVP - Troubleshooting Guide

## Table of Contents
1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [Installation Problems](#installation-problems)
4. [Performance Issues](#performance-issues)
5. [Model and AI Issues](#model-and-ai-issues)
6. [Notification Problems](#notification-problems)
7. [Data and Export Issues](#data-and-export-issues)
8. [UI and Display Problems](#ui-and-display-problems)
9. [System Integration Issues](#system-integration-issues)
10. [Advanced Troubleshooting](#advanced-troubleshooting)
11. [Log Analysis](#log-analysis)
12. [Recovery Procedures](#recovery-procedures)

## Quick Diagnostics

### System Health Check
Run this quick diagnostic to identify common issues:

```bash
# Run built-in diagnostics
python -c "
from saafe_mvp.utils.diagnostics import run_system_diagnostics
results = run_system_diagnostics()
print('System Status:', results['overall_status'])
for component, status in results['components'].items():
    print(f'{component}: {status}')
"
```

### Basic Troubleshooting Steps
1. **Restart the Application**: Close and reopen Saafe
2. **Check System Resources**: Ensure adequate RAM and CPU availability
3. **Verify Installation**: Confirm all files are present and not corrupted
4. **Review Error Logs**: Check recent log entries for specific errors
5. **Test with Default Settings**: Reset configuration to defaults

### Emergency Recovery
If the application won't start at all:
1. Delete user configuration: `config/user_config.json`
2. Clear cache directory: `data/cache/`
3. Restart with safe mode: `python app.py --safe-mode`

## Common Issues

### Issue: Application Won't Start

#### Symptoms
- Error message on launch
- Application crashes immediately
- No response when clicking executable

#### Possible Causes
- Missing dependencies
- Corrupted installation
- Insufficient permissions
- Antivirus interference
- System compatibility issues

#### Solutions

**Solution 1: Check System Requirements**
```bash
# Verify Python version (if running from source)
python --version  # Should be 3.8+

# Check available memory
# Windows: Task Manager > Performance > Memory
# macOS: Activity Monitor > Memory
# Linux: free -h
```

**Solution 2: Run as Administrator**
- Windows: Right-click executable → "Run as administrator"
- macOS: `sudo ./saafe-mvp-macos.app/Contents/MacOS/saafe-mvp`
- Linux: `sudo ./saafe-mvp-linux`

**Solution 3: Check Antivirus**
- Add Saafe to antivirus whitelist
- Temporarily disable real-time protection
- Check quarantine folder for blocked files

**Solution 4: Reinstall Application**
```bash
# Backup configuration first
cp config/user_config.json config/user_config_backup.json

# Clean installation
rm -rf saafe-mvp/  # Remove old installation
# Download and install fresh copy
# Restore configuration
cp config/user_config_backup.json config/user_config.json
```

### Issue: No Sensor Data Displayed

#### Symptoms
- Dashboard shows all zeros
- No readings updating
- Scenario buttons don't work

#### Possible Causes
- No scenario selected
- Data generation failure
- Threading issues
- Memory constraints

#### Solutions

**Solution 1: Select a Scenario**
1. Click one of the scenario buttons (Normal, Cooking, Fire)
2. Wait 2-3 seconds for data generation to start
3. Verify the active scenario indicator is highlighted

**Solution 2: Restart Data Generation**
```python
# In Python console or debug mode
from saafe_mvp.core.scenario_manager import ScenarioManager, ScenarioType

manager = ScenarioManager()
manager.stop_scenario()  # Stop any running scenario
manager.start_scenario(ScenarioType.NORMAL)  # Start fresh
```

**Solution 3: Check System Resources**
- Close other applications to free memory
- Check CPU usage isn't at 100%
- Verify disk space availability

### Issue: High Processing Times

#### Symptoms
- Processing times >100ms consistently
- Slow UI response
- High CPU usage

#### Possible Causes
- Insufficient system resources
- GPU not being utilized
- Model loading issues
- Background processes

#### Solutions

**Solution 1: Enable GPU Acceleration**
```python
# Check GPU availability
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

**Solution 2: Optimize Performance Settings**
1. Open Settings → System Preferences
2. Change Performance Mode to "Fast Response"
3. Increase Update Frequency to 2-3 seconds
4. Disable anti-hallucination if not needed

**Solution 3: System Optimization**
- Close unnecessary applications
- Disable Windows visual effects
- Set application to "High Priority" in Task Manager
- Ensure adequate cooling (check CPU temperature)

## Installation Problems

### Issue: Missing Dependencies

#### Symptoms
- Import errors on startup
- "Module not found" errors
- Incomplete functionality

#### Solutions

**For Source Installation:**
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Install specific missing modules
pip install torch torchvision
pip install streamlit plotly pandas numpy
pip install twilio smtplib

# Clear pip cache if issues persist
pip cache purge
```

**For Executable Installation:**
- Download complete installer package
- Verify file integrity (check file size matches expected)
- Run installer with administrator privileges

### Issue: Permission Errors

#### Symptoms
- "Access denied" errors
- Cannot write to directories
- Configuration not saving

#### Solutions

**Windows:**
```cmd
# Run as administrator
# Or change folder permissions
icacls "C:\Program Files\Saafe" /grant Users:F /T
```

**macOS/Linux:**
```bash
# Fix permissions
chmod -R 755 /Applications/Saafe.app
# Or for user directory
chmod -R 755 ~/Applications/Saafe
```

### Issue: Firewall/Antivirus Blocking

#### Symptoms
- Network features not working
- Application blocked on startup
- Files quarantined

#### Solutions
1. **Add to Firewall Exceptions**:
   - Windows Defender: Settings → Update & Security → Windows Security → Firewall
   - Add Saafe executable to allowed apps

2. **Antivirus Whitelist**:
   - Add Saafe installation directory to exclusions
   - Add executable file to trusted applications

3. **Network Configuration**:
   - Allow outbound connections on ports 587 (SMTP), 443 (HTTPS)
   - Configure proxy settings if behind corporate firewall

## Performance Issues

### Issue: High Memory Usage

#### Symptoms
- Memory usage >2GB
- System slowdown
- Out of memory errors

#### Diagnostic Steps
```python
# Check memory usage
import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")

# Check GPU memory (if available)
import torch
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
```

#### Solutions
1. **Reduce Buffer Sizes**:
   ```python
   # In configuration
   {
     "performance_settings": {
       "max_stored_readings": 1000,  # Reduce from default
       "history_buffer_size": 500
     }
   }
   ```

2. **Enable Memory Optimization**:
   - Settings → Performance → Enable "Memory Optimization"
   - Reduce update frequency
   - Disable detailed logging

3. **System-Level Fixes**:
   - Increase virtual memory/swap space
   - Close other memory-intensive applications
   - Restart application periodically for long sessions

### Issue: Slow Startup

#### Symptoms
- Application takes >30 seconds to start
- Loading screen appears stuck
- Timeout errors during initialization

#### Solutions
1. **Model Loading Optimization**:
   ```python
   # Enable fast model loading
   {
     "model_settings": {
       "lazy_loading": true,
       "preload_models": false,
       "use_fallback_first": true
     }
   }
   ```

2. **Disable Unnecessary Features**:
   - Turn off auto-export
   - Disable notification services during startup
   - Skip model validation on startup

3. **System Optimization**:
   - Install on SSD instead of HDD
   - Defragment disk (Windows)
   - Clear temporary files

## Model and AI Issues

### Issue: Model Loading Failures

#### Symptoms
- "Model not found" errors
- Fallback model warnings
- Prediction errors

#### Diagnostic Steps
```python
# Check model files
import os
model_path = "models/transformer_model.pth"
print(f"Model exists: {os.path.exists(model_path)}")
print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")

# Test model loading
from saafe_mvp.models.model_manager import ModelManager
manager = ModelManager()
success, message = manager.load_model(model_path)
print(f"Load success: {success}, Message: {message}")
```

#### Solutions
1. **Verify Model Files**:
   - Check `models/` directory contains required files
   - Verify file integrity (not corrupted)
   - Re-download models if necessary

2. **Use Fallback Models**:
   ```python
   # Force fallback model creation
   manager = ModelManager()
   success, message = manager.create_fallback_model()
   ```

3. **Model Path Configuration**:
   ```json
   {
     "model_settings": {
       "model_path": "models/transformer_model.pth",
       "fallback_enabled": true,
       "auto_download": true
     }
   }
   ```

### Issue: Incorrect Predictions

#### Symptoms
- Risk scores don't match scenario
- Normal conditions showing high risk
- Fire scenarios showing low risk

#### Diagnostic Steps
```python
# Test prediction pipeline
from saafe_mvp.core.fire_detection_pipeline import FireDetectionPipeline
from saafe_mvp.core.data_models import SensorReading
from datetime import datetime

pipeline = FireDetectionPipeline(model_manager)

# Test with known values
test_reading = SensorReading(
    timestamp=datetime.now(),
    temperature=25.0,  # Normal
    pm25=10.0,         # Normal
    co2=400.0,         # Normal
    audio_level=35.0,  # Normal
    location="test"
)

result = pipeline.predict([test_reading])
print(f"Risk score: {result.risk_score}")
print(f"Predicted class: {result.predicted_class}")
```

#### Solutions
1. **Model Recalibration**:
   - Reset to default model
   - Clear model cache
   - Restart application

2. **Data Validation**:
   - Verify sensor readings are in expected ranges
   - Check for data corruption
   - Validate preprocessing pipeline

3. **Anti-Hallucination Settings**:
   ```python
   # Adjust anti-hallucination sensitivity
   {
     "anti_hallucination": {
       "enabled": true,
       "cooking_threshold": 0.7,
       "confidence_threshold": 0.8
     }
   }
   ```

### Issue: Anti-Hallucination Not Working

#### Symptoms
- False fire alarms during cooking
- Cooking scenarios triggering critical alerts
- Anti-hallucination messages not appearing

#### Solutions
1. **Enable Anti-Hallucination**:
   ```python
   # In pipeline initialization
   pipeline = FireDetectionPipeline(
       model_manager,
       enable_anti_hallucination=True
   )
   ```

2. **Adjust Thresholds**:
   ```json
   {
     "anti_hallucination_settings": {
       "cooking_detection_threshold": 0.6,
       "ensemble_agreement_threshold": 0.7,
       "conservative_mode": true
     }
   }
   ```

3. **Verify Ensemble Models**:
   - Ensure multiple models are loaded
   - Check ensemble voting is working
   - Review cooking pattern detection

## Notification Problems

### Issue: SMS Not Working

#### Symptoms
- No SMS messages received
- SMS test fails
- Twilio errors in logs

#### Diagnostic Steps
```python
# Test SMS configuration
from saafe_mvp.services.sms_service import SMSService, SMSConfig

config = SMSConfig(
    account_sid="your_account_sid",
    auth_token="your_auth_token",
    from_number="+1234567890"
)

sms_service = SMSService(config)
result = sms_service.send_test_sms("+1234567890")
print(f"SMS test result: {result}")
```

#### Solutions
1. **Verify Twilio Configuration**:
   - Check account SID and auth token
   - Verify phone number is verified in Twilio
   - Ensure sufficient Twilio credits

2. **Phone Number Format**:
   - Use international format: +1234567890
   - Remove spaces, dashes, parentheses
   - Verify number is SMS-capable

3. **Network Issues**:
   - Check internet connectivity
   - Verify firewall allows outbound HTTPS
   - Test with different phone number

### Issue: Email Not Working

#### Symptoms
- No emails received
- SMTP authentication errors
- Email test fails

#### Solutions
1. **SMTP Configuration**:
   ```json
   {
     "email_settings": {
       "smtp_server": "smtp.gmail.com",
       "smtp_port": 587,
       "use_tls": true,
       "username": "your_email@gmail.com",
       "password": "your_app_password"
     }
   }
   ```

2. **Gmail App Passwords**:
   - Enable 2-factor authentication
   - Generate app-specific password
   - Use app password instead of regular password

3. **Email Provider Settings**:
   - **Gmail**: smtp.gmail.com:587 (TLS)
   - **Outlook**: smtp-mail.outlook.com:587 (TLS)
   - **Yahoo**: smtp.mail.yahoo.com:587 (TLS)

### Issue: Push Notifications Not Working

#### Symptoms
- No browser notifications
- Permission denied errors
- Push test fails

#### Solutions
1. **Browser Permissions**:
   - Allow notifications when prompted
   - Check browser notification settings
   - Clear browser cache and cookies

2. **Browser Compatibility**:
   - Use Chrome, Firefox, or Edge
   - Update browser to latest version
   - Disable ad blockers temporarily

3. **System Notifications**:
   - Enable system notifications (Windows/macOS)
   - Check Do Not Disturb settings
   - Verify notification center is working

## Data and Export Issues

### Issue: Export Fails

#### Symptoms
- Export buttons don't work
- Empty export files
- Export errors in logs

#### Diagnostic Steps
```python
# Test export functionality
from saafe_mvp.services.export_service import ExportService
from saafe_mvp.services.session_manager import SessionManager

export_service = ExportService()
session_manager = SessionManager()

# Start test session
session_id = session_manager.start_session("test_export")
session_manager.end_session()

# Try export
try:
    json_path = export_service.export_to_json(session_id)
    print(f"Export successful: {json_path}")
except Exception as e:
    print(f"Export failed: {e}")
```

#### Solutions
1. **Check Permissions**:
   - Verify write permissions to export directory
   - Create exports directory if missing
   - Run with administrator privileges

2. **Session Data**:
   - Ensure session has data to export
   - Run scenario before exporting
   - Check session is properly ended

3. **File System Issues**:
   - Check disk space availability
   - Verify export path is valid
   - Clear temporary files

### Issue: Missing Data in Exports

#### Symptoms
- Partial data in export files
- Missing sensor readings
- Incomplete session information

#### Solutions
1. **Session Management**:
   ```python
   # Ensure proper session lifecycle
   session_manager.start_session("my_session")
   # ... collect data ...
   session_manager.end_session()  # Important!
   ```

2. **Data Collection**:
   - Verify scenarios are running during data collection
   - Check data buffer sizes
   - Ensure adequate session duration

3. **Export Configuration**:
   ```json
   {
     "export_settings": {
       "include_raw_data": true,
       "include_predictions": true,
       "include_alerts": true,
       "date_range": "all"
     }
   }
   ```

## UI and Display Problems

### Issue: Dashboard Not Loading

#### Symptoms
- Blank dashboard
- Loading screen stuck
- UI components missing

#### Solutions
1. **Browser Issues**:
   - Clear browser cache
   - Disable browser extensions
   - Try different browser
   - Check JavaScript is enabled

2. **Streamlit Issues**:
   ```bash
   # Clear Streamlit cache
   streamlit cache clear
   
   # Restart with fresh cache
   python app.py --server.enableCORS false
   ```

3. **Port Conflicts**:
   - Check if port 8501 is in use
   - Change port in configuration
   - Kill conflicting processes

### Issue: Charts Not Displaying

#### Symptoms
- Empty chart areas
- Chart loading errors
- Plotly errors in console

#### Solutions
1. **JavaScript Issues**:
   - Enable JavaScript in browser
   - Check browser console for errors
   - Update browser to latest version

2. **Data Issues**:
   - Verify data is being generated
   - Check chart data format
   - Clear chart cache

3. **Plotly Configuration**:
   ```python
   # Reset Plotly configuration
   import plotly.io as pio
   pio.renderers.default = "browser"
   ```

### Issue: Responsive Layout Problems

#### Symptoms
- UI elements overlapping
- Text cut off
- Poor mobile display

#### Solutions
1. **Browser Zoom**:
   - Reset browser zoom to 100%
   - Use Ctrl+0 (Windows) or Cmd+0 (Mac)

2. **Screen Resolution**:
   - Minimum resolution: 1024x768
   - Adjust display scaling
   - Use full-screen mode

3. **Streamlit Configuration**:
   ```toml
   # .streamlit/config.toml
   [theme]
   base = "light"
   primaryColor = "#FF6B6B"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F0F2F6"
   textColor = "#262730"
   ```

## System Integration Issues

### Issue: Component Communication Failures

#### Symptoms
- Components not responding
- Data not flowing between modules
- Threading errors

#### Diagnostic Steps
```python
# Test component integration
from saafe_mvp.core.scenario_manager import ScenarioManager
from saafe_mvp.core.fire_detection_pipeline import FireDetectionPipeline
from saafe_mvp.models.model_manager import ModelManager

# Test each component
manager = ModelManager()
print(f"Model manager: {manager.get_system_status()}")

scenario_manager = ScenarioManager()
print(f"Scenario manager: {scenario_manager.get_scenario_info()}")

pipeline = FireDetectionPipeline(manager)
print(f"Pipeline: {pipeline.get_performance_metrics()}")
```

#### Solutions
1. **Component Initialization Order**:
   ```python
   # Correct initialization sequence
   model_manager = ModelManager()
   model_manager.create_fallback_model()
   
   pipeline = FireDetectionPipeline(model_manager)
   scenario_manager = ScenarioManager()
   alert_engine = AlertEngine()
   ```

2. **Threading Issues**:
   - Check for deadlocks
   - Verify thread safety
   - Use proper synchronization

3. **Memory Sharing**:
   - Verify shared data structures
   - Check for race conditions
   - Use thread-safe collections

### Issue: Configuration Not Applied

#### Symptoms
- Settings changes not taking effect
- Default values used instead of configured
- Configuration file not loading

#### Solutions
1. **Configuration File Location**:
   ```python
   # Check configuration paths
   import os
   config_paths = [
       "config/user_config.json",
       "config/app_config.json",
       os.path.expanduser("~/.saafe/config.json")
   ]
   
   for path in config_paths:
       print(f"{path}: {os.path.exists(path)}")
   ```

2. **Configuration Format**:
   - Verify JSON syntax is valid
   - Check for missing commas/brackets
   - Validate configuration schema

3. **Configuration Loading**:
   - Restart application after changes
   - Check file permissions
   - Verify configuration is not read-only

## Advanced Troubleshooting

### Memory Leak Detection

```python
# Monitor memory usage over time
import psutil
import time
import matplotlib.pyplot as plt

def monitor_memory(duration_minutes=10):
    times = []
    memory_usage = []
    
    start_time = time.time()
    while time.time() - start_time < duration_minutes * 60:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        times.append(time.time() - start_time)
        memory_usage.append(memory_mb)
        
        time.sleep(10)  # Check every 10 seconds
    
    plt.plot(times, memory_usage)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Time')
    plt.show()

# Run memory monitoring
monitor_memory(5)  # Monitor for 5 minutes
```

### Performance Profiling

```python
# Profile application performance
import cProfile
import pstats

def profile_prediction():
    from saafe_mvp.core.fire_detection_pipeline import FireDetectionPipeline
    from saafe_mvp.models.model_manager import ModelManager
    from saafe_mvp.core.data_models import SensorReading
    from datetime import datetime
    
    # Setup
    model_manager = ModelManager()
    model_manager.create_fallback_model()
    pipeline = FireDetectionPipeline(model_manager)
    
    # Create test data
    readings = [
        SensorReading(
            timestamp=datetime.now(),
            temperature=25.0,
            pm25=15.0,
            co2=450.0,
            audio_level=40.0,
            location="test"
        )
    ]
    
    # Profile predictions
    def run_predictions():
        for _ in range(100):
            result = pipeline.predict(readings)
    
    # Run profiler
    profiler = cProfile.Profile()
    profiler.enable()
    run_predictions()
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

# Run profiling
profile_prediction()
```

### Network Diagnostics

```python
# Test network connectivity for notifications
import requests
import socket

def test_network_connectivity():
    tests = [
        ("Google DNS", "8.8.8.8", 53),
        ("Twilio API", "api.twilio.com", 443),
        ("Gmail SMTP", "smtp.gmail.com", 587),
        ("Outlook SMTP", "smtp-mail.outlook.com", 587)
    ]
    
    for name, host, port in tests:
        try:
            socket.create_connection((host, port), timeout=5)
            print(f"✓ {name}: Connected")
        except Exception as e:
            print(f"✗ {name}: Failed - {e}")

# Test HTTP connectivity
def test_http_connectivity():
    urls = [
        "https://api.twilio.com",
        "https://smtp.gmail.com",
        "https://www.google.com"
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=5)
            print(f"✓ {url}: {response.status_code}")
        except Exception as e:
            print(f"✗ {url}: Failed - {e}")

# Run network tests
test_network_connectivity()
test_http_connectivity()
```

## Log Analysis

### Log File Locations
- **Windows**: `%APPDATA%\Saafe\logs\`
- **macOS**: `~/Library/Application Support/Saafe/logs/`
- **Linux**: `~/.saafe/logs/`

### Important Log Files
- `saafe.log`: Main application log
- `model.log`: AI model operations
- `performance.log`: Performance metrics
- `error.log`: Error conditions
- `notification.log`: Notification events

### Log Analysis Commands

```bash
# Find recent errors
grep -i "error" logs/saafe.log | tail -20

# Check model loading issues
grep -i "model" logs/saafe.log | grep -i "error"

# Monitor performance issues
grep -i "processing time" logs/performance.log | tail -10

# Check notification failures
grep -i "notification.*failed" logs/notification.log

# Find memory issues
grep -i "memory\|out of memory" logs/saafe.log

# Check startup issues
grep -A 5 -B 5 "initialization" logs/saafe.log
```

### Log Level Configuration

```json
{
  "logging": {
    "level": "INFO",
    "file_level": "DEBUG",
    "console_level": "INFO",
    "max_file_size": "10MB",
    "backup_count": 5
  }
}
```

## Recovery Procedures

### Complete System Reset

```bash
# 1. Stop application
pkill -f saafe

# 2. Backup important data
mkdir backup_$(date +%Y%m%d)
cp -r config/ backup_$(date +%Y%m%d)/
cp -r exports/ backup_$(date +%Y%m%d)/

# 3. Clear all data
rm -rf config/user_config.json
rm -rf data/cache/
rm -rf logs/

# 4. Reset to defaults
python app.py --reset-config

# 5. Restore important settings
# Manually reconfigure notifications, etc.
```

### Model Recovery

```python
# Reset models to default state
from saafe_mvp.models.model_manager import ModelManager

def reset_models():
    manager = ModelManager()
    
    # Clear model registry
    manager.registry.models.clear()
    manager.registry.metadata.clear()
    
    # Create fresh fallback model
    success, message = manager.create_fallback_model()
    print(f"Model reset: {success}, {message}")
    
    return manager

# Execute model reset
manager = reset_models()
```

### Configuration Recovery

```python
# Restore configuration from backup
import json
import shutil
from datetime import datetime

def restore_configuration(backup_path):
    try:
        # Backup current config
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy('config/user_config.json', f'config/user_config_backup_{timestamp}.json')
        
        # Restore from backup
        shutil.copy(backup_path, 'config/user_config.json')
        
        # Validate restored config
        with open('config/user_config.json', 'r') as f:
            config = json.load(f)
        
        print("Configuration restored successfully")
        return True
        
    except Exception as e:
        print(f"Configuration restore failed: {e}")
        return False

# Usage
# restore_configuration('backup_20241201/user_config.json')
```

### Emergency Contacts

For critical issues that cannot be resolved:

1. **Check Documentation**: Review user manual and technical documentation
2. **Search Logs**: Look for specific error messages in log files
3. **System Information**: Collect system specs, OS version, error messages
4. **Screenshots**: Capture any error dialogs or unusual behavior
5. **Contact Support**: Provide detailed information about the issue

### Support Information Template

```
Saafe MVP Support Request

System Information:
- OS: [Windows 10/macOS 11.6/Ubuntu 20.04]
- RAM: [8GB]
- CPU: [Intel i7-8700K]
- GPU: [NVIDIA GTX 1060 / None]
- Python Version: [3.9.7] (if applicable)

Application Information:
- Saafe Version: [1.0.0]
- Installation Type: [Executable/Source]
- Installation Date: [2024-12-01]

Issue Description:
- What were you trying to do?
- What happened instead?
- When did this start occurring?
- Any recent changes to system/configuration?

Error Messages:
[Paste any error messages here]

Log Excerpts:
[Paste relevant log entries here]

Steps to Reproduce:
1. [Step 1]
2. [Step 2]
3. [Step 3]

Troubleshooting Attempted:
- [List what you've already tried]
```

---

*This troubleshooting guide covers the most common issues encountered with Saafe Fire Detection MVP. For additional support, please refer to the user manual and technical documentation.*

**Version**: 1.0.0  
15 August 2025 
**Document Version**: 1.0