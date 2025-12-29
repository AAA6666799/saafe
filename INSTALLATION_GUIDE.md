# Saafe MVP Installation Guide

This guide covers the complete installation and setup process for the Saafe Fire Detection MVP.

## Overview

The Saafe MVP installation system provides:

- **Automated dependency checking** and installation
- **Model download** and validation
- **First-run setup wizard** for configuration
- **Platform-specific startup scripts**
- **Automatic update system**

## Quick Installation

### Prerequisites

- **Python 3.8+** installed on your system
- **Internet connection** for dependency installation
- **1GB free disk space** minimum

### One-Command Installation

```bash
# Download and run the installer
python installer.py
```

The installer will:
1. Check system requirements
2. Install Python dependencies
3. Download AI models
4. Create configuration files
5. Set up startup scripts
6. Run first-time setup wizard

## Manual Installation

If you prefer manual installation or need to troubleshoot:

### 1. System Requirements

**Minimum Requirements:**
- Python 3.8 or later
- 1GB free disk space
- 4GB RAM
- Internet connection (for initial setup)

**Recommended:**
- Python 3.9 or later
- 2GB free disk space
- 8GB RAM
- GPU support (optional, for faster processing)

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install build tools (optional)
pip install -r build_requirements.txt
```

### 3. Create Directory Structure

```bash
# Create required directories
mkdir -p config models data/cache logs exports
```

### 4. Download Models

The installer automatically downloads placeholder models. In a production deployment, you would:

```bash
# Download actual AI models (URLs would be provided)
# wget https://models.saafe-mvp.com/transformer_model.pth -O models/
# wget https://models.saafe-mvp.com/anti_hallucination.pkl -O models/
```

### 5. Configure Application

Run the setup wizard:

```bash
python setup_wizard.py
```

Or manually create configuration files (see Configuration section).

## Platform-Specific Installation

### Windows

1. **Install Python** from [python.org](https://python.org)
2. **Run installer:**
   ```cmd
   python installer.py
   ```
3. **Start application:**
   ```cmd
   start_safeguard.bat
   ```

### macOS

1. **Install Python** (if not already installed):
   ```bash
   # Using Homebrew
   brew install python
   
   # Or download from python.org
   ```

2. **Run installer:**
   ```bash
   python3 installer.py
   ```

3. **Start application:**
   ```bash
   ./start_safeguard.sh
   ```

### Linux

1. **Install Python** (most distributions include it):
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install python3 python3-pip
   
   # CentOS/RHEL
   sudo yum install python3 python3-pip
   
   # Arch Linux
   sudo pacman -S python python-pip
   ```

2. **Run installer:**
   ```bash
   python3 installer.py
   ```

3. **Start application:**
   ```bash
   ./start_safeguard.sh
   ```

## Configuration

### Automatic Configuration

The setup wizard guides you through:

- **Theme selection** (light/dark)
- **Update frequency** (0.5s, 1s, 2s)
- **Server port** (default: 8501)
- **Notification settings** (SMS, email, push)
- **Alert thresholds** (normal, mild, elevated, critical)
- **Advanced options** (GPU/CPU, debug mode)

### Manual Configuration

Configuration files are stored in the `config/` directory:

#### `config/app_config.json`
```json
{
  "app": {
    "name": "Saafe Fire Detection MVP",
    "version": "1.0.0",
    "debug": false
  },
  "models": {
    "transformer_path": "models/transformer_model.pth",
    "anti_hallucination_path": "models/anti_hallucination.pkl",
    "device": "auto"
  },
  "ui": {
    "theme": "light",
    "update_frequency": 1,
    "port": 8501
  },
  "notifications": {
    "sms_enabled": false,
    "email_enabled": false,
    "push_enabled": true,
    "phone_numbers": [],
    "email_addresses": []
  },
  "alerts": {
    "thresholds": {
      "normal": [0, 30],
      "mild": [31, 50],
      "elevated": [51, 85],
      "critical": [86, 100]
    }
  }
}
```

#### `config/user_config.json`
```json
{
  "first_run": false,
  "setup_completed": true,
  "user_preferences": {
    "theme": "light",
    "notifications": {
      "enabled": false
    }
  }
}
```

## Starting the Application

### Using Startup Scripts

**Windows:**
```cmd
start_safeguard.bat
```

**macOS/Linux:**
```bash
./start_safeguard.sh
```

### Manual Start

```bash
# Direct Python execution
python main.py

# Or using Streamlit directly
streamlit run app.py --server.port=8501
```

### Accessing the Interface

1. **Open your web browser**
2. **Navigate to:** `http://localhost:8501`
3. **Complete any remaining setup** if prompted

## Updates and Maintenance

### Automatic Updates

The update system can check for and install updates automatically:

```bash
# Check for updates
python update_manager.py --check

# Interactive update
python update_manager.py --update

# Configure auto-updates
python update_manager.py --configure
```

### Manual Updates

1. **Download new version** from the official source
2. **Backup current installation:**
   ```bash
   cp -r . ../safeguard_backup_$(date +%Y%m%d)
   ```
3. **Extract and install** new version
4. **Migrate configuration** if needed

### Backup and Restore

**Create Backup:**
```bash
# Full backup
tar -czf safeguard_backup_$(date +%Y%m%d).tar.gz \
  config/ models/ data/ logs/ main.py app.py saafe_mvp/

# Configuration only
cp -r config/ config_backup_$(date +%Y%m%d)/
```

**Restore Backup:**
```bash
# Extract full backup
tar -xzf safeguard_backup_YYYYMMDD.tar.gz

# Restore configuration
cp -r config_backup_YYYYMMDD/* config/
```

## Troubleshooting

### Common Installation Issues

#### Python Not Found
**Error:** `python: command not found`

**Solutions:**
- Install Python from [python.org](https://python.org)
- Use `python3` instead of `python` on some systems
- Add Python to your system PATH

#### Permission Denied
**Error:** `Permission denied` when running scripts

**Solutions:**
```bash
# Make scripts executable
chmod +x start_safeguard.sh
chmod +x installer.py

# Or run with python explicitly
python installer.py
```

#### Dependency Installation Failed
**Error:** `pip install` fails

**Solutions:**
```bash
# Update pip
python -m pip install --upgrade pip

# Install with user flag
pip install --user -r requirements.txt

# Use virtual environment
python -m venv safeguard_env
source safeguard_env/bin/activate  # Linux/macOS
# or
safeguard_env\Scripts\activate     # Windows
pip install -r requirements.txt
```

#### Port Already in Use
**Error:** `Port 8501 is already in use`

**Solutions:**
- Change port in `config/app_config.json`
- Kill process using the port: `lsof -ti:8501 | xargs kill`
- Use different port: `streamlit run app.py --server.port=8502`

### Application Issues

#### Models Not Loading
**Symptoms:** Error messages about missing models

**Solutions:**
1. **Re-run installer:** `python installer.py`
2. **Check model files:** Ensure files exist in `models/` directory
3. **Download models manually** if needed
4. **Check file permissions**

#### Configuration Errors
**Symptoms:** Application won't start or behaves unexpectedly

**Solutions:**
1. **Reset configuration:** Delete `config/` directory and re-run setup
2. **Run setup wizard:** `python setup_wizard.py`
3. **Check JSON syntax** in configuration files
4. **Restore from backup** if available

#### Performance Issues
**Symptoms:** Slow response, high CPU usage

**Solutions:**
1. **Increase update frequency** in settings (2 seconds instead of 0.5)
2. **Enable GPU acceleration** if available
3. **Close other applications** to free resources
4. **Check system requirements**

### Getting Help

#### Log Files

Check log files for detailed error information:
- `logs/safeguard_errors.log` - Application errors
- `installation_report.txt` - Installation details
- `build_test_report.txt` - Build system test results

#### Debug Mode

Enable debug mode for more detailed logging:
1. **Edit** `config/app_config.json`
2. **Set** `"debug": true`
3. **Restart** the application

#### Support Resources

- **Documentation:** Check all `.md` files in the project
- **Configuration:** Review `config/` directory contents
- **Test Scripts:** Run `python test_build.py` for system validation

## Uninstallation

### Complete Removal

```bash
# Stop the application
# Kill any running processes

# Remove application files
rm -rf saafe_mvp/
rm -rf config/
rm -rf models/
rm -rf data/
rm -rf logs/
rm -rf exports/
rm main.py app.py
rm start_safeguard.*
rm *.py

# Remove Python packages (optional)
pip uninstall -r requirements.txt
```

### Keep User Data

```bash
# Remove application but keep configuration and data
rm -rf saafe_mvp/
rm main.py app.py
rm start_safeguard.*
rm *.py

# Keep: config/, models/, data/, logs/, exports/
```

## Advanced Installation

### Docker Installation

```dockerfile
# Dockerfile (example)
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python installer.py --non-interactive

EXPOSE 8501
CMD ["python", "main.py"]
```

### Virtual Environment

```bash
# Create virtual environment
python -m venv safeguard_env

# Activate (Linux/macOS)
source safeguard_env/bin/activate

# Activate (Windows)
safeguard_env\Scripts\activate

# Install in virtual environment
python installer.py

# Deactivate when done
deactivate
```

### System Service (Linux)

Create a systemd service for automatic startup:

```ini
# /etc/systemd/system/saafe-mvp.service
[Unit]
Description=Saafe Fire Detection MVP
After=network.target

[Service]
Type=simple
User=saafe
WorkingDirectory=/opt/saafe-mvp
ExecStart=/usr/bin/python3 main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable saafe-mvp
sudo systemctl start saafe-mvp
```

---

For additional support or questions, refer to the project documentation or contact the development team.