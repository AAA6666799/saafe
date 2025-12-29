# Saafe Fire Detection MVP - Technical Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [System Components](#system-components)
3. [AI/ML Pipeline](#aiml-pipeline)
4. [Data Models](#data-models)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Development Setup](#development-setup)
8. [Testing](#testing)
9. [Deployment](#deployment)
10. [Performance Optimization](#performance-optimization)
11. [Security Considerations](#security-considerations)
12. [Maintenance](#maintenance)

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Saafe MVP Application                │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Streamlit Web Interface)                        │
│  ├── Dashboard Components                                   │
│  ├── Scenario Controls                                      │
│  ├── Real-time Charts                                       │
│  └── Configuration Panels                                   │
├─────────────────────────────────────────────────────────────┤
│  Core Engine                                                │
│  ├── Scenario Manager                                       │
│  ├── Data Generator                                         │
│  ├── AI Model Pipeline                                      │
│  └── Alert Engine                                           │
├─────────────────────────────────────────────────────────────┤
│  AI/ML Layer                                                │
│  ├── Spatio-Temporal Transformer                           │
│  ├── Anti-Hallucination Logic                              │
│  ├── Ensemble Voting                                        │
│  └── Model Management                                       │
├─────────────────────────────────────────────────────────────┤
│  Notification Services                                      │
│  ├── SMS Gateway Integration                                │
│  ├── Email Service                                          │
│  ├── Push Notification Service                             │
│  └── Alert History                                          │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ├── Model Storage                                          │
│  ├── Configuration Storage                                  │
│  ├── Session Logging                                        │
│  └── Export Services                                        │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack
- **Frontend**: Streamlit (Python web framework)
- **AI/ML**: PyTorch with custom Transformer models
- **Data Processing**: NumPy, Pandas
- **Visualization**: Plotly, Matplotlib
- **Notifications**: Twilio (SMS), SMTP (Email), WebPush
- **Storage**: SQLite, JSON, CSV
- **Packaging**: PyInstaller
- **Testing**: pytest, unittest

### Design Principles
1. **Modularity**: Clear separation of concerns with well-defined interfaces
2. **Reliability**: Comprehensive error handling and fallback mechanisms
3. **Performance**: Optimized for real-time processing with <50ms inference
4. **Scalability**: Designed for future hardware integration
5. **Maintainability**: Clean code with extensive documentation and testing

## System Components

### Core Components

#### 1. Model Manager (`saafe_mvp/models/model_manager.py`)
**Purpose**: Manages AI model loading, validation, and fallback mechanisms

**Key Classes**:
- `ModelManager`: Main model management interface
- `ModelRegistry`: Registry for multiple models
- `ModelMetadata`: Model information and statistics

**Key Methods**:
```python
def load_model(self, model_path: str) -> Tuple[bool, str]
def create_fallback_model(self) -> Tuple[bool, str]
def get_model(self, model_id: str = None) -> Optional[SpatioTemporalTransformer]
def get_system_status(self) -> Dict[str, Any]
```

#### 2. Fire Detection Pipeline (`saafe_mvp/core/fire_detection_pipeline.py`)
**Purpose**: Orchestrates data preprocessing, model inference, and result formatting

**Key Classes**:
- `FireDetectionPipeline`: Main inference pipeline
- `DataPreprocessor`: Sensor data preprocessing
- `PredictionResult`: Structured prediction output

**Key Methods**:
```python
def predict(self, sensor_readings: List[SensorReading]) -> PredictionResult
def predict_batch(self, batch_readings: List[List[SensorReading]]) -> List[PredictionResult]
def get_performance_metrics(self) -> Dict[str, Any]
```

#### 3. Scenario Manager (`saafe_mvp/core/scenario_manager.py`)
**Purpose**: Manages different environmental scenarios and data generation

**Key Classes**:
- `ScenarioManager`: Coordinates scenario execution
- `ScenarioType`: Enumeration of available scenarios

**Key Methods**:
```python
def start_scenario(self, scenario_type: ScenarioType) -> bool
def stop_scenario(self) -> None
def get_current_data(self) -> Optional[SensorReading]
```

#### 4. Alert Engine (`saafe_mvp/core/alert_engine.py`)
**Purpose**: Converts risk scores to alerts and manages alert history

**Key Classes**:
- `AlertEngine`: Main alert processing system
- `AlertLevel`: Alert severity enumeration
- `AlertData`: Structured alert information

**Key Methods**:
```python
def process_prediction(self, prediction_result: PredictionResult) -> AlertData
def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]
def update_thresholds(self, new_thresholds: AlertThresholds)
```

### Service Components

#### 1. Notification Manager (`saafe_mvp/services/notification_manager.py`)
**Purpose**: Unified interface for all notification channels

**Key Features**:
- Multi-channel notification delivery
- Configurable alert level thresholds
- Delivery status tracking
- Fallback mechanisms

#### 2. Export Service (`saafe_mvp/services/export_service.py`)
**Purpose**: Data export and report generation

**Supported Formats**:
- CSV: Raw sensor data
- JSON: Structured session data
- PDF: Professional reports (when ReportLab available)

#### 3. Session Manager (`saafe_mvp/services/session_manager.py`)
**Purpose**: Manages data collection sessions and coordinates exports

**Key Features**:
- Session lifecycle management
- Real-time data collection
- Performance monitoring integration
- Export coordination

### AI/ML Components

#### 1. Spatio-Temporal Transformer (`saafe_mvp/models/transformer.py`)
**Purpose**: Core AI model for fire detection

**Architecture**:
- Multi-head attention mechanism
- Temporal sequence processing
- Spatial sensor correlation
- Risk score regression output

#### 2. Anti-Hallucination Engine (`saafe_mvp/models/anti_hallucination.py`)
**Purpose**: Prevents false alarms through ensemble voting and pattern recognition

**Key Features**:
- Cooking pattern detection
- Ensemble model voting
- Conservative risk assessment
- Confidence adjustment

## AI/ML Pipeline

### Data Flow

```
Sensor Readings → Preprocessing → Model Inference → Anti-Hallucination → Alert Generation
      ↓               ↓              ↓                    ↓                  ↓
  Raw Values    Normalized      Risk Score         Validated Score      Alert Level
                Tensors                           + Confidence
```

### Model Architecture

#### Spatio-Temporal Transformer
```python
class SpatioTemporalTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        # Input embedding
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(config.d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, 1)  # Risk score
```

#### Anti-Hallucination Logic
1. **Ensemble Voting**: Multiple models vote on predictions
2. **Cooking Detection**: Pattern recognition for cooking activities
3. **Conservative Assessment**: Applies safety margins to predictions
4. **Confidence Adjustment**: Modifies confidence based on validation

### Training Data Requirements
- **Normal Conditions**: Stable sensor readings with natural variations
- **Cooking Scenarios**: Elevated PM2.5/CO₂ without temperature spikes
- **Fire Events**: Rapid temperature increases with multiple fire indicators
- **Temporal Patterns**: Time-series data with realistic sensor correlations

## Data Models

### Core Data Structures

#### SensorReading
```python
@dataclass
class SensorReading:
    timestamp: datetime
    temperature: float      # Celsius
    pm25: float            # μg/m³
    co2: float             # ppm
    audio_level: float     # dB
    location: str          # Sensor location
```

#### PredictionResult
```python
@dataclass
class PredictionResult:
    risk_score: float                    # 0-100 risk score
    confidence: float                    # Model confidence 0-1
    predicted_class: str                 # 'normal', 'cooking', 'fire'
    feature_importance: Dict[str, float] # Feature contributions
    processing_time: float               # Inference time in ms
    ensemble_votes: Dict[str, float]     # Individual model votes
    anti_hallucination: ValidationResult # Validation result
    timestamp: datetime                  # Prediction timestamp
    model_metadata: Dict[str, Any]       # Model information
```

#### AlertData
```python
@dataclass
class AlertData:
    alert_level: AlertLevel
    risk_score: float
    confidence: float
    message: str
    timestamp: datetime
    sensor_readings: Optional[SensorReading]
    prediction_result: Optional[PredictionResult]
    context_info: Dict[str, Any]
    alert_id: str
```

### Configuration Models

#### ModelConfig
```python
@dataclass
class ModelConfig:
    input_dim: int = 4          # Number of sensor features
    d_model: int = 256          # Model dimension
    num_heads: int = 8          # Attention heads
    num_layers: int = 6         # Transformer layers
    dropout: float = 0.1        # Dropout rate
    max_sequence_length: int = 60  # Maximum sequence length
```

#### NotificationConfig
```python
@dataclass
class NotificationConfig:
    sms_enabled: bool = True
    email_enabled: bool = True
    push_enabled: bool = True
    phone_numbers: List[str]
    email_addresses: List[str]
    sms_min_level: AlertLevel = AlertLevel.ELEVATED
    email_min_level: AlertLevel = AlertLevel.MILD
    push_min_level: AlertLevel = AlertLevel.MILD
```

## API Reference

### Model Manager API

#### Loading Models
```python
# Load a specific model
success, message = model_manager.load_model("models/transformer_model.pth")

# Create fallback model
success, message = model_manager.create_fallback_model()

# Get model for inference
model = model_manager.get_model()
```

#### Model Information
```python
# Get system status
status = model_manager.get_system_status()

# Get model metadata
metadata = model_manager.registry.get_metadata("model_id")
```

### Fire Detection Pipeline API

#### Making Predictions
```python
# Single prediction
result = pipeline.predict([sensor_reading])

# Batch predictions
results = pipeline.predict_batch([
    [reading1, reading2],
    [reading3, reading4]
])
```

#### Performance Monitoring
```python
# Get performance metrics
metrics = pipeline.get_performance_metrics()

# Reset performance statistics
pipeline.reset_performance_stats()
```

### Scenario Manager API

#### Scenario Control
```python
# Start a scenario
success = scenario_manager.start_scenario(ScenarioType.NORMAL)

# Get current data
reading = scenario_manager.get_current_data()

# Stop scenario
scenario_manager.stop_scenario()
```

#### Scenario Information
```python
# Get scenario info
info = scenario_manager.get_scenario_info()

# Check if running
is_running = scenario_manager.is_running()
```

### Alert Engine API

#### Processing Alerts
```python
# Process prediction result
alert = alert_engine.process_prediction(prediction_result, sensor_reading)

# Get alert history
history = alert_engine.get_alert_history(hours=24)
```

#### Configuration
```python
# Update thresholds
new_thresholds = AlertThresholds(normal_max=25.0, mild_max=45.0)
alert_engine.update_thresholds(new_thresholds)
```

### Notification Manager API

#### Sending Notifications
```python
# Send alert notification
result = notification_manager.send_alert(AlertLevel.CRITICAL, risk_score=95)

# Send test notifications
result = notification_manager.send_test_notifications()
```

#### Configuration Management
```python
# Add contacts
notification_manager.add_phone_number("+1234567890")
notification_manager.add_email_address("user@example.com")

# Update preferences
notification_manager.update_notification_preferences(
    sms_enabled=True,
    email_min_level=AlertLevel.MILD
)
```

## Configuration

### Application Configuration

#### config/app_config.json
```json
{
  "model_settings": {
    "model_path": "models/transformer_model.pth",
    "fallback_enabled": true,
    "gpu_enabled": true,
    "batch_size": 1
  },
  "alert_settings": {
    "normal_threshold": 30.0,
    "mild_threshold": 50.0,
    "elevated_threshold": 85.0,
    "hysteresis_margin": 5.0
  },
  "performance_settings": {
    "update_frequency": 1.0,
    "max_processing_time": 100.0,
    "memory_limit_mb": 2000
  }
}
```

#### config/user_config.json
```json
{
  "ui_settings": {
    "theme": "dark",
    "update_frequency": 1,
    "show_advanced": false
  },
  "notification_settings": {
    "sms_enabled": true,
    "email_enabled": true,
    "push_enabled": true,
    "phone_numbers": [],
    "email_addresses": []
  },
  "export_settings": {
    "auto_export": false,
    "export_format": "json",
    "export_interval": 3600
  }
}
```

### Environment Variables
```bash
# Model configuration
SAFEGUARD_MODEL_PATH=/path/to/models
SAFEGUARD_GPU_ENABLED=true

# Notification services
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Performance settings
SAFEGUARD_MAX_MEMORY=2048
SAFEGUARD_LOG_LEVEL=INFO
```

## Development Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation Steps
```bash
# Clone repository
git clone <repository_url>
cd saafe-mvp

# Create virtual environment
python -m venv safeguard_env
source safeguard_env/bin/activate  # Linux/Mac
# or
safeguard_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r build_requirements.txt  # For building

# Install development dependencies
pip install -r dev_requirements.txt
```

### Development Dependencies
```txt
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=0.991
pre-commit>=2.20.0
sphinx>=5.0.0
```

### Code Style and Linting
```bash
# Format code
black saafe_mvp/

# Lint code
flake8 saafe_mvp/

# Type checking
mypy saafe_mvp/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Project Structure
```
saafe-mvp/
├── saafe_mvp/           # Main application package
│   ├── core/                # Core business logic
│   ├── models/              # AI/ML models
│   ├── services/            # External services
│   ├── ui/                  # User interface components
│   └── utils/               # Utility functions
├── tests/                   # Test suite
├── docs/                    # Documentation
├── config/                  # Configuration files
├── models/                  # Pre-trained models
├── data/                    # Data files
├── exports/                 # Export output
└── logs/                    # Log files
```

## Testing

### Test Structure
```
tests/
├── unit/                    # Unit tests
│   ├── test_models/
│   ├── test_core/
│   ├── test_services/
│   └── test_utils/
├── integration/             # Integration tests
├── performance/             # Performance tests
└── fixtures/                # Test data and fixtures
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=saafe_mvp --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run specific test file
pytest tests/unit/test_models/test_model_manager.py

# Run with verbose output
pytest -v

# Run integration test
python test_system_integration_simple.py
```

### Test Configuration
```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow running tests
```

### Writing Tests
```python
# Example unit test
import pytest
from saafe_mvp.models.model_manager import ModelManager

class TestModelManager:
    def test_initialization(self):
        manager = ModelManager()
        assert manager is not None
        assert manager.device is not None
    
    def test_fallback_model_creation(self):
        manager = ModelManager()
        success, message = manager.create_fallback_model()
        assert success is True
        assert "fallback" in message.lower()
    
    @pytest.mark.integration
    def test_model_inference(self):
        manager = ModelManager()
        manager.create_fallback_model()
        model = manager.get_model()
        assert model is not None
```

## Deployment

### Building Executables

#### Using PyInstaller
```bash
# Install PyInstaller
pip install pyinstaller

# Build executable
python build.py

# Manual build (alternative)
pyinstaller --onefile --windowed --name saafe-mvp app.py
```

#### Build Configuration (build_config.py)
```python
BUILD_CONFIG = {
    'name': 'saafe-mvp',
    'version': '1.0.0',
    'description': 'Saafe Fire Detection MVP',
    'author': 'Saafe Team',
    'icon': 'assets/icon.ico',
    'hidden_imports': [
        'torch',
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ],
    'data_files': [
        ('models', 'models'),
        ('config', 'config'),
        ('assets', 'assets')
    ]
}
```

### Platform-Specific Builds

#### Windows
```bash
# Build Windows executable
python build.py --platform windows
# Output: dist/saafe-mvp-windows.exe
```

#### macOS
```bash
# Build macOS application
python build.py --platform macos
# Output: dist/saafe-mvp-macos.app
```

#### Linux
```bash
# Build Linux binary
python build.py --platform linux
# Output: dist/saafe-mvp-linux
```

### Distribution

#### Creating Installers
```bash
# Windows (using NSIS)
makensis installer.nsi

# macOS (using create-dmg)
create-dmg --volname "Saafe MVP" --window-size 600 400 \
  saafe-mvp-installer.dmg dist/saafe-mvp-macos.app

# Linux (using AppImage)
python create_appimage.py
```

#### Package Structure
```
saafe-mvp-package/
├── saafe-mvp(.exe)      # Main executable
├── models/                  # AI models
├── config/                  # Configuration files
├── assets/                  # Icons and resources
├── README.txt               # Quick start guide
└── LICENSE.txt              # License information
```

## Performance Optimization

### Model Optimization

#### GPU Acceleration
```python
# Enable GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimize for inference
model.eval()
torch.set_grad_enabled(False)
```

#### Model Quantization
```python
# Post-training quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

#### TensorRT Optimization (NVIDIA GPUs)
```python
# Convert to TensorRT (if available)
import torch_tensorrt

trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch.randn(1, 60, 4, 4).cuda()],
    enabled_precisions={torch.float, torch.half}
)
```

### Memory Optimization

#### Batch Processing
```python
# Process multiple readings efficiently
def predict_batch(self, readings_batch: List[List[SensorReading]]):
    # Batch preprocessing
    tensors = [self.preprocess(readings) for readings in readings_batch]
    batch_tensor = torch.stack(tensors)
    
    # Batch inference
    with torch.no_grad():
        results = self.model(batch_tensor)
    
    return results
```

#### Memory Management
```python
# Clear GPU cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use memory-efficient data structures
from collections import deque
self.reading_buffer = deque(maxlen=1000)  # Fixed-size buffer
```

### Performance Monitoring

#### Metrics Collection
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'inference_times': deque(maxlen=1000),
            'memory_usage': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100)
        }
    
    def record_inference_time(self, time_ms: float):
        self.metrics['inference_times'].append(time_ms)
    
    def get_average_inference_time(self) -> float:
        times = list(self.metrics['inference_times'])
        return sum(times) / len(times) if times else 0.0
```

#### Performance Targets
- **Inference Time**: <50ms per prediction
- **Memory Usage**: <2GB RAM
- **CPU Usage**: <50% on recommended hardware
- **Startup Time**: <10 seconds
- **Response Time**: <100ms for UI interactions

## Security Considerations

### Data Security

#### Local Data Storage
- All sensor data processed locally
- No external data transmission (except notifications)
- Configurable data retention policies
- Secure deletion of sensitive data

#### Model Security
```python
# Model integrity verification
def verify_model_integrity(model_path: str) -> bool:
    expected_hash = load_expected_hash()
    actual_hash = calculate_file_hash(model_path)
    return expected_hash == actual_hash
```

### Network Security

#### Notification Security
```python
# Secure SMTP configuration
smtp_config = {
    'server': 'smtp.gmail.com',
    'port': 587,
    'use_tls': True,
    'username': os.getenv('SMTP_USERNAME'),
    'password': os.getenv('SMTP_PASSWORD')  # Use app passwords
}

# SMS security (Twilio)
twilio_config = {
    'account_sid': os.getenv('TWILIO_ACCOUNT_SID'),
    'auth_token': os.getenv('TWILIO_AUTH_TOKEN'),
    'verify_ssl': True
}
```

#### Input Validation
```python
def validate_sensor_reading(reading: SensorReading) -> bool:
    # Range validation
    if not (0 <= reading.temperature <= 100):
        return False
    if not (0 <= reading.pm25 <= 1000):
        return False
    if not (0 <= reading.co2 <= 5000):
        return False
    if not (0 <= reading.audio_level <= 120):
        return False
    
    return True
```

### Application Security

#### Error Handling
```python
# Secure error handling
try:
    result = process_sensitive_data(data)
except Exception as e:
    # Log error without sensitive data
    logger.error(f"Processing failed: {type(e).__name__}")
    # Return safe error message
    return {"error": "Processing failed", "code": "PROC_001"}
```

#### Configuration Security
```python
# Secure configuration loading
def load_secure_config():
    config_path = get_secure_config_path()
    if not os.path.exists(config_path):
        create_default_config(config_path)
    
    # Validate configuration
    config = load_config(config_path)
    validate_config(config)
    
    return config
```

## Maintenance

### Logging

#### Log Configuration
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'logs/saafe.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
```

#### Log Categories
- **Application**: General application events
- **Model**: AI model operations and performance
- **Alerts**: Alert generation and notification events
- **Performance**: System performance metrics
- **Errors**: Error conditions and exceptions
- **Security**: Security-related events

### Health Monitoring

#### System Health Checks
```python
def perform_health_check() -> Dict[str, Any]:
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'healthy',
        'components': {}
    }
    
    # Check model availability
    health_status['components']['models'] = check_model_health()
    
    # Check memory usage
    health_status['components']['memory'] = check_memory_health()
    
    # Check disk space
    health_status['components']['disk'] = check_disk_health()
    
    # Check notification services
    health_status['components']['notifications'] = check_notification_health()
    
    return health_status
```

#### Automated Diagnostics
```python
def run_diagnostics() -> Dict[str, Any]:
    diagnostics = {
        'system_info': get_system_info(),
        'performance_metrics': get_performance_metrics(),
        'error_summary': get_error_summary(),
        'configuration_status': validate_configuration()
    }
    
    return diagnostics
```

### Updates and Versioning

#### Version Management
```python
# version.py
VERSION = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'build': '20241201',
    'string': '1.0.0'
}

def get_version_string() -> str:
    return f"{VERSION['major']}.{VERSION['minor']}.{VERSION['patch']}"
```

#### Update Mechanism
```python
class UpdateManager:
    def check_for_updates(self) -> Optional[Dict[str, Any]]:
        # Check for available updates
        pass
    
    def download_update(self, update_info: Dict[str, Any]) -> bool:
        # Download and verify update
        pass
    
    def apply_update(self) -> bool:
        # Apply update with rollback capability
        pass
```

### Backup and Recovery

#### Configuration Backup
```python
def backup_configuration():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"backups/config_backup_{timestamp}.json"
    
    config = load_current_config()
    save_config(config, backup_path)
    
    return backup_path
```

#### Data Recovery
```python
def recover_from_backup(backup_path: str) -> bool:
    try:
        backup_config = load_config(backup_path)
        validate_config(backup_config)
        apply_config(backup_config)
        return True
    except Exception as e:
        logger.error(f"Recovery failed: {e}")
        return False
```

---

## Appendices

### A. Error Codes
- **INIT_001**: Application initialization failed
- **MODEL_001**: Model loading failed
- **MODEL_002**: Model inference failed
- **DATA_001**: Invalid sensor data
- **ALERT_001**: Alert generation failed
- **NOTIF_001**: Notification delivery failed
- **EXPORT_001**: Data export failed

### B. Performance Benchmarks
- **Inference Time**: 15-45ms (typical)
- **Memory Usage**: 200-800MB (typical)
- **Startup Time**: 3-8 seconds
- **CPU Usage**: 5-25% (idle to active)

### C. Supported Platforms
- **Windows**: 10, 11 (x64)
- **macOS**: 10.14+ (Intel/Apple Silicon)
- **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 10+

---

*Saafe Fire Detection MVP - Technical Documentation*

**Version**: 1.0.0  
15 August 2025 
**Document Version**: 1.0