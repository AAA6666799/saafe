"""
Core data models for the Saafe MVP system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np
import uuid


@dataclass
class SensorReading:
    """Individual sensor reading with timestamp and location."""
    timestamp: datetime
    temperature: float  # Celsius
    pm25: float        # μg/m³ (particulate matter)
    co2: float         # ppm (parts per million)
    audio_level: float # dB (decibels)
    location: str = "default"  # Sensor location identifier
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'temperature': self.temperature,
            'pm25': self.pm25,
            'co2': self.co2,
            'audio_level': self.audio_level,
            'location': self.location
        }
    
    def to_array(self) -> np.ndarray:
        """Convert sensor values to numpy array for model input."""
        return np.array([
            self.temperature,
            self.pm25,
            self.co2,
            self.audio_level
        ])


@dataclass
class ScenarioConfig:
    """Configuration for scenario generation."""
    duration_seconds: int = 300  # 5 minutes default
    update_frequency: float = 1.0  # Updates per second
    noise_level: float = 0.1  # Noise intensity (0-1)
    location: str = "demo_location"
    
    @property
    def total_samples(self) -> int:
        """Calculate total number of samples for the scenario."""
        return int(self.duration_seconds * self.update_frequency)


@dataclass
class SensorLimits:
    """Sensor value limits and normal ranges."""
    temperature_min: float = -10.0
    temperature_max: float = 100.0
    temperature_normal: tuple = (18.0, 25.0)
    
    pm25_min: float = 0.0
    pm25_max: float = 500.0
    pm25_normal: tuple = (5.0, 25.0)
    
    co2_min: float = 300.0
    co2_max: float = 5000.0
    co2_normal: tuple = (400.0, 600.0)
    
    audio_min: float = 20.0
    audio_max: float = 120.0
    audio_normal: tuple = (30.0, 50.0)
    
    def validate_reading(self, reading: SensorReading) -> bool:
        """Validate that sensor reading is within acceptable limits."""
        return (
            self.temperature_min <= reading.temperature <= self.temperature_max and
            self.pm25_min <= reading.pm25 <= self.pm25_max and
            self.co2_min <= reading.co2 <= self.co2_max and
            self.audio_min <= reading.audio_level <= self.audio_max
        )
    
    def clamp_reading(self, reading: SensorReading) -> SensorReading:
        """Clamp sensor values to valid ranges."""
        return SensorReading(
            timestamp=reading.timestamp,
            temperature=np.clip(reading.temperature, self.temperature_min, self.temperature_max),
            pm25=np.clip(reading.pm25, self.pm25_min, self.pm25_max),
            co2=np.clip(reading.co2, self.co2_min, self.co2_max),
            audio_level=np.clip(reading.audio_level, self.audio_min, self.audio_max),
            location=reading.location
        )


class AlertLevel(Enum):
    """Alert level enumeration with numeric values."""
    NORMAL = (1, "Normal", "🟢")
    MILD = (2, "Mild Anomaly", "🟡")
    ELEVATED = (3, "Elevated Risk", "🟠")
    CRITICAL = (4, "CRITICAL FIRE ALERT", "🔴")
    
    def __init__(self, level: int, description: str, icon: str):
        self.level = level
        self.description = description
        self.icon = icon
    
    @classmethod
    def from_risk_score(cls, risk_score: float) -> 'AlertLevel':
        """Convert risk score to alert level."""
        if risk_score <= 30:
            return cls.NORMAL
        elif risk_score <= 50:
            return cls.MILD
        elif risk_score <= 85:
            return cls.ELEVATED
        else:
            return cls.CRITICAL


@dataclass
class ValidationResult:
    """Anti-hallucination validation result."""
    is_valid: bool
    confidence: float
    reason: str = ""


@dataclass
class PredictionResult:
    """AI model prediction result with comprehensive information."""
    risk_score: float           # 0-100 risk score
    confidence: float           # Model confidence 0-1
    predicted_class: str        # 'normal', 'cooking', 'fire'
    feature_importance: Dict[str, float] = field(default_factory=dict)  # Feature contributions
    processing_time: float = 0.0      # Inference time in ms
    ensemble_votes: Dict[str, float] = field(default_factory=dict)  # Individual model votes
    anti_hallucination: Optional[ValidationResult] = None  # Anti-hallucination validation
    timestamp: datetime = field(default_factory=datetime.now)         # When prediction was made
    model_metadata: Dict[str, Any] = field(default_factory=dict)  # Model information


@dataclass
class AlertData:
    """Comprehensive alert information."""
    alert_level: AlertLevel
    risk_score: float
    confidence: float
    message: str
    timestamp: datetime
    sensor_readings: Optional[SensorReading] = None
    prediction_result: Optional[PredictionResult] = None
    context_info: Dict[str, Any] = field(default_factory=dict)
    alert_id: str = ""
    
    def __post_init__(self):
        """Generate alert ID if not provided."""
        if not self.alert_id:
            self.alert_id = f"alert_{int(self.timestamp.timestamp() * 1000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'alert_level': {
                'level': self.alert_level.level,
                'description': self.alert_level.description,
                'icon': self.alert_level.icon
            },
            'risk_score': self.risk_score,
            'confidence': self.confidence,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'context_info': self.context_info,
            'sensor_data': self.sensor_readings.to_dict() if self.sensor_readings else None
        }