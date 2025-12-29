"""
Core Interfaces and Abstract Base Classes

Defines the fundamental interfaces that establish system boundaries
and ensure consistent implementation across all components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SensorData:
    """Core data structure for sensor readings"""
    timestamp: float
    thermal_frame: Optional[np.ndarray] = None  # 384x288 thermal image
    gas_readings: Optional[Dict[str, float]] = None  # gas_type -> concentration
    environmental_data: Optional[Dict[str, float]] = None  # parameter -> value
    sensor_health: Optional[Dict[str, float]] = None  # sensor_id -> health_score


@dataclass
class FeatureVector:
    """Extracted features from sensor data"""
    timestamp: float
    thermal_features: Optional[np.ndarray] = None  # 8 thermal features
    gas_features: Optional[np.ndarray] = None  # 5 gas features
    environmental_features: Optional[np.ndarray] = None  # 4 environmental features
    fusion_features: Optional[np.ndarray] = None  # 3 fusion features
    feature_quality: float = 1.0


@dataclass
class PredictionResult:
    """Model prediction output"""
    timestamp: float
    fire_probability: float
    confidence_score: float
    lead_time_estimate: float
    contributing_factors: Dict[str, float]
    model_ensemble_votes: Dict[str, float]


@dataclass
class RiskAssessment:
    """Risk assessment from analysis agent"""
    timestamp: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    fire_probability: float
    confidence_level: float
    contributing_sensors: List[str]
    recommended_actions: List[str]
    escalation_required: bool


class DataGenerator(ABC):
    """Abstract base class for all data generators"""
    
    @abstractmethod
    def generate(self, scenario_params: Dict[str, Any], timestamp: float) -> Any:
        """Generate data based on scenario parameters"""
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate scenario parameters"""
        pass


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors"""
    
    @abstractmethod
    def extract_features(self, sensor_data: SensorData) -> np.ndarray:
        """Extract features from sensor data"""
        pass
    
    @abstractmethod
    def validate_features(self, features: np.ndarray) -> bool:
        """Validate extracted features"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features"""
        pass


class Model(ABC):
    """Abstract base class for all prediction models"""
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Make prediction and return (probability, confidence)"""
        pass
    
    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk"""
        pass


class Agent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.state = {}
        self.message_queue = []
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process incoming data"""
        pass
    
    @abstractmethod
    def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle inter-agent messages"""
        pass
    
    def send_message(self, recipient: str, message: Dict[str, Any]) -> None:
        """Send message to another agent"""
        # Implementation will be handled by AgentCoordinator
        pass


class HardwareInterface(ABC):
    """Abstract base class for hardware interfaces"""
    
    @abstractmethod
    def get_sensor_data(self) -> SensorData:
        """Get current sensor readings"""
        pass
    
    @abstractmethod
    def calibrate_sensors(self, calibration_params: Dict[str, Any]) -> None:
        """Calibrate sensors"""
        pass
    
    @abstractmethod
    def validate_sensor_health(self) -> Dict[str, float]:
        """Check sensor health status"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if hardware is connected"""
        pass


class SystemComponent(ABC):
    """Abstract base class for system components"""
    
    def __init__(self, component_id: str, config: Dict[str, Any]):
        self.component_id = component_id
        self.config = config
        self.is_running = False
    
    @abstractmethod
    def start(self) -> None:
        """Start the component"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the component"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get component status"""
        pass