"""
Fire Classification Agent for the synthetic fire prediction system.

This module implements the FireClassificationAgent class, which is responsible for
classifying detected fires based on sensor data and model predictions.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from ..base import Agent, Message
from ...ml.models.classification import FireClassificationModel  # Assuming this exists

# Configure logging
logger = logging.getLogger(__name__)


class FireClassificationAgent(Agent):
    """
    Agent responsible for classifying detected fires.
    
    This agent analyzes sensor data and model predictions to determine
    the type and characteristics of detected fires.
    """
    
    # Define fire types
    FIRE_TYPES = {
        0: "electrical",
        1: "chemical",
        2: "smoldering",
        3: "rapid_combustion",
        4: "unknown"
    }
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the fire classification agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Dictionary containing configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Initialize classification thresholds
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        
        # Initialize classification state
        self.classification_history = []
        self.last_classification = None
        self.last_classification_time = None
        
        # Load fire classification model
        self.model = self._load_model()
        
        # Register message handlers
        self.register_message_handler("update_thresholds", self._handle_update_thresholds)
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["model_path"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Validate threshold values
        if "confidence_threshold" in self.config:
            threshold = self.config["confidence_threshold"]
            if not 0 <= threshold <= 1:
                raise ValueError(f"Confidence threshold must be between 0 and 1, got {threshold}")
    
    def _load_model(self) -> FireClassificationModel:
        """
        Load the fire classification model.
        
        Returns:
            Loaded model
        """
        try:
            model_path = self.config["model_path"]
            # This is a placeholder - actual implementation would depend on the model framework
            model = FireClassificationModel.load(model_path)
            logger.info(f"Loaded fire classification model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading fire classification model: {e}")
            raise
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and classify detected fires.
        
        Args:
            data: Input data containing sensor readings, features, and fire detection results
            
        Returns:
            Dictionary containing classification results
        """
        try:
            # Check if fire is detected
            if not data.get("is_fire_detected", False):
                return {
                    "timestamp": datetime.now().isoformat(),
                    "fire_type": None,
                    "confidence": 0.0,
                    "severity": 0,
                    "characteristics": {}
                }
            
            # Extract features for the model
            features = self._extract_features(data)
            
            # Get model prediction
            fire_type_idx = self.model.predict(features)[0]
            confidences = self.model.predict_proba(features)[0]
            confidence = confidences[fire_type_idx]
            
            # Map to fire type
            fire_type = self.FIRE_TYPES.get(fire_type_idx, "unknown")
            
            # Determine severity (1-10 scale)
            severity = self._calculate_severity(data, fire_type)
            
            # Extract fire characteristics
            characteristics = self._extract_characteristics(data, fire_type)
            
            # Update classification history
            self._update_classification_history(fire_type, confidence, severity, characteristics)
            
            # Prepare results
            results = {
                "timestamp": datetime.now().isoformat(),
                "fire_type": fire_type,
                "confidence": float(confidence),
                "severity": severity,
                "characteristics": characteristics
            }
            
            # Log classification
            logger.info(f"Fire classified as {fire_type} with confidence {confidence:.2f}, severity {severity}")
            self.last_classification = fire_type
            self.last_classification_time = datetime.now()
            
            return results
        except Exception as e:
            logger.error(f"Error in fire classification processing: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "fire_type": "unknown",
                "confidence": 0.0,
                "severity": 0,
                "characteristics": {}
            }
    
    def _extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from input data for model prediction.
        
        Args:
            data: Input data containing sensor readings
            
        Returns:
            Feature array for model input
        """
        # This is a placeholder - actual implementation would depend on the data format
        features = []
        
        # Extract thermal features
        if "thermal" in data:
            thermal_data = data["thermal"]
            features.extend([
                thermal_data.get("max_temperature", 0),
                thermal_data.get("mean_temperature", 0),
                thermal_data.get("hotspot_area_percentage", 0),
                thermal_data.get("temperature_rise_slope", 0),
                thermal_data.get("thermal_entropy", 0)
            ])
        
        # Extract gas features
        if "gas" in data:
            gas_data = data["gas"]
            features.extend([
                gas_data.get("methane_ppm", 0),
                gas_data.get("propane_ppm", 0),
                gas_data.get("hydrogen_ppm", 0),
                gas_data.get("co_ppm", 0),
                gas_data.get("co2_ppm", 0),
                gas_data.get("concentration_slope", 0)
            ])
        
        # Extract environmental features
        if "environmental" in data:
            env_data = data["environmental"]
            features.extend([
                env_data.get("temperature", 0),
                env_data.get("humidity", 0),
                env_data.get("voc_level", 0)
            ])
        
        return np.array(features).reshape(1, -1)  # Reshape for single sample prediction
    
    def _calculate_severity(self, data: Dict[str, Any], fire_type: str) -> int:
        """
        Calculate the severity of the fire on a scale of 1-10.
        
        Args:
            data: Input data containing sensor readings
            fire_type: Type of fire
            
        Returns:
            Severity score (1-10)
        """
        severity = 1  # Default minimal severity
        
        # Base severity on temperature
        if "thermal" in data:
            max_temp = data["thermal"].get("max_temperature", 0)
            if max_temp > 500:  # Very high temperature
                severity += 4
            elif max_temp > 300:
                severity += 3
            elif max_temp > 200:
                severity += 2
            elif max_temp > 100:
                severity += 1
            
            # Consider temperature rise rate
            temp_rise = data["thermal"].get("temperature_rise_slope", 0)
            if temp_rise > 10:  # Fast rising temperature
                severity += 2
            elif temp_rise > 5:
                severity += 1
        
        # Consider gas concentrations
        if "gas" in data:
            gas_data = data["gas"]
            
            # Different gases contribute differently based on fire type
            if fire_type == "chemical":
                # Chemical fires are more severe with certain gases
                if gas_data.get("hydrogen_ppm", 0) > 1000:
                    severity += 2
                if gas_data.get("methane_ppm", 0) > 5000:
                    severity += 1
            
            # CO is dangerous in any fire
            if gas_data.get("co_ppm", 0) > 1000:
                severity += 2
            elif gas_data.get("co_ppm", 0) > 500:
                severity += 1
        
        # Consider area affected
        if "thermal" in data and data["thermal"].get("hotspot_area_percentage", 0) > 30:
            severity += 1
        
        # Cap at 10
        return min(severity, 10)
    
    def _extract_characteristics(self, data: Dict[str, Any], fire_type: str) -> Dict[str, Any]:
        """
        Extract characteristics of the fire.
        
        Args:
            data: Input data containing sensor readings
            fire_type: Type of fire
            
        Returns:
            Dictionary of fire characteristics
        """
        characteristics = {
            "growth_rate": "unknown",
            "smoke_production": "unknown",
            "heat_output": "unknown",
            "spread_pattern": "unknown"
        }
        
        # Determine growth rate
        if "thermal" in data:
            temp_rise = data["thermal"].get("temperature_rise_slope", 0)
            if temp_rise > 15:
                characteristics["growth_rate"] = "very_fast"
            elif temp_rise > 10:
                characteristics["growth_rate"] = "fast"
            elif temp_rise > 5:
                characteristics["growth_rate"] = "moderate"
            else:
                characteristics["growth_rate"] = "slow"
        
        # Determine smoke production based on fire type and gas readings
        if fire_type == "smoldering":
            characteristics["smoke_production"] = "high"
        elif fire_type == "electrical":
            characteristics["smoke_production"] = "moderate"
        elif "gas" in data and data["gas"].get("co_ppm", 0) > 500:
            characteristics["smoke_production"] = "high"
        else:
            characteristics["smoke_production"] = "low"
        
        # Determine heat output
        if "thermal" in data:
            max_temp = data["thermal"].get("max_temperature", 0)
            if max_temp > 400:
                characteristics["heat_output"] = "very_high"
            elif max_temp > 300:
                characteristics["heat_output"] = "high"
            elif max_temp > 200:
                characteristics["heat_output"] = "moderate"
            else:
                characteristics["heat_output"] = "low"
        
        # Determine spread pattern based on thermal distribution
        if "thermal" in data:
            hotspot_percentage = data["thermal"].get("hotspot_area_percentage", 0)
            entropy = data["thermal"].get("thermal_entropy", 0)
            
            if hotspot_percentage > 50:
                characteristics["spread_pattern"] = "widespread"
            elif hotspot_percentage > 20:
                characteristics["spread_pattern"] = "expanding"
            elif entropy > 0.7:  # High entropy indicates irregular pattern
                characteristics["spread_pattern"] = "irregular"
            else:
                characteristics["spread_pattern"] = "localized"
        
        return characteristics
    
    def _update_classification_history(self, fire_type: str, confidence: float, severity: int, characteristics: Dict[str, Any]) -> None:
        """
        Update the classification history with a new classification.
        
        Args:
            fire_type: Type of fire
            confidence: Confidence score for the classification
            severity: Severity score
            characteristics: Fire characteristics
        """
        # Add to history
        self.classification_history.append({
            "timestamp": datetime.now(),
            "fire_type": fire_type,
            "confidence": confidence,
            "severity": severity,
            "characteristics": characteristics
        })
        
        # Limit history size
        max_history = self.config.get("max_history_size", 100)
        if len(self.classification_history) > max_history:
            self.classification_history = self.classification_history[-max_history:]
    
    def _handle_update_thresholds(self, message: Message) -> Optional[Message]:
        """
        Handle a message to update classification thresholds.
        
        Args:
            message: Message containing new threshold values
            
        Returns:
            Optional response message
        """
        content = message.content
        
        if "confidence_threshold" in content:
            new_threshold = content["confidence_threshold"]
            if 0 <= new_threshold <= 1:
                self.confidence_threshold = new_threshold
                logger.info(f"Updated confidence threshold to {new_threshold}")
            else:
                logger.error(f"Invalid confidence threshold: {new_threshold}")
        
        # Send acknowledgment
        return self.create_message(
            message.sender_id,
            "threshold_update_ack",
            {
                "confidence_threshold": self.confidence_threshold
            }
        )
    
    def default_message_handler(self, message: Message) -> Optional[Message]:
        """
        Default handler for message types without a specific handler.
        
        Args:
            message: Incoming message
            
        Returns:
            Optional response message
        """
        logger.warning(f"Received unhandled message type: {message.message_type}")
        return None
    
    def create_message(self, receiver_id: str, message_type: str, content: Dict[str, Any], priority: int = 0) -> Message:
        """
        Create a new message to send to another agent.
        
        Args:
            receiver_id: ID of the receiving agent
            message_type: Type of message
            content: Message content
            priority: Message priority
            
        Returns:
            Created message
        """
        return Message(self.agent_id, receiver_id, message_type, content, priority)
    
    def save_state(self, filepath: str) -> None:
        """
        Save the agent's state to a file.
        
        Args:
            filepath: Path to save the state
        """
        import json
        
        state = {
            "confidence_threshold": self.confidence_threshold,
            "last_classification": self.last_classification,
            "last_classification_time": self.last_classification_time.isoformat() if self.last_classification_time else None,
            "classification_history": [
                {
                    "timestamp": entry["timestamp"].isoformat(),
                    "fire_type": entry["fire_type"],
                    "confidence": float(entry["confidence"]),
                    "severity": entry["severity"],
                    "characteristics": entry["characteristics"]
                }
                for entry in self.classification_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """
        Load the agent's state from a file.
        
        Args:
            filepath: Path to load the state from
        """
        import json
        from datetime import datetime
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.confidence_threshold = state["confidence_threshold"]
        self.last_classification = state["last_classification"]
        self.last_classification_time = datetime.fromisoformat(state["last_classification_time"]) if state["last_classification_time"] else None
        
        self.classification_history = [
            {
                "timestamp": datetime.fromisoformat(entry["timestamp"]),
                "fire_type": entry["fire_type"],
                "confidence": entry["confidence"],
                "severity": entry["severity"],
                "characteristics": entry["characteristics"]
            }
            for entry in state["classification_history"]
        ]