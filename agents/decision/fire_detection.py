"""
Fire Detection Agent for the synthetic fire prediction system.

This module implements the FireDetectionAgent class, which is responsible for
making decisions about fire detection based on sensor data and model predictions.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from ..base import Agent, Message
from ...ml.models.identification import FireIdentificationModel  # Assuming this exists

# Configure logging
logger = logging.getLogger(__name__)


class FireDetectionAgent(Agent):
    """
    Agent responsible for making decisions about fire detection.
    
    This agent analyzes sensor data and model predictions to determine
    if a fire is present in the monitored environment.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the fire detection agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Dictionary containing configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Initialize detection thresholds
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.consecutive_detections_required = config.get("consecutive_detections_required", 3)
        
        # Initialize detection state
        self.detection_history = []
        self.current_detection_count = 0
        self.last_detection_time = None
        
        # Load fire identification model
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
        
        if "consecutive_detections_required" in self.config:
            count = self.config["consecutive_detections_required"]
            if not isinstance(count, int) or count < 1:
                raise ValueError(f"Consecutive detections required must be a positive integer, got {count}")
    
    def _load_model(self) -> FireIdentificationModel:
        """
        Load the fire identification model.
        
        Returns:
            Loaded model
        """
        try:
            model_path = self.config["model_path"]
            # This is a placeholder - actual implementation would depend on the model framework
            model = FireIdentificationModel.load(model_path)
            logger.info(f"Loaded fire identification model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading fire identification model: {e}")
            raise
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and make fire detection decisions.
        
        Args:
            data: Input data containing sensor readings and features
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Extract features for the model
            features = self._extract_features(data)
            
            # Get model prediction
            prediction = self.model.predict(features)
            confidence = self.model.predict_proba(features)[1]  # Assuming binary classification
            
            # Update detection history
            self._update_detection_history(prediction, confidence)
            
            # Make detection decision
            is_fire_detected = self._make_detection_decision()
            
            # Prepare results
            results = {
                "timestamp": datetime.now().isoformat(),
                "is_fire_detected": is_fire_detected,
                "confidence": confidence,
                "prediction": prediction,
                "consecutive_detections": self.current_detection_count,
                "threshold": self.confidence_threshold
            }
            
            # Log detection if fire is detected
            if is_fire_detected:
                logger.warning(f"Fire detected with confidence {confidence:.2f}")
                self.last_detection_time = datetime.now()
            
            return results
        except Exception as e:
            logger.error(f"Error in fire detection processing: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "is_fire_detected": False
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
                thermal_data.get("temperature_rise_slope", 0)
            ])
        
        # Extract gas features
        if "gas" in data:
            gas_data = data["gas"]
            features.extend([
                gas_data.get("methane_ppm", 0),
                gas_data.get("propane_ppm", 0),
                gas_data.get("hydrogen_ppm", 0),
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
    
    def _update_detection_history(self, prediction: int, confidence: float) -> None:
        """
        Update the detection history with a new prediction.
        
        Args:
            prediction: Binary prediction (1 for fire, 0 for no fire)
            confidence: Confidence score for the prediction
        """
        # Add to history
        self.detection_history.append({
            "timestamp": datetime.now(),
            "prediction": prediction,
            "confidence": confidence
        })
        
        # Limit history size
        max_history = self.config.get("max_history_size", 100)
        if len(self.detection_history) > max_history:
            self.detection_history = self.detection_history[-max_history:]
        
        # Update consecutive detection count
        if prediction == 1 and confidence >= self.confidence_threshold:
            self.current_detection_count += 1
        else:
            self.current_detection_count = 0
    
    def _make_detection_decision(self) -> bool:
        """
        Make a fire detection decision based on detection history.
        
        Returns:
            True if fire is detected, False otherwise
        """
        return self.current_detection_count >= self.consecutive_detections_required
    
    def _handle_update_thresholds(self, message: Message) -> Optional[Message]:
        """
        Handle a message to update detection thresholds.
        
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
        
        if "consecutive_detections_required" in content:
            new_count = content["consecutive_detections_required"]
            if isinstance(new_count, int) and new_count >= 1:
                self.consecutive_detections_required = new_count
                logger.info(f"Updated consecutive detections required to {new_count}")
            else:
                logger.error(f"Invalid consecutive detections required: {new_count}")
        
        # Send acknowledgment
        return self.create_message(
            message.sender_id,
            "threshold_update_ack",
            {
                "confidence_threshold": self.confidence_threshold,
                "consecutive_detections_required": self.consecutive_detections_required
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
            "consecutive_detections_required": self.consecutive_detections_required,
            "current_detection_count": self.current_detection_count,
            "last_detection_time": self.last_detection_time.isoformat() if self.last_detection_time else None,
            "detection_history": [
                {
                    "timestamp": entry["timestamp"].isoformat(),
                    "prediction": int(entry["prediction"]),
                    "confidence": float(entry["confidence"])
                }
                for entry in self.detection_history
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
        self.consecutive_detections_required = state["consecutive_detections_required"]
        self.current_detection_count = state["current_detection_count"]
        self.last_detection_time = datetime.fromisoformat(state["last_detection_time"]) if state["last_detection_time"] else None
        
        self.detection_history = [
            {
                "timestamp": datetime.fromisoformat(entry["timestamp"]),
                "prediction": entry["prediction"],
                "confidence": entry["confidence"]
            }
            for entry in state["detection_history"]
        ]