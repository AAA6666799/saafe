"""
Base interfaces for agent components.

This module defines the core interfaces and abstract classes for all agent components
in the synthetic fire prediction system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime
import uuid
import json


class Message:
    """
    Class representing a message exchanged between agents.
    """
    
    def __init__(self, 
                sender_id: str,
                receiver_id: str,
                message_type: str,
                content: Dict[str, Any],
                priority: int = 0):
        """
        Initialize a message.
        
        Args:
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent
            message_type: Type of message
            content: Message content
            priority: Message priority (higher values indicate higher priority)
        """
        self.id = str(uuid.uuid4())
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.content = content
        self.priority = priority
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        Create a message from a dictionary.
        
        Args:
            data: Dictionary representation of a message
            
        Returns:
            Message instance
        """
        message = cls(
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=data["message_type"],
            content=data["content"],
            priority=data["priority"]
        )
        message.id = data["id"]
        message.timestamp = datetime.fromisoformat(data["timestamp"])
        return message


class Agent(ABC):
    """
    Base abstract class for all agents in the system.
    
    This class defines the common interface that all agents must implement,
    regardless of their specific role (monitoring, analysis, response, learning).
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Dictionary containing configuration parameters
        """
        self.agent_id = agent_id
        self.config = config
        self.state = {}
        self.message_handlers = {}
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and update agent state.
        
        Args:
            data: Input data to process
            
        Returns:
            Dictionary containing processing results
        """
        pass
    
    def register_message_handler(self, message_type: str, handler: Callable[[Message], None]) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Function to call when a message of this type is received
        """
        self.message_handlers[message_type] = handler
    
    def handle_message(self, message: Message) -> Optional[Message]:
        """
        Handle an incoming message.
        
        Args:
            message: Incoming message
            
        Returns:
            Optional response message
        """
        if message.receiver_id != self.agent_id:
            return None
        
        if message.message_type in self.message_handlers:
            return self.message_handlers[message.message_type](message)
        
        return self.default_message_handler(message)
    
    @abstractmethod
    def default_message_handler(self, message: Message) -> Optional[Message]:
        """
        Default handler for message types without a specific handler.
        
        Args:
            message: Incoming message
            
        Returns:
            Optional response message
        """
        pass
    
    @abstractmethod
    def create_message(self, 
                      receiver_id: str,
                      message_type: str,
                      content: Dict[str, Any],
                      priority: int = 0) -> Message:
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
        pass
    
    @abstractmethod
    def save_state(self, filepath: str) -> None:
        """
        Save the agent's state to a file.
        
        Args:
            filepath: Path to save the state
        """
        pass
    
    @abstractmethod
    def load_state(self, filepath: str) -> None:
        """
        Load the agent's state from a file.
        
        Args:
            filepath: Path to load the state from
        """
        pass


class MonitoringAgent(Agent):
    """
    Abstract base class for monitoring agents.
    
    This class extends the base Agent with monitoring-specific methods.
    """
    
    @abstractmethod
    def detect_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the input data.
        
        Args:
            data: Input data to analyze
            
        Returns:
            List of detected anomalies
        """
        pass
    
    @abstractmethod
    def update_baseline(self, data: Dict[str, Any]) -> None:
        """
        Update the baseline model with new data.
        
        Args:
            data: New data to incorporate into the baseline
        """
        pass
    
    @abstractmethod
    def check_sensor_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the health of sensors based on their data.
        
        Args:
            data: Sensor data to analyze
            
        Returns:
            Dictionary containing sensor health information
        """
        pass


class AnalysisAgent(Agent):
    """
    Abstract base class for analysis agents.
    
    This class extends the base Agent with analysis-specific methods.
    """
    
    @abstractmethod
    def analyze_pattern(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze patterns in the input data.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @abstractmethod
    def calculate_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """
        Calculate confidence level for analysis results.
        
        Args:
            analysis_results: Results of data analysis
            
        Returns:
            Confidence level as a float between 0 and 1
        """
        pass
    
    @abstractmethod
    def match_fire_signature(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match data patterns against known fire signatures.
        
        Args:
            data: Input data to match
            
        Returns:
            Dictionary containing matching results
        """
        pass


class ResponseAgent(Agent):
    """
    Abstract base class for response agents.
    
    This class extends the base Agent with response-specific methods.
    """
    
    @abstractmethod
    def determine_response_level(self, risk_assessment: Dict[str, Any]) -> int:
        """
        Determine the appropriate response level based on risk assessment.
        
        Args:
            risk_assessment: Risk assessment data
            
        Returns:
            Response level as an integer (higher values indicate more severe response)
        """
        pass
    
    @abstractmethod
    def generate_alerts(self, risk_assessment: Dict[str, Any], response_level: int) -> List[Dict[str, Any]]:
        """
        Generate alerts based on risk assessment and response level.
        
        Args:
            risk_assessment: Risk assessment data
            response_level: Determined response level
            
        Returns:
            List of alerts to send
        """
        pass
    
    @abstractmethod
    def generate_recommendations(self, risk_assessment: Dict[str, Any], response_level: int) -> List[str]:
        """
        Generate action recommendations based on risk assessment and response level.
        
        Args:
            risk_assessment: Risk assessment data
            response_level: Determined response level
            
        Returns:
            List of recommended actions
        """
        pass


class LearningAgent(Agent):
    """
    Abstract base class for learning agents.
    
    This class extends the base Agent with learning-specific methods.
    """
    
    @abstractmethod
    def track_performance(self, metrics: Dict[str, float]) -> None:
        """
        Track system performance metrics.
        
        Args:
            metrics: Performance metrics to track
        """
        pass
    
    @abstractmethod
    def analyze_errors(self, error_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze system errors to identify patterns.
        
        Args:
            error_data: List of error data to analyze
            
        Returns:
            Dictionary containing error analysis results
        """
        pass
    
    @abstractmethod
    def recommend_improvements(self, performance_data: Dict[str, Any], error_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recommend system improvements based on performance and error analysis.
        
        Args:
            performance_data: Performance tracking data
            error_analysis: Error analysis results
            
        Returns:
            List of improvement recommendations
        """
        pass


class AgentCoordinator:
    """
    Class for coordinating multiple agents.
    
    This class manages agent communication and coordination.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the agent coordinator.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.agents = {}
        self.message_queue = []
    
    def register_agent(self, agent: Agent) -> None:
        """
        Register an agent with the coordinator.
        
        Args:
            agent: Agent to register
        """
        self.agents[agent.agent_id] = agent
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the coordinator.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def send_message(self, message: Message) -> None:
        """
        Send a message from one agent to another.
        
        Args:
            message: Message to send
        """
        self.message_queue.append(message)
    
    def process_messages(self) -> None:
        """
        Process all messages in the queue.
        """
        # Sort messages by priority
        self.message_queue.sort(key=lambda m: m.priority, reverse=True)
        
        while self.message_queue:
            message = self.message_queue.pop(0)
            if message.receiver_id in self.agents:
                response = self.agents[message.receiver_id].handle_message(message)
                if response:
                    self.message_queue.append(response)
    
    def broadcast_message(self, sender_id: str, message_type: str, content: Dict[str, Any], priority: int = 0) -> None:
        """
        Broadcast a message to all registered agents.
        
        Args:
            sender_id: ID of the sending agent
            message_type: Type of message
            content: Message content
            priority: Message priority
        """
        for agent_id in self.agents:
            if agent_id != sender_id:
                message = Message(sender_id, agent_id, message_type, content, priority)
                self.message_queue.append(message)