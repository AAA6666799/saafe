"""
Response Recommendation Agent for the synthetic fire prediction system.

This module implements the ResponseRecommendationAgent class, which is responsible for
recommending responses to detected fires based on fire classification and alert information.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from datetime import datetime
import logging
import json

from ..base import Agent, Message

# Configure logging
logger = logging.getLogger(__name__)


class ResponseAction:
    """
    Class representing a recommended response action.
    """
    
    # Action types
    TYPE_EVACUATION = "evacuation"
    TYPE_SUPPRESSION = "suppression"
    TYPE_VENTILATION = "ventilation"
    TYPE_ISOLATION = "isolation"
    TYPE_MONITORING = "monitoring"
    TYPE_NOTIFICATION = "notification"
    
    # Action priorities
    PRIORITY_LOW = 0
    PRIORITY_MEDIUM = 1
    PRIORITY_HIGH = 2
    PRIORITY_CRITICAL = 3
    
    def __init__(self, 
                action_type: str,
                description: str,
                priority: int,
                target_personnel: List[str],
                estimated_time: Optional[int] = None,
                prerequisites: Optional[List[str]] = None,
                equipment_needed: Optional[List[str]] = None):
        """
        Initialize a response action.
        
        Args:
            action_type: Type of action
            description: Detailed description of the action
            priority: Priority level (0-3, higher is more urgent)
            target_personnel: Personnel roles that should perform this action
            estimated_time: Estimated time to complete in seconds (if applicable)
            prerequisites: List of action IDs that must be completed before this one
            equipment_needed: List of equipment needed for this action
        """
        self.action_type = action_type
        self.description = description
        self.priority = priority
        self.target_personnel = target_personnel
        self.estimated_time = estimated_time
        self.prerequisites = prerequisites or []
        self.equipment_needed = equipment_needed or []
        self.completed = False
        self.completed_time = None
        self.completed_by = None
    
    def complete(self, personnel_id: str) -> None:
        """
        Mark the action as completed.
        
        Args:
            personnel_id: ID of the personnel who completed the action
        """
        self.completed = True
        self.completed_time = datetime.now()
        self.completed_by = personnel_id
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the action to a dictionary.
        
        Returns:
            Dictionary representation of the action
        """
        return {
            "action_type": self.action_type,
            "description": self.description,
            "priority": self.priority,
            "target_personnel": self.target_personnel,
            "estimated_time": self.estimated_time,
            "prerequisites": self.prerequisites,
            "equipment_needed": self.equipment_needed,
            "completed": self.completed,
            "completed_time": self.completed_time.isoformat() if self.completed_time else None,
            "completed_by": self.completed_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResponseAction':
        """
        Create a response action from a dictionary.
        
        Args:
            data: Dictionary representation of a response action
            
        Returns:
            ResponseAction instance
        """
        action = cls(
            action_type=data["action_type"],
            description=data["description"],
            priority=data["priority"],
            target_personnel=data["target_personnel"],
            estimated_time=data["estimated_time"],
            prerequisites=data["prerequisites"],
            equipment_needed=data["equipment_needed"]
        )
        
        action.completed = data["completed"]
        if data["completed_time"]:
            action.completed_time = datetime.fromisoformat(data["completed_time"])
        action.completed_by = data["completed_by"]
        
        return action


class ResponsePlan:
    """
    Class representing a response plan with multiple actions.
    """
    
    def __init__(self, 
                plan_id: str,
                fire_type: str,
                severity: int,
                location: str,
                created_time: datetime,
                actions: List[ResponseAction]):
        """
        Initialize a response plan.
        
        Args:
            plan_id: Unique identifier for the plan
            fire_type: Type of fire
            severity: Fire severity (1-10)
            location: Location of the fire
            created_time: Time when the plan was created
            actions: List of response actions
        """
        self.plan_id = plan_id
        self.fire_type = fire_type
        self.severity = severity
        self.location = location
        self.created_time = created_time
        self.actions = actions
        self.completed = False
        self.completed_time = None
    
    def add_action(self, action: ResponseAction) -> None:
        """
        Add an action to the plan.
        
        Args:
            action: Response action to add
        """
        self.actions.append(action)
    
    def get_actions_by_priority(self) -> Dict[int, List[ResponseAction]]:
        """
        Get actions grouped by priority.
        
        Returns:
            Dictionary mapping priority levels to lists of actions
        """
        result = {
            ResponseAction.PRIORITY_CRITICAL: [],
            ResponseAction.PRIORITY_HIGH: [],
            ResponseAction.PRIORITY_MEDIUM: [],
            ResponseAction.PRIORITY_LOW: []
        }
        
        for action in self.actions:
            result[action.priority].append(action)
        
        return result
    
    def get_incomplete_actions(self) -> List[ResponseAction]:
        """
        Get all incomplete actions.
        
        Returns:
            List of incomplete actions
        """
        return [action for action in self.actions if not action.completed]
    
    def get_next_actions(self) -> List[ResponseAction]:
        """
        Get the next actions that can be performed.
        
        Returns:
            List of actions that can be performed next
        """
        incomplete = self.get_incomplete_actions()
        completed_ids = [a.action_id for a in self.actions if a.completed]
        
        return [
            action for action in incomplete
            if all(prereq in completed_ids for prereq in action.prerequisites)
        ]
    
    def check_completion(self) -> bool:
        """
        Check if the plan is completed.
        
        Returns:
            True if all actions are completed, False otherwise
        """
        if all(action.completed for action in self.actions):
            self.completed = True
            self.completed_time = datetime.now()
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the plan to a dictionary.
        
        Returns:
            Dictionary representation of the plan
        """
        return {
            "plan_id": self.plan_id,
            "fire_type": self.fire_type,
            "severity": self.severity,
            "location": self.location,
            "created_time": self.created_time.isoformat(),
            "actions": [action.to_dict() for action in self.actions],
            "completed": self.completed,
            "completed_time": self.completed_time.isoformat() if self.completed_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResponsePlan':
        """
        Create a response plan from a dictionary.
        
        Args:
            data: Dictionary representation of a response plan
            
        Returns:
            ResponsePlan instance
        """
        actions = [ResponseAction.from_dict(action_dict) for action_dict in data["actions"]]
        
        plan = cls(
            plan_id=data["plan_id"],
            fire_type=data["fire_type"],
            severity=data["severity"],
            location=data["location"],
            created_time=datetime.fromisoformat(data["created_time"]),
            actions=actions
        )
        
        plan.completed = data["completed"]
        if data["completed_time"]:
            plan.completed_time = datetime.fromisoformat(data["completed_time"])
        
        return plan


class ResponseRecommendationAgent(Agent):
    """
    Agent responsible for recommending responses to detected fires.
    
    This agent analyzes fire classification and alert information to generate
    appropriate response recommendations.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the response recommendation agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Dictionary containing configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Load response templates
        self.response_templates = self._load_response_templates()
        
        # Initialize response state
        self.active_plans = {}  # plan_id -> ResponsePlan
        self.plan_history = []
        
        # Register message handlers
        self.register_message_handler("action_completed", self._handle_action_completed)
        self.register_message_handler("update_templates", self._handle_update_templates)
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["templates_path"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
    
    def _load_response_templates(self) -> Dict[str, Any]:
        """
        Load response templates from the configuration.
        
        Returns:
            Dictionary of response templates
        """
        try:
            templates_path = self.config["templates_path"]
            with open(templates_path, 'r') as f:
                templates = json.load(f)
            
            logger.info(f"Loaded response templates from {templates_path}")
            return templates
        except Exception as e:
            logger.error(f"Error loading response templates: {e}")
            # Return default templates
            return self._get_default_templates()
    
    def _get_default_templates(self) -> Dict[str, Any]:
        """
        Get default response templates.
        
        Returns:
            Dictionary of default response templates
        """
        return {
            "fire_types": {
                "electrical": {
                    "actions": [
                        {
                            "action_type": ResponseAction.TYPE_ISOLATION,
                            "description": "Shut off power to affected area",
                            "priority": ResponseAction.PRIORITY_CRITICAL,
                            "target_personnel": ["electrician", "safety_officer"],
                            "equipment_needed": ["power_shutoff_key"]
                        },
                        {
                            "action_type": ResponseAction.TYPE_SUPPRESSION,
                            "description": "Use Class C fire extinguisher",
                            "priority": ResponseAction.PRIORITY_HIGH,
                            "target_personnel": ["fire_response_team"],
                            "equipment_needed": ["class_c_extinguisher"]
                        },
                        {
                            "action_type": ResponseAction.TYPE_EVACUATION,
                            "description": "Evacuate personnel from affected area",
                            "priority": ResponseAction.PRIORITY_HIGH,
                            "target_personnel": ["safety_officer", "floor_warden"]
                        }
                    ]
                },
                "chemical": {
                    "actions": [
                        {
                            "action_type": ResponseAction.TYPE_EVACUATION,
                            "description": "Evacuate personnel from affected area",
                            "priority": ResponseAction.PRIORITY_CRITICAL,
                            "target_personnel": ["safety_officer", "floor_warden"]
                        },
                        {
                            "action_type": ResponseAction.TYPE_ISOLATION,
                            "description": "Isolate chemical source if safe to do so",
                            "priority": ResponseAction.PRIORITY_HIGH,
                            "target_personnel": ["hazmat_team"],
                            "equipment_needed": ["hazmat_suit", "breathing_apparatus"]
                        },
                        {
                            "action_type": ResponseAction.TYPE_SUPPRESSION,
                            "description": "Use appropriate fire suppression for chemical type",
                            "priority": ResponseAction.PRIORITY_HIGH,
                            "target_personnel": ["hazmat_team", "fire_response_team"],
                            "equipment_needed": ["chemical_fire_extinguisher"]
                        }
                    ]
                },
                "smoldering": {
                    "actions": [
                        {
                            "action_type": ResponseAction.TYPE_MONITORING,
                            "description": "Monitor area for full ignition",
                            "priority": ResponseAction.PRIORITY_HIGH,
                            "target_personnel": ["fire_response_team"]
                        },
                        {
                            "action_type": ResponseAction.TYPE_SUPPRESSION,
                            "description": "Apply water mist to smoldering area",
                            "priority": ResponseAction.PRIORITY_MEDIUM,
                            "target_personnel": ["fire_response_team"],
                            "equipment_needed": ["water_mist_extinguisher"]
                        },
                        {
                            "action_type": ResponseAction.TYPE_VENTILATION,
                            "description": "Increase ventilation to remove smoke",
                            "priority": ResponseAction.PRIORITY_MEDIUM,
                            "target_personnel": ["facility_manager"],
                            "equipment_needed": ["ventilation_controls"]
                        }
                    ]
                },
                "rapid_combustion": {
                    "actions": [
                        {
                            "action_type": ResponseAction.TYPE_EVACUATION,
                            "description": "Immediate evacuation of all personnel",
                            "priority": ResponseAction.PRIORITY_CRITICAL,
                            "target_personnel": ["all_personnel", "safety_officer"]
                        },
                        {
                            "action_type": ResponseAction.TYPE_NOTIFICATION,
                            "description": "Notify emergency services",
                            "priority": ResponseAction.PRIORITY_CRITICAL,
                            "target_personnel": ["security_officer", "facility_manager"]
                        },
                        {
                            "action_type": ResponseAction.TYPE_ISOLATION,
                            "description": "Activate fire doors and containment systems",
                            "priority": ResponseAction.PRIORITY_HIGH,
                            "target_personnel": ["facility_manager", "safety_officer"],
                            "equipment_needed": ["fire_control_panel"]
                        }
                    ]
                },
                "unknown": {
                    "actions": [
                        {
                            "action_type": ResponseAction.TYPE_EVACUATION,
                            "description": "Evacuate personnel from affected area",
                            "priority": ResponseAction.PRIORITY_HIGH,
                            "target_personnel": ["safety_officer", "floor_warden"]
                        },
                        {
                            "action_type": ResponseAction.TYPE_NOTIFICATION,
                            "description": "Notify fire response team for investigation",
                            "priority": ResponseAction.PRIORITY_HIGH,
                            "target_personnel": ["security_officer"]
                        },
                        {
                            "action_type": ResponseAction.TYPE_MONITORING,
                            "description": "Monitor situation and prepare for escalation",
                            "priority": ResponseAction.PRIORITY_MEDIUM,
                            "target_personnel": ["fire_response_team", "safety_officer"]
                        }
                    ]
                }
            },
            "severity_modifiers": {
                "high": {
                    "min_severity": 7,
                    "additional_actions": [
                        {
                            "action_type": ResponseAction.TYPE_NOTIFICATION,
                            "description": "Notify external emergency services",
                            "priority": ResponseAction.PRIORITY_CRITICAL,
                            "target_personnel": ["security_officer", "facility_manager"]
                        },
                        {
                            "action_type": ResponseAction.TYPE_EVACUATION,
                            "description": "Full building evacuation",
                            "priority": ResponseAction.PRIORITY_CRITICAL,
                            "target_personnel": ["all_personnel", "safety_officer"]
                        }
                    ]
                },
                "medium": {
                    "min_severity": 4,
                    "additional_actions": [
                        {
                            "action_type": ResponseAction.TYPE_NOTIFICATION,
                            "description": "Notify facility management",
                            "priority": ResponseAction.PRIORITY_HIGH,
                            "target_personnel": ["security_officer"]
                        },
                        {
                            "action_type": ResponseAction.TYPE_EVACUATION,
                            "description": "Evacuate immediate area and adjacent rooms",
                            "priority": ResponseAction.PRIORITY_HIGH,
                            "target_personnel": ["floor_warden", "safety_officer"]
                        }
                    ]
                },
                "low": {
                    "min_severity": 1,
                    "additional_actions": [
                        {
                            "action_type": ResponseAction.TYPE_MONITORING,
                            "description": "Continuous monitoring of the situation",
                            "priority": ResponseAction.PRIORITY_MEDIUM,
                            "target_personnel": ["fire_response_team"]
                        }
                    ]
                }
            }
        }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and generate response recommendations.
        
        Args:
            data: Input data containing fire detection, classification, and alert information
            
        Returns:
            Dictionary containing response recommendation results
        """
        try:
            # Check if fire is detected
            if not data.get("is_fire_detected", False):
                return {
                    "timestamp": datetime.now().isoformat(),
                    "plan_generated": False,
                    "reason": "no_fire_detected"
                }
            
            # Extract fire information
            fire_type = data.get("fire_type", "unknown")
            severity = data.get("severity", 0)
            location = data.get("location", "Unknown")
            alert_id = data.get("alert_id")
            
            # Generate response plan
            plan = self._generate_response_plan(fire_type, severity, location, alert_id)
            
            # Store plan
            self.active_plans[plan.plan_id] = plan
            self.plan_history.append(plan)
            
            # Limit plan history
            max_history = self.config.get("max_history_size", 100)
            if len(self.plan_history) > max_history:
                self.plan_history = self.plan_history[-max_history:]
            
            # Log plan generation
            logger.info(f"Generated response plan for {fire_type} fire with severity {severity}")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "plan_generated": True,
                "plan": plan.to_dict()
            }
        except Exception as e:
            logger.error(f"Error in response recommendation processing: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "plan_generated": False
            }
    
    def _generate_response_plan(self, fire_type: str, severity: int, location: str, alert_id: Optional[str] = None) -> ResponsePlan:
        """
        Generate a response plan based on fire type and severity.
        
        Args:
            fire_type: Type of fire
            severity: Fire severity (1-10)
            location: Location of the fire
            alert_id: ID of the associated alert
            
        Returns:
            Generated response plan
        """
        import uuid
        
        # Create plan ID
        plan_id = str(uuid.uuid4())
        
        # Get base actions for fire type
        if fire_type in self.response_templates["fire_types"]:
            base_actions = self.response_templates["fire_types"][fire_type]["actions"]
        else:
            # Use unknown fire type if specific type not found
            base_actions = self.response_templates["fire_types"]["unknown"]["actions"]
        
        # Determine severity level
        severity_level = "low"
        if severity >= self.response_templates["severity_modifiers"]["high"]["min_severity"]:
            severity_level = "high"
        elif severity >= self.response_templates["severity_modifiers"]["medium"]["min_severity"]:
            severity_level = "medium"
        
        # Add severity-specific actions
        additional_actions = self.response_templates["severity_modifiers"][severity_level]["additional_actions"]
        
        # Combine actions
        all_action_dicts = base_actions + additional_actions
        
        # Create ResponseAction objects
        actions = []
        for i, action_dict in enumerate(all_action_dicts):
            action = ResponseAction(
                action_type=action_dict["action_type"],
                description=action_dict["description"],
                priority=action_dict["priority"],
                target_personnel=action_dict["target_personnel"],
                estimated_time=action_dict.get("estimated_time"),
                prerequisites=action_dict.get("prerequisites", []),
                equipment_needed=action_dict.get("equipment_needed", [])
            )
            actions.append(action)
        
        # Create response plan
        plan = ResponsePlan(
            plan_id=plan_id,
            fire_type=fire_type,
            severity=severity,
            location=location,
            created_time=datetime.now(),
            actions=actions
        )
        
        return plan
    
    def _handle_action_completed(self, message: Message) -> Optional[Message]:
        """
        Handle a message indicating that an action has been completed.
        
        Args:
            message: Message containing action completion information
            
        Returns:
            Optional response message
        """
        content = message.content
        
        if "plan_id" not in content or "action_index" not in content or "personnel_id" not in content:
            logger.error("Missing plan_id, action_index, or personnel_id in action_completed message")
            return self.create_message(
                message.sender_id,
                "action_completed_nack",
                {
                    "error": "Missing plan_id, action_index, or personnel_id"
                }
            )
        
        plan_id = content["plan_id"]
        action_index = content["action_index"]
        personnel_id = content["personnel_id"]
        
        if plan_id in self.active_plans:
            plan = self.active_plans[plan_id]
            
            if 0 <= action_index < len(plan.actions):
                action = plan.actions[action_index]
                action.complete(personnel_id)
                
                logger.info(f"Action {action_index} of plan {plan_id} completed by {personnel_id}")
                
                # Check if plan is completed
                plan_completed = plan.check_completion()
                if plan_completed:
                    logger.info(f"Plan {plan_id} completed")
                
                return self.create_message(
                    message.sender_id,
                    "action_completed_ack",
                    {
                        "plan_id": plan_id,
                        "action_index": action_index,
                        "completed": True,
                        "plan_completed": plan_completed
                    }
                )
            else:
                logger.warning(f"Invalid action index {action_index} for plan {plan_id}")
                
                return self.create_message(
                    message.sender_id,
                    "action_completed_nack",
                    {
                        "plan_id": plan_id,
                        "action_index": action_index,
                        "error": "Invalid action index"
                    }
                )
        else:
            logger.warning(f"Attempt to complete action for unknown plan {plan_id}")
            
            return self.create_message(
                message.sender_id,
                "action_completed_nack",
                {
                    "plan_id": plan_id,
                    "error": "Plan not found"
                }
            )
    
    def _handle_update_templates(self, message: Message) -> Optional[Message]:
        """
        Handle a message to update response templates.
        
        Args:
            message: Message containing new templates
            
        Returns:
            Optional response message
        """
        content = message.content
        
        if "templates" in content:
            try:
                new_templates = content["templates"]
                self.response_templates = new_templates
                
                # Save templates to file if path provided
                if "save_path" in content:
                    save_path = content["save_path"]
                    with open(save_path, 'w') as f:
                        json.dump(new_templates, f, indent=2)
                    
                    logger.info(f"Saved updated templates to {save_path}")
                
                logger.info("Updated response templates")
                
                return self.create_message(
                    message.sender_id,
                    "templates_update_ack",
                    {
                        "success": True
                    }
                )
            except Exception as e:
                logger.error(f"Error updating templates: {e}")
                
                return self.create_message(
                    message.sender_id,
                    "templates_update_nack",
                    {
                        "error": str(e)
                    }
                )
        else:
            logger.error("Missing templates in update_templates message")
            
            return self.create_message(
                message.sender_id,
                "templates_update_nack",
                {
                    "error": "Missing templates"
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
        state = {
            "active_plans": {
                plan_id: plan.to_dict()
                for plan_id, plan in self.active_plans.items()
            },
            "plan_history": [plan.to_dict() for plan in self.plan_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """
        Load the agent's state from a file.
        
        Args:
            filepath: Path to load the state from
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.active_plans = {
            plan_id: ResponsePlan.from_dict(plan_dict)
            for plan_id, plan_dict in state["active_plans"].items()
        }
        
        self.plan_history = [
            ResponsePlan.from_dict(plan_dict)
            for plan_dict in state["plan_history"]
        ]