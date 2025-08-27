"""
Agent Architecture for the synthetic fire prediction system.

This module defines the overall agent system architecture, providing interfaces
for different types of agents, managing agent lifecycle and state, handling agent
configuration and initialization, and providing mechanisms for agent discovery and registration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Type, Set
import importlib
import inspect
import os
import json
import logging
from pathlib import Path
import uuid
from datetime import datetime

from ..base import Agent, Message

# Configure logging
logger = logging.getLogger(__name__)


class AgentState:
    """
    Class representing the state of an agent.
    """
    
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"
    
    def __init__(self, initial_state: str = INITIALIZING):
        """
        Initialize the agent state.
        
        Args:
            initial_state: Initial state of the agent
        """
        self.current_state = initial_state
        self.state_history = [(initial_state, datetime.now())]
        self.error_message = None
        
    def transition_to(self, new_state: str, error_message: Optional[str] = None) -> None:
        """
        Transition to a new state.
        
        Args:
            new_state: New state to transition to
            error_message: Error message if transitioning to ERROR state
        """
        self.current_state = new_state
        self.state_history.append((new_state, datetime.now()))
        
        if new_state == self.ERROR:
            self.error_message = error_message
            logger.error(f"Agent transitioned to ERROR state: {error_message}")
    
    def get_current_state(self) -> str:
        """
        Get the current state.
        
        Returns:
            Current state
        """
        return self.current_state
    
    def get_state_history(self) -> List[tuple]:
        """
        Get the state history.
        
        Returns:
            List of (state, timestamp) tuples
        """
        return self.state_history
    
    def get_error_message(self) -> Optional[str]:
        """
        Get the error message.
        
        Returns:
            Error message if in ERROR state, None otherwise
        """
        return self.error_message
    
    def is_active(self) -> bool:
        """
        Check if the agent is active (READY or RUNNING).
        
        Returns:
            True if active, False otherwise
        """
        return self.current_state in [self.READY, self.RUNNING]


class AgentConfiguration:
    """
    Class for managing agent configuration.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize the agent configuration.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        self.config = config_dict
        self.validate()
    
    def validate(self) -> None:
        """
        Validate the configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["agent_id", "agent_type"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return self.config
    
    @classmethod
    def from_file(cls, filepath: str) -> 'AgentConfiguration':
        """
        Load configuration from a file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            AgentConfiguration instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save configuration to a file.
        
        Args:
            filepath: Path to save the configuration
        """
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)


class AgentLifecycleManager:
    """
    Class for managing agent lifecycle.
    """
    
    def __init__(self):
        """
        Initialize the agent lifecycle manager.
        """
        self.agents = {}  # agent_id -> Agent
        self.agent_states = {}  # agent_id -> AgentState
    
    def register_agent(self, agent: Agent) -> None:
        """
        Register an agent with the lifecycle manager.
        
        Args:
            agent: Agent to register
        """
        self.agents[agent.agent_id] = agent
        self.agent_states[agent.agent_id] = AgentState()
        logger.info(f"Registered agent: {agent.agent_id}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the lifecycle manager.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.agent_states[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to get
            
        Returns:
            Agent instance if found, None otherwise
        """
        return self.agents.get(agent_id)
    
    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """
        Get the state of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            AgentState instance if found, None otherwise
        """
        return self.agent_states.get(agent_id)
    
    def start_agent(self, agent_id: str) -> bool:
        """
        Start an agent.
        
        Args:
            agent_id: ID of the agent to start
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agents:
            logger.error(f"Cannot start agent {agent_id}: Agent not found")
            return False
        
        agent_state = self.agent_states[agent_id]
        if agent_state.get_current_state() not in [AgentState.INITIALIZING, AgentState.READY, AgentState.PAUSED]:
            logger.error(f"Cannot start agent {agent_id}: Invalid state {agent_state.get_current_state()}")
            return False
        
        try:
            # Transition to RUNNING state
            agent_state.transition_to(AgentState.RUNNING)
            logger.info(f"Started agent: {agent_id}")
            return True
        except Exception as e:
            agent_state.transition_to(AgentState.ERROR, str(e))
            logger.error(f"Error starting agent {agent_id}: {e}")
            return False
    
    def pause_agent(self, agent_id: str) -> bool:
        """
        Pause an agent.
        
        Args:
            agent_id: ID of the agent to pause
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agents:
            logger.error(f"Cannot pause agent {agent_id}: Agent not found")
            return False
        
        agent_state = self.agent_states[agent_id]
        if agent_state.get_current_state() != AgentState.RUNNING:
            logger.error(f"Cannot pause agent {agent_id}: Not in RUNNING state")
            return False
        
        try:
            # Transition to PAUSED state
            agent_state.transition_to(AgentState.PAUSED)
            logger.info(f"Paused agent: {agent_id}")
            return True
        except Exception as e:
            agent_state.transition_to(AgentState.ERROR, str(e))
            logger.error(f"Error pausing agent {agent_id}: {e}")
            return False
    
    def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate an agent.
        
        Args:
            agent_id: ID of the agent to terminate
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agents:
            logger.error(f"Cannot terminate agent {agent_id}: Agent not found")
            return False
        
        try:
            # Transition to TERMINATED state
            self.agent_states[agent_id].transition_to(AgentState.TERMINATED)
            logger.info(f"Terminated agent: {agent_id}")
            return True
        except Exception as e:
            self.agent_states[agent_id].transition_to(AgentState.ERROR, str(e))
            logger.error(f"Error terminating agent {agent_id}: {e}")
            return False
    
    def get_all_agents(self) -> Dict[str, Agent]:
        """
        Get all registered agents.
        
        Returns:
            Dictionary mapping agent IDs to Agent instances
        """
        return self.agents
    
    def get_active_agents(self) -> Dict[str, Agent]:
        """
        Get all active agents.
        
        Returns:
            Dictionary mapping agent IDs to Agent instances for active agents
        """
        return {
            agent_id: agent
            for agent_id, agent in self.agents.items()
            if self.agent_states[agent_id].is_active()
        }


class AgentRegistry:
    """
    Class for agent discovery and registration.
    """
    
    def __init__(self):
        """
        Initialize the agent registry.
        """
        self.agent_classes = {}  # agent_type -> Agent class
    
    def register_agent_class(self, agent_type: str, agent_class: Type[Agent]) -> None:
        """
        Register an agent class.
        
        Args:
            agent_type: Type identifier for the agent class
            agent_class: Agent class to register
        """
        self.agent_classes[agent_type] = agent_class
        logger.info(f"Registered agent class: {agent_type}")
    
    def get_agent_class(self, agent_type: str) -> Optional[Type[Agent]]:
        """
        Get an agent class by type.
        
        Args:
            agent_type: Type identifier for the agent class
            
        Returns:
            Agent class if found, None otherwise
        """
        return self.agent_classes.get(agent_type)
    
    def create_agent(self, agent_type: str, agent_id: str, config: Dict[str, Any]) -> Optional[Agent]:
        """
        Create an agent instance.
        
        Args:
            agent_type: Type identifier for the agent class
            agent_id: ID for the new agent
            config: Configuration for the new agent
            
        Returns:
            Agent instance if successful, None otherwise
        """
        agent_class = self.get_agent_class(agent_type)
        if not agent_class:
            logger.error(f"Cannot create agent: Unknown agent type {agent_type}")
            return None
        
        try:
            agent = agent_class(agent_id, config)
            logger.info(f"Created agent: {agent_id} of type {agent_type}")
            return agent
        except Exception as e:
            logger.error(f"Error creating agent {agent_id} of type {agent_type}: {e}")
            return None
    
    def discover_agent_classes(self, package_path: str) -> None:
        """
        Discover agent classes in a package.
        
        Args:
            package_path: Path to the package to search
        """
        try:
            # Import the package
            package = importlib.import_module(package_path)
            
            # Get the directory of the package
            package_dir = os.path.dirname(package.__file__)
            
            # Iterate over Python files in the package
            for filename in os.listdir(package_dir):
                if filename.endswith('.py') and not filename.startswith('__'):
                    # Import the module
                    module_name = f"{package_path}.{filename[:-3]}"
                    module = importlib.import_module(module_name)
                    
                    # Find agent classes in the module
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, Agent) and obj != Agent:
                            # Register the agent class
                            agent_type = name
                            self.register_agent_class(agent_type, obj)
        except Exception as e:
            logger.error(f"Error discovering agent classes in {package_path}: {e}")


class AgentArchitecture:
    """
    Class defining the overall agent system architecture.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the agent architecture.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration if provided
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Initialize components
        self.lifecycle_manager = AgentLifecycleManager()
        self.registry = AgentRegistry()
        
        # Discover agent classes
        self._discover_agent_classes()
    
    def _discover_agent_classes(self) -> None:
        """
        Discover agent classes in the agents package.
        """
        agent_packages = [
            "src.agents.decision",
            "src.agents.monitoring",
            "src.agents.analysis",
            "src.agents.response",
            "src.agents.learning"
        ]
        
        for package in agent_packages:
            try:
                self.registry.discover_agent_classes(package)
            except Exception as e:
                logger.error(f"Error discovering agent classes in {package}: {e}")
    
    def create_agent(self, agent_type: str, config: Dict[str, Any]) -> Optional[Agent]:
        """
        Create and register an agent.
        
        Args:
            agent_type: Type of agent to create
            config: Configuration for the agent
            
        Returns:
            Created agent if successful, None otherwise
        """
        # Generate a unique ID if not provided
        if "agent_id" not in config:
            config["agent_id"] = f"{agent_type}_{str(uuid.uuid4())[:8]}"
        
        # Create the agent
        agent = self.registry.create_agent(agent_type, config["agent_id"], config)
        if not agent:
            return None
        
        # Register the agent with the lifecycle manager
        self.lifecycle_manager.register_agent(agent)
        
        return agent
    
    def start_agent(self, agent_id: str) -> bool:
        """
        Start an agent.
        
        Args:
            agent_id: ID of the agent to start
            
        Returns:
            True if successful, False otherwise
        """
        return self.lifecycle_manager.start_agent(agent_id)
    
    def pause_agent(self, agent_id: str) -> bool:
        """
        Pause an agent.
        
        Args:
            agent_id: ID of the agent to pause
            
        Returns:
            True if successful, False otherwise
        """
        return self.lifecycle_manager.pause_agent(agent_id)
    
    def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate an agent.
        
        Args:
            agent_id: ID of the agent to terminate
            
        Returns:
            True if successful, False otherwise
        """
        return self.lifecycle_manager.terminate_agent(agent_id)
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent instance if found, None otherwise
        """
        return self.lifecycle_manager.get_agent(agent_id)
    
    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """
        Get the state of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            AgentState instance if found, None otherwise
        """
        return self.lifecycle_manager.get_agent_state(agent_id)
    
    def get_all_agents(self) -> Dict[str, Agent]:
        """
        Get all registered agents.
        
        Returns:
            Dictionary mapping agent IDs to Agent instances
        """
        return self.lifecycle_manager.get_all_agents()
    
    def get_active_agents(self) -> Dict[str, Agent]:
        """
        Get all active agents.
        
        Returns:
            Dictionary mapping agent IDs to Agent instances for active agents
        """
        return self.lifecycle_manager.get_active_agents()
    
    def register_agent_class(self, agent_type: str, agent_class: Type[Agent]) -> None:
        """
        Register an agent class.
        
        Args:
            agent_type: Type identifier for the agent class
            agent_class: Agent class to register
        """
        self.registry.register_agent_class(agent_type, agent_class)
    
    def save_configuration(self, filepath: str) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            filepath: Path to save the configuration
        """
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_configuration(self, filepath: str) -> None:
        """
        Load configuration from a file.
        
        Args:
            filepath: Path to the configuration file
        """
        with open(filepath, 'r') as f:
            self.config = json.load(f)