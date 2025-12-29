"""
Scenario manager for coordinating different environmental scenarios.
"""

from typing import Dict, Optional, List
from enum import Enum
import threading
import time

from .data_models import SensorReading, ScenarioConfig
from .normal_scenario import NormalScenarioGenerator
from .cooking_scenario import CookingScenarioGenerator
from .fire_scenario import FireScenarioGenerator


class ScenarioType(Enum):
    """Available scenario types."""
    NORMAL = "normal"
    COOKING = "cooking"
    FIRE = "fire"


class ScenarioManager:
    """Manages different environmental scenarios and data generation."""
    
    def __init__(self, default_config: Optional[ScenarioConfig] = None):
        self.config = default_config or ScenarioConfig()
        self.current_scenario = None
        self.current_generator = None
        self._lock = threading.Lock()
        
        # Initialize generators
        self.generators = {
            ScenarioType.NORMAL: NormalScenarioGenerator(self.config),
            ScenarioType.COOKING: CookingScenarioGenerator(self.config),
            ScenarioType.FIRE: FireScenarioGenerator(self.config)
        }
    
    def start_scenario(self, scenario_type: ScenarioType) -> bool:
        """Start a specific scenario simulation.
        
        Args:
            scenario_type: Type of scenario to start
            
        Returns:
            bool: True if scenario started successfully
        """
        with self._lock:
            # Stop current scenario if running
            if self.current_generator:
                self.current_generator.stop_streaming()
            
            # Start new scenario
            if scenario_type in self.generators:
                self.current_scenario = scenario_type
                self.current_generator = self.generators[scenario_type]
                self.current_generator.start_streaming()
                return True
            
            return False
    
    def stop_scenario(self) -> None:
        """Stop current scenario simulation."""
        with self._lock:
            if self.current_generator:
                self.current_generator.stop_streaming()
                self.current_generator = None
                self.current_scenario = None
    
    def get_current_data(self) -> Optional[SensorReading]:
        """Get current sensor readings from active scenario.
        
        Returns:
            SensorReading or None if no scenario is active
        """
        with self._lock:
            if self.current_generator:
                return self.current_generator.get_current_reading()
            return None
    
    def get_current_scenario(self) -> Optional[ScenarioType]:
        """Get currently active scenario type."""
        with self._lock:
            return self.current_scenario
    
    def is_running(self) -> bool:
        """Check if any scenario is currently running."""
        with self._lock:
            return self.current_generator is not None
    
    def generate_scenario_sequence(self, scenario_type: ScenarioType) -> List[SensorReading]:
        """Generate complete sequence for a scenario (for batch processing).
        
        Args:
            scenario_type: Type of scenario to generate
            
        Returns:
            List of sensor readings for the complete scenario
        """
        if scenario_type in self.generators:
            generator = self.generators[scenario_type]
            return generator.generate_sequence()
        return []
    
    def update_config(self, new_config: ScenarioConfig) -> None:
        """Update configuration for all generators.
        
        Args:
            new_config: New scenario configuration
        """
        with self._lock:
            self.config = new_config
            
            # Update all generators with new config
            self.generators = {
                ScenarioType.NORMAL: NormalScenarioGenerator(new_config),
                ScenarioType.COOKING: CookingScenarioGenerator(new_config),
                ScenarioType.FIRE: FireScenarioGenerator(new_config)
            }
            
            # Restart current scenario if one was running
            if self.current_scenario:
                current = self.current_scenario
                self.stop_scenario()
                self.start_scenario(current)
    
    def get_scenario_info(self) -> Dict:
        """Get information about available scenarios and current status.
        
        Returns:
            Dictionary with scenario information
        """
        with self._lock:
            return {
                'available_scenarios': [s.value for s in ScenarioType],
                'current_scenario': self.current_scenario.value if self.current_scenario else None,
                'is_running': self.is_running(),
                'config': {
                    'duration_seconds': self.config.duration_seconds,
                    'update_frequency': self.config.update_frequency,
                    'noise_level': self.config.noise_level,
                    'location': self.config.location
                }
            }