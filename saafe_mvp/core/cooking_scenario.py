"""
Enhanced cooking scenario generator with realistic cooking physics and methods.
Simulates different cooking techniques: frying, boiling, grilling, baking, etc.
"""

import numpy as np
from enum import Enum
from .scenario_generator import BaseScenarioGenerator
from .data_models import ScenarioConfig


class CookingMethod(Enum):
    """Different cooking methods with distinct sensor signatures."""
    FRYING = "frying"              # High heat, oil particles, sizzling
    GRILLING = "grilling"          # Very high heat, smoke, browning
    BOILING = "boiling"            # Steam, moderate heat, bubbling
    BAKING = "baking"              # Gradual heat, minimal particles
    SAUTEING = "sauteing"          # Medium heat, brief particles
    TOASTING = "toasting"          # Dry heat, browning particles


class CookingPhase(Enum):
    """Cooking process phases."""
    PREP = "prep"                  # Preparation, minimal activity
    HEATING = "heating"            # Heating up cookware/oven
    ACTIVE_COOKING = "active"      # Main cooking process
    BROWNING = "browning"          # Maillard reactions, higher particles
    FINISHING = "finishing"        # Final touches, reducing heat
    CLEANUP = "cleanup"            # Post-cooking, residual heat


class CookingScenarioGenerator(BaseScenarioGenerator):
    """Generator for realistic cooking activity with method-specific patterns."""
    
    def __init__(self, config: ScenarioConfig):
        super().__init__(config)
        
        # Randomly select cooking method for variety
        self.cooking_method = np.random.choice(list(CookingMethod))
        
        # Base kitchen environment values
        self.base_temperature = 23.0 + np.random.uniform(-1, 2)
        self.base_pm25 = 18.0 + np.random.uniform(-3, 5)
        self.base_co2 = 450.0 + np.random.uniform(-30, 50)
        self.base_audio = 38.0 + np.random.uniform(-5, 5)
        
        # Cooking phase durations
        self.prep_duration = 0.10         # 10% - preparation
        self.heating_duration = 0.15      # 15% - heating up
        self.active_duration = 0.45       # 45% - main cooking
        self.browning_duration = 0.15     # 15% - browning/finishing
        self.finishing_duration = 0.10    # 10% - final touches
        self.cleanup_duration = 0.05      # 5% - cleanup
        
        # Setup method-specific characteristics
        self._setup_cooking_characteristics()
    
    def _setup_cooking_characteristics(self):
        """Setup cooking-specific characteristics based on method."""
        if self.cooking_method == CookingMethod.FRYING:
            self.peak_temperature = 35.0      # Oil heating
            self.peak_pm25 = 95.0             # Oil particles, browning
            self.peak_co2 = 750.0             # Moderate CO2
            self.peak_audio = 62.0            # Sizzling sounds
            self.particle_burst_prob = 0.4    # Oil spattering
            self.temp_variation = 3.0         # Heat fluctuations
            
        elif self.cooking_method == CookingMethod.GRILLING:
            self.peak_temperature = 42.0      # High heat from grill
            self.peak_pm25 = 120.0            # Smoke from drippings
            self.peak_co2 = 680.0             # Less CO2, more ventilation
            self.peak_audio = 58.0            # Sizzling, some smoke
            self.particle_burst_prob = 0.6    # Smoke bursts
            self.temp_variation = 4.0         # Grill heat variations
            
        elif self.cooking_method == CookingMethod.BOILING:
            self.peak_temperature = 28.0      # Steam heat
            self.peak_pm25 = 35.0             # Minimal particles
            self.peak_co2 = 520.0             # Low CO2
            self.peak_audio = 48.0            # Bubbling sounds
            self.particle_burst_prob = 0.1    # Steam bursts
            self.temp_variation = 1.5         # Stable heat
            
        elif self.cooking_method == CookingMethod.BAKING:
            self.peak_temperature = 32.0      # Oven heat
            self.peak_pm25 = 45.0             # Minimal particles
            self.peak_co2 = 580.0             # Moderate CO2
            self.peak_audio = 42.0            # Quiet operation
            self.particle_burst_prob = 0.05   # Rare browning bursts
            self.temp_variation = 2.0         # Gradual heat
            
        elif self.cooking_method == CookingMethod.SAUTEING:
            self.peak_temperature = 33.0      # Medium-high heat
            self.peak_pm25 = 75.0             # Moderate particles
            self.peak_co2 = 650.0             # Moderate CO2
            self.peak_audio = 55.0            # Active cooking sounds
            self.particle_burst_prob = 0.3    # Stirring effects
            self.temp_variation = 2.5         # Medium variations
            
        else:  # TOASTING
            self.peak_temperature = 30.0      # Dry heat
            self.peak_pm25 = 65.0             # Browning particles
            self.peak_co2 = 480.0             # Low CO2
            self.peak_audio = 45.0            # Minimal sounds
            self.particle_burst_prob = 0.2    # Browning bursts
            self.temp_variation = 1.8         # Steady heat
    
    def generate_baseline_values(self, sample_index: int) -> tuple:
        """Generate cooking scenario values with realistic method-specific patterns."""
        progress = sample_index / self.config.total_samples
        
        # Determine current cooking phase
        phase, phase_progress = self._get_cooking_phase(progress)
        
        if phase == CookingPhase.PREP:
            return self._generate_prep_phase(phase_progress, sample_index)
        elif phase == CookingPhase.HEATING:
            return self._generate_heating_phase(phase_progress, sample_index)
        elif phase == CookingPhase.ACTIVE_COOKING:
            return self._generate_active_cooking_phase(phase_progress, sample_index)
        elif phase == CookingPhase.BROWNING:
            return self._generate_browning_phase(phase_progress, sample_index)
        elif phase == CookingPhase.FINISHING:
            return self._generate_finishing_phase(phase_progress, sample_index)
        else:  # CLEANUP
            return self._generate_cleanup_phase(phase_progress, sample_index)
    
    def _get_cooking_phase(self, progress: float) -> tuple:
        """Determine current cooking phase and progress within that phase."""
        if progress < self.prep_duration:
            phase_progress = progress / self.prep_duration
            return CookingPhase.PREP, phase_progress
        
        progress -= self.prep_duration
        if progress < self.heating_duration:
            phase_progress = progress / self.heating_duration
            return CookingPhase.HEATING, phase_progress
        
        progress -= self.heating_duration
        if progress < self.active_duration:
            phase_progress = progress / self.active_duration
            return CookingPhase.ACTIVE_COOKING, phase_progress
        
        progress -= self.active_duration
        if progress < self.browning_duration:
            phase_progress = progress / self.browning_duration
            return CookingPhase.BROWNING, phase_progress
        
        progress -= self.browning_duration
        if progress < self.finishing_duration:
            phase_progress = progress / self.finishing_duration
            return CookingPhase.FINISHING, phase_progress
        
        progress -= self.finishing_duration
        phase_progress = min(1.0, progress / self.cleanup_duration)
        return CookingPhase.CLEANUP, phase_progress
    
    def _generate_prep_phase(self, phase_progress: float, sample_index: int) -> tuple:
        """Generate preparation phase - minimal activity."""
        # Slight increases from movement and preparation
        temperature = self.base_temperature + phase_progress * 1.0
        pm25 = self.base_pm25 + np.random.uniform(-2, 3)
        co2 = self.base_co2 + phase_progress * 20.0
        audio_level = self.base_audio + phase_progress * 5.0
        
        # Occasional prep sounds
        if np.random.random() < 0.1:
            audio_level += np.random.uniform(3, 8)
        
        return (temperature, pm25, co2, audio_level)
    
    def _generate_heating_phase(self, phase_progress: float, sample_index: int) -> tuple:
        """Generate heating phase - cookware/oven heating up."""
        # Gradual temperature rise as equipment heats
        temp_rise = self._smooth_step(phase_progress) * (self.peak_temperature - self.base_temperature) * 0.4
        temperature = self.base_temperature + temp_rise
        
        # Minimal particles during heating
        pm25 = self.base_pm25 + phase_progress * 8.0
        
        # Slight CO2 increase from gas/electric usage
        co2 = self.base_co2 + phase_progress * 60.0
        
        # Equipment sounds
        audio_level = self.base_audio + phase_progress * 8.0
        
        # Method-specific heating effects
        if self.cooking_method == CookingMethod.FRYING and phase_progress > 0.7:
            # Oil starting to heat - occasional small pops
            if np.random.random() < 0.1:
                audio_level += np.random.uniform(2, 5)
        
        return (temperature, pm25, co2, audio_level)
    
    def _generate_active_cooking_phase(self, phase_progress: float, sample_index: int) -> tuple:
        """Generate active cooking phase - main cooking process."""
        # Peak cooking conditions with method-specific variations
        base_intensity = 0.8 + 0.2 * np.sin(2 * np.pi * phase_progress * 2)  # Cooking rhythm
        
        temperature = self.base_temperature + base_intensity * (self.peak_temperature - self.base_temperature)
        pm25 = self.base_pm25 + base_intensity * (self.peak_pm25 - self.base_pm25)
        co2 = self.base_co2 + base_intensity * (self.peak_co2 - self.base_co2)
        audio_level = self.base_audio + base_intensity * (self.peak_audio - self.base_audio)
        
        # Add temperature variations from cooking method
        temperature += np.random.uniform(-self.temp_variation/2, self.temp_variation/2)
        
        # Method-specific active cooking effects
        if np.random.random() < self.particle_burst_prob * base_intensity:
            if self.cooking_method == CookingMethod.FRYING:
                pm25 += np.random.exponential(12)  # Oil spattering
                audio_level += np.random.exponential(6)  # Sizzle bursts
            elif self.cooking_method == CookingMethod.GRILLING:
                pm25 += np.random.exponential(20)  # Smoke bursts
                temperature += np.random.uniform(2, 6)  # Heat flares
            elif self.cooking_method == CookingMethod.BOILING:
                audio_level += np.random.uniform(3, 8)  # Bubbling variations
            elif self.cooking_method == CookingMethod.SAUTEING:
                pm25 += np.random.uniform(5, 15)  # Stirring particles
                audio_level += np.random.uniform(4, 10)  # Stirring sounds
        
        # Stirring/flipping events
        if np.random.random() < 0.08:  # Occasional cooking actions
            pm25 += np.random.uniform(8, 20)
            audio_level += np.random.uniform(5, 12)
            if self.cooking_method in [CookingMethod.FRYING, CookingMethod.GRILLING]:
                temperature += np.random.uniform(1, 4)
        
        return (temperature, pm25, co2, audio_level)
    
    def _generate_browning_phase(self, phase_progress: float, sample_index: int) -> tuple:
        """Generate browning phase - Maillard reactions, higher particles."""
        # Elevated conditions for browning reactions
        browning_intensity = 0.9 + 0.1 * phase_progress
        
        temperature = self.peak_temperature * browning_intensity
        pm25 = self.peak_pm25 * browning_intensity * 1.2  # Higher particles from browning
        co2 = self.peak_co2 * browning_intensity
        audio_level = self.peak_audio * browning_intensity * 0.9  # Slightly quieter
        
        # Browning-specific effects
        if np.random.random() < 0.3:  # Browning particle bursts
            pm25 += np.random.exponential(15)
        
        if np.random.random() < 0.2:  # Caramelization sounds
            audio_level += np.random.uniform(3, 8)
        
        # Method-specific browning
        if self.cooking_method == CookingMethod.GRILLING:
            if np.random.random() < 0.4:  # Grill marks forming
                pm25 += np.random.exponential(25)
                temperature += np.random.uniform(2, 5)
        
        return (temperature, pm25, co2, audio_level)
    
    def _generate_finishing_phase(self, phase_progress: float, sample_index: int) -> tuple:
        """Generate finishing phase - final touches, reducing heat."""
        # Gradually decreasing intensity
        finish_intensity = 1.0 - phase_progress * 0.6
        
        temperature = self.peak_temperature * finish_intensity
        pm25 = self.peak_pm25 * finish_intensity * 0.7
        co2 = self.peak_co2 * finish_intensity
        audio_level = self.peak_audio * finish_intensity * 0.8
        
        # Final cooking actions
        if np.random.random() < 0.15:  # Plating, final seasoning
            pm25 += np.random.uniform(3, 10)
            audio_level += np.random.uniform(2, 6)
        
        return (temperature, pm25, co2, audio_level)
    
    def _generate_cleanup_phase(self, phase_progress: float, sample_index: int) -> tuple:
        """Generate cleanup phase - post-cooking, residual heat."""
        # Returning to baseline with residual effects
        cleanup_factor = 1.0 - phase_progress
        
        temperature = self.base_temperature + cleanup_factor * (self.peak_temperature - self.base_temperature) * 0.3
        pm25 = self.base_pm25 + cleanup_factor * 15.0  # Residual particles
        co2 = self.base_co2 + cleanup_factor * 80.0    # Residual CO2
        audio_level = self.base_audio + cleanup_factor * 8.0  # Cleanup sounds
        
        # Cleanup activities
        if np.random.random() < 0.2:  # Washing, moving cookware
            audio_level += np.random.uniform(5, 12)
        
        return (temperature, pm25, co2, audio_level)
    
    def _smooth_step(self, x: float) -> float:
        """Smooth step function for gradual transitions."""
        return x * x * (3 - 2 * x)