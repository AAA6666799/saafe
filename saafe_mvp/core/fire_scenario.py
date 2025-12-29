"""
Enhanced fire scenario generator with realistic fire physics and development phases.
Based on actual fire science: Incipient → Growth → Flashover → Fully Developed
"""

import numpy as np
from enum import Enum
from .scenario_generator import BaseScenarioGenerator
from .data_models import ScenarioConfig


class FireType(Enum):
    """Different types of fires with distinct characteristics."""
    ELECTRICAL = "electrical"      # Fast temperature rise, moderate smoke
    GREASE = "grease"              # Very high temperature, heavy smoke
    PAPER_WOOD = "paper_wood"      # Gradual buildup, heavy smoke
    CHEMICAL = "chemical"          # Rapid development, toxic gases


class FirePhase(Enum):
    """Fire development phases based on fire science."""
    INCIPIENT = "incipient"        # Initial heating, minimal visible signs
    GROWTH = "growth"              # Visible flames, rapid development
    FLASHOVER = "flashover"        # Sudden intense burning
    FULLY_DEVELOPED = "fully_developed"  # Peak fire conditions
    DECAY = "decay"                # Fire consuming available fuel


class FireScenarioGenerator(BaseScenarioGenerator):
    """Generator for realistic fire emergency with physics-based development."""
    
    def __init__(self, config: ScenarioConfig):
        super().__init__(config)
        
        # Randomly select fire type for variety
        self.fire_type = np.random.choice(list(FireType))
        
        # Initial normal values
        self.initial_temperature = 22.0 + np.random.uniform(-1, 2)
        self.initial_pm25 = 12.0 + np.random.uniform(-3, 5)
        self.initial_co2 = 420.0 + np.random.uniform(-30, 60)
        self.initial_audio = 35.0 + np.random.uniform(-5, 8)
        
        # Fire development timeline (based on real fire science)
        self.incipient_duration = 0.20    # 20% - slow heating phase
        self.growth_duration = 0.15       # 15% - rapid development
        self.flashover_duration = 0.05    # 5% - sudden intense phase
        self.fully_developed_duration = 0.45  # 45% - peak burning
        self.decay_duration = 0.15        # 15% - fuel depletion
        
        # Fire type specific characteristics
        self._setup_fire_characteristics()
    
    def _setup_fire_characteristics(self):
        """Setup fire-specific characteristics based on fire type."""
        if self.fire_type == FireType.ELECTRICAL:
            self.peak_temperature = 95.0
            self.peak_pm25 = 180.0
            self.peak_co2 = 1200.0
            self.peak_audio = 65.0
            self.temp_growth_rate = 4.0
            self.smoke_intensity = 0.7
            
        elif self.fire_type == FireType.GREASE:
            self.peak_temperature = 120.0  # Grease fires burn very hot
            self.peak_pm25 = 350.0         # Heavy black smoke
            self.peak_co2 = 2200.0
            self.peak_audio = 80.0         # Violent burning sounds
            self.temp_growth_rate = 5.0
            self.smoke_intensity = 1.2
            
        elif self.fire_type == FireType.PAPER_WOOD:
            self.peak_temperature = 85.0
            self.peak_pm25 = 280.0         # Heavy smoke from cellulose
            self.peak_co2 = 1800.0
            self.peak_audio = 70.0         # Crackling sounds
            self.temp_growth_rate = 2.5
            self.smoke_intensity = 1.0
            
        else:  # CHEMICAL
            self.peak_temperature = 110.0
            self.peak_pm25 = 220.0
            self.peak_co2 = 2500.0         # High CO2 from chemical reactions
            self.peak_audio = 75.0
            self.temp_growth_rate = 6.0
            self.smoke_intensity = 0.9
    
    def generate_baseline_values(self, sample_index: int) -> tuple:
        """Generate fire scenario values with realistic physics-based development."""
        progress = sample_index / self.config.total_samples
        
        # Determine current fire phase
        phase, phase_progress = self._get_fire_phase(progress)
        
        if phase == FirePhase.INCIPIENT:
            return self._generate_incipient_phase(phase_progress, sample_index)
        elif phase == FirePhase.GROWTH:
            return self._generate_growth_phase(phase_progress, sample_index)
        elif phase == FirePhase.FLASHOVER:
            return self._generate_flashover_phase(phase_progress, sample_index)
        elif phase == FirePhase.FULLY_DEVELOPED:
            return self._generate_fully_developed_phase(phase_progress, sample_index)
        else:  # DECAY
            return self._generate_decay_phase(phase_progress, sample_index)
    
    def _get_fire_phase(self, progress: float) -> tuple:
        """Determine current fire phase and progress within that phase."""
        if progress < self.incipient_duration:
            phase_progress = progress / self.incipient_duration
            return FirePhase.INCIPIENT, phase_progress
        
        progress -= self.incipient_duration
        if progress < self.growth_duration:
            phase_progress = progress / self.growth_duration
            return FirePhase.GROWTH, phase_progress
        
        progress -= self.growth_duration
        if progress < self.flashover_duration:
            phase_progress = progress / self.flashover_duration
            return FirePhase.FLASHOVER, phase_progress
        
        progress -= self.flashover_duration
        if progress < self.fully_developed_duration:
            phase_progress = progress / self.fully_developed_duration
            return FirePhase.FULLY_DEVELOPED, phase_progress
        
        progress -= self.fully_developed_duration
        phase_progress = min(1.0, progress / self.decay_duration)
        return FirePhase.DECAY, phase_progress
    
    def _generate_incipient_phase(self, phase_progress: float, sample_index: int) -> tuple:
        """Generate incipient phase - slow heating, minimal visible signs."""
        # Very gradual temperature increase (pyrolysis beginning)
        temp_rise = phase_progress ** 2 * 8.0  # Slow quadratic rise
        temperature = self.initial_temperature + temp_rise
        
        # Minimal smoke production initially
        pm25_rise = phase_progress ** 3 * 15.0  # Very slow smoke increase
        pm25 = self.initial_pm25 + pm25_rise
        
        # Slight CO2 increase from early decomposition
        co2_rise = phase_progress ** 2 * 80.0
        co2 = self.initial_co2 + co2_rise
        
        # Minimal audio changes
        audio_level = self.initial_audio + np.random.uniform(-2, 3)
        
        # Add subtle pre-fire indicators
        if phase_progress > 0.7:  # Late incipient phase
            if np.random.random() < 0.1:  # Occasional small temperature spikes
                temperature += np.random.exponential(2)
            if np.random.random() < 0.05:  # Rare smoke puffs
                pm25 += np.random.exponential(8)
        
        return (temperature, pm25, co2, audio_level)
    
    def _generate_growth_phase(self, phase_progress: float, sample_index: int) -> tuple:
        """Generate growth phase - visible flames, rapid development."""
        # Exponential temperature rise (fire triangle established)
        temp_intensity = phase_progress ** (1/self.temp_growth_rate)
        temperature = self.initial_temperature + temp_intensity * (self.peak_temperature * 0.6 - self.initial_temperature)
        
        # Rapid smoke production
        smoke_intensity = phase_progress ** 1.2 * self.smoke_intensity
        pm25 = self.initial_pm25 + smoke_intensity * (self.peak_pm25 * 0.7 - self.initial_pm25)
        
        # CO2 from active combustion
        co2_intensity = phase_progress ** 1.1
        co2 = self.initial_co2 + co2_intensity * (self.peak_co2 * 0.6 - self.initial_co2)
        
        # Fire sounds becoming audible
        audio_intensity = phase_progress ** 0.8
        audio_level = self.initial_audio + audio_intensity * (self.peak_audio * 0.5 - self.initial_audio)
        
        # Growth phase specific effects
        if phase_progress > 0.3:
            # Temperature spikes from flame spread
            if np.random.random() < 0.25 * phase_progress:
                temperature += np.random.exponential(8)
            
            # Smoke bursts from material ignition
            if np.random.random() < 0.3 * phase_progress:
                pm25 += np.random.exponential(25)
            
            # Crackling and popping sounds
            if np.random.random() < 0.4 * phase_progress:
                audio_level += np.random.exponential(6)
        
        return (temperature, pm25, co2, audio_level)
    
    def _generate_flashover_phase(self, phase_progress: float, sample_index: int) -> tuple:
        """Generate flashover phase - sudden intense burning."""
        # Dramatic temperature spike (flashover = 500-600°C, but sensor maxes out)
        flashover_intensity = 1.0 + phase_progress * 0.5  # Beyond normal scale
        temperature = self.peak_temperature * flashover_intensity
        
        # Massive smoke production
        pm25 = self.peak_pm25 * (1.2 + phase_progress * 0.3)
        
        # Peak CO2 from intense combustion
        co2 = self.peak_co2 * (1.1 + phase_progress * 0.4)
        
        # Loud fire sounds
        audio_level = self.peak_audio * (1.0 + phase_progress * 0.3)
        
        # Flashover specific effects - very chaotic
        if np.random.random() < 0.8:  # Frequent temperature spikes
            temperature += np.random.exponential(15)
        
        if np.random.random() < 0.9:  # Massive smoke bursts
            pm25 += np.random.exponential(50)
        
        if np.random.random() < 0.7:  # Roaring sounds
            audio_level += np.random.exponential(12)
        
        return (temperature, pm25, co2, audio_level)
    
    def _generate_fully_developed_phase(self, phase_progress: float, sample_index: int) -> tuple:
        """Generate fully developed phase - peak fire conditions."""
        # High but somewhat stable temperatures
        base_intensity = 0.9 + 0.1 * np.sin(2 * np.pi * phase_progress * 4)  # Fire breathing
        temperature = self.peak_temperature * base_intensity
        
        # Heavy smoke production
        pm25 = self.peak_pm25 * (0.8 + 0.2 * np.sin(2 * np.pi * phase_progress * 3))
        
        # High CO2 from sustained combustion
        co2 = self.peak_co2 * (0.85 + 0.15 * np.sin(2 * np.pi * phase_progress * 2))
        
        # Sustained fire sounds
        audio_level = self.peak_audio * (0.9 + 0.1 * np.sin(2 * np.pi * phase_progress * 5))
        
        # Fully developed fire effects
        if np.random.random() < 0.4:  # Regular flare-ups
            temperature += np.random.exponential(12)
            pm25 += np.random.exponential(35)
        
        if np.random.random() < 0.3:  # Structural burning sounds
            audio_level += np.random.exponential(10)
        
        if np.random.random() < 0.2:  # Ventilation effects
            pm25 += np.random.exponential(60)
            co2 += np.random.exponential(200)
        
        return (temperature, pm25, co2, audio_level)
    
    def _generate_decay_phase(self, phase_progress: float, sample_index: int) -> tuple:
        """Generate decay phase - fire consuming available fuel."""
        # Gradually decreasing temperature
        decay_factor = 1.0 - phase_progress ** 0.7
        temperature = self.peak_temperature * decay_factor * 0.8
        
        # Smoke may increase initially (smoldering) then decrease
        if phase_progress < 0.3:
            smoke_factor = 1.0 + phase_progress * 0.5  # Initial smoke increase
        else:
            smoke_factor = 1.5 - phase_progress  # Then decrease
        pm25 = self.peak_pm25 * decay_factor * smoke_factor * 0.6
        
        # CO2 remains elevated from smoldering
        co2 = self.peak_co2 * decay_factor * 0.7
        
        # Audio decreases but may have occasional collapses
        audio_level = self.peak_audio * decay_factor * 0.5
        
        # Decay phase effects
        if phase_progress < 0.5 and np.random.random() < 0.2:  # Smoldering bursts
            pm25 += np.random.exponential(30)
            co2 += np.random.exponential(150)
        
        if np.random.random() < 0.1:  # Structural collapse sounds
            audio_level += np.random.exponential(15)
        
        return (temperature, pm25, co2, audio_level)