"""
Scenario generation package for synthetic fire data.

This package provides functionality for generating complete fire scenarios
that combine thermal, gas, and environmental data.
"""

from .scenario_generator import ScenarioGenerator
from .specific_generators import (
    SpecificScenarioGenerator,
    NormalScenarioGenerator,
    ElectricalFireScenarioGenerator,
    ChemicalFireScenarioGenerator,
    SmolderingFireScenarioGenerator,
    RapidCombustionScenarioGenerator
)
from .false_positive_generator import FalsePositiveGenerator
from .scenario_mixer import ScenarioMixer

__all__ = [
    'ScenarioGenerator',
    'SpecificScenarioGenerator',
    'NormalScenarioGenerator',
    'ElectricalFireScenarioGenerator',
    'ChemicalFireScenarioGenerator',
    'SmolderingFireScenarioGenerator',
    'RapidCombustionScenarioGenerator',
    'FalsePositiveGenerator',
    'ScenarioMixer'
]