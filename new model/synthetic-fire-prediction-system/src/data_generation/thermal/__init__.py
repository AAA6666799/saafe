"""
Thermal data generation package.

This package provides components for generating synthetic thermal images
for fire prediction system training and testing.
"""

from .thermal_image_generator import ThermalImageGenerator
from .hotspot_simulator import HotspotSimulator, HotspotShape, GrowthPattern
from .temporal_evolution_model import TemporalEvolutionModel, FireType, FireStage
from .noise_injector import NoiseInjector, NoiseType

__all__ = [
    'ThermalImageGenerator',
    'HotspotSimulator',
    'HotspotShape',
    'GrowthPattern',
    'TemporalEvolutionModel',
    'FireType',
    'FireStage',
    'NoiseInjector',
    'NoiseType'
]