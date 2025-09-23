"""
Hardware specific implementations.

This module contains concrete implementations of hardware interfaces
for specific sensor models.
"""

from .flir_lepton35_interface import FLIRLepton35Interface, create_flir_lepton35_interface
from .scd41_interface import SCD41Interface, create_scd41_interface
from .synthetic_flir_interface import SyntheticFLIRInterface, create_synthetic_flir_interface
from .synthetic_scd41_interface import SyntheticSCD41Interface, create_synthetic_scd41_interface
from .mlx90640_interface import MLX90640Interface, create_mlx90640_interface

__all__ = [
    'FLIRLepton35Interface',
    'create_flir_lepton35_interface',
    'SCD41Interface',
    'create_scd41_interface',
    'SyntheticFLIRInterface',
    'create_synthetic_flir_interface',
    'SyntheticSCD41Interface',
    'create_synthetic_scd41_interface',
    'MLX90640Interface',
    'create_mlx90640_interface'
]