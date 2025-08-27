"""
Model registry for the fire prediction system.

This package provides a central registry for all available models,
with functionality to discover, instantiate, and manage models.
"""

from .model_registry import ModelRegistry

__all__ = ['ModelRegistry']