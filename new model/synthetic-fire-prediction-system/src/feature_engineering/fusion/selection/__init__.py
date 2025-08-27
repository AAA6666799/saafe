"""
Feature selection mechanisms for the synthetic fire prediction system.

This package provides implementations of feature selection techniques that
select the most relevant features for the fire prediction task.
"""

from .correlation_based_selection import CorrelationBasedSelection
from .mutual_information_selection import MutualInformationSelection
from .principal_component_analysis import PrincipalComponentAnalysis
from .recursive_feature_elimination import RecursiveFeatureElimination
from .genetic_algorithm_selection import GeneticAlgorithmSelection

__all__ = [
    'CorrelationBasedSelection',
    'MutualInformationSelection',
    'PrincipalComponentAnalysis',
    'RecursiveFeatureElimination',
    'GeneticAlgorithmSelection'
]