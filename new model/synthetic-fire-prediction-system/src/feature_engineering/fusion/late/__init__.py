"""
Late fusion techniques for the synthetic fire prediction system.

This package provides implementations of late fusion techniques that combine
features at the decision level or late in the feature extraction process.
"""

from .decision_level_fusion import DecisionLevelFusion
from .probability_fusion import ProbabilityFusion
from .ranking_fusion import RankingFusion
from .voting_fusion import VotingFusion

__all__ = [
    'DecisionLevelFusion',
    'ProbabilityFusion',
    'RankingFusion',
    'VotingFusion'
]