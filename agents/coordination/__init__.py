"""
Coordination agents package for the synthetic fire prediction system.

This package provides implementations of coordination components for
multi-agent system orchestration and workflow management.
"""

from .multi_agent_coordinator import MultiAgentFireDetectionSystem, create_multi_agent_fire_system

__all__ = ['MultiAgentFireDetectionSystem', 'create_multi_agent_fire_system']