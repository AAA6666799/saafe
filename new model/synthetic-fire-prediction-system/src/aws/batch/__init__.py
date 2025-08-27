"""
AWS Batch service integration.

This package provides functionality for submitting and managing jobs on AWS Batch
for parallel dataset generation.
"""

from .service import BatchService, BatchServiceImpl

__all__ = ['BatchService', 'BatchServiceImpl']