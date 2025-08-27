"""
AWS service integration for the synthetic fire prediction system.

This package provides functionality for integrating with AWS services
such as S3, SageMaker, Lambda, CloudWatch, and Batch.
"""

from .base import AWSServiceBase, S3Service, SageMakerService, LambdaService, CloudWatchService, BatchService
from .s3.service import S3ServiceImpl
from .batch.service import BatchServiceImpl

__all__ = [
    'AWSServiceBase',
    'S3Service',
    'SageMakerService',
    'LambdaService',
    'CloudWatchService',
    'BatchService',
    'S3ServiceImpl',
    'BatchServiceImpl'
]