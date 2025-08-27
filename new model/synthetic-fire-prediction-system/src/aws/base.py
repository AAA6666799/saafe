"""
Base interfaces for AWS service integration.

This module defines the core interfaces and abstract classes for AWS service integration
in the synthetic fire prediction system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, BinaryIO
import boto3
import botocore
import json
import os
from datetime import datetime


class AWSServiceBase(ABC):
    """
    Base abstract class for all AWS service integrations.
    
    This class defines the common interface that all AWS service integrations must implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AWS service integration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.session = None
        self.client = None
        self.resource = None
        self.validate_config()
        self.initialize_session()
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    def initialize_session(self) -> None:
        """
        Initialize the AWS session using the provided configuration.
        """
        session_kwargs = {}
        
        if 'profile_name' in self.config:
            session_kwargs['profile_name'] = self.config['profile_name']
        
        if 'region_name' in self.config:
            session_kwargs['region_name'] = self.config['region_name']
        
        if 'aws_access_key_id' in self.config and 'aws_secret_access_key' in self.config:
            session_kwargs['aws_access_key_id'] = self.config['aws_access_key_id']
            session_kwargs['aws_secret_access_key'] = self.config['aws_secret_access_key']
            
            if 'aws_session_token' in self.config:
                session_kwargs['aws_session_token'] = self.config['aws_session_token']
        
        self.session = boto3.Session(**session_kwargs)
    
    @abstractmethod
    def initialize_client(self) -> None:
        """
        Initialize the AWS service client.
        """
        pass
    
    @abstractmethod
    def initialize_resource(self) -> None:
        """
        Initialize the AWS service resource (if applicable).
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test the connection to the AWS service.
        
        Returns:
            True if connection is successful, False otherwise
        """
        pass


class S3Service(AWSServiceBase):
    """
    Interface for Amazon S3 service integration.
    """
    
    def initialize_client(self) -> None:
        """
        Initialize the S3 client.
        """
        self.client = self.session.client('s3')
    
    def initialize_resource(self) -> None:
        """
        Initialize the S3 resource.
        """
        self.resource = self.session.resource('s3')
    
    def validate_config(self) -> None:
        """
        Validate the S3 configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_params = ['default_bucket']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
    
    def test_connection(self) -> bool:
        """
        Test the connection to S3.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.client.list_buckets()
            return True
        except botocore.exceptions.ClientError:
            return False
    
    @abstractmethod
    def upload_file(self, 
                   local_path: str, 
                   s3_key: str, 
                   bucket: Optional[str] = None,
                   extra_args: Optional[Dict[str, Any]] = None) -> bool:
        """
        Upload a file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 object key
            bucket: Optional bucket name (uses default_bucket if not provided)
            extra_args: Optional extra arguments for upload
            
        Returns:
            True if upload is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def download_file(self, 
                     s3_key: str, 
                     local_path: str, 
                     bucket: Optional[str] = None) -> bool:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local file path
            bucket: Optional bucket name (uses default_bucket if not provided)
            
        Returns:
            True if download is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def list_objects(self, 
                    prefix: str, 
                    bucket: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List objects in an S3 bucket.
        
        Args:
            prefix: Prefix to filter objects
            bucket: Optional bucket name (uses default_bucket if not provided)
            
        Returns:
            List of object information dictionaries
        """
        pass


class SageMakerService(AWSServiceBase):
    """
    Interface for Amazon SageMaker service integration.
    """
    
    def initialize_client(self) -> None:
        """
        Initialize the SageMaker client.
        """
        self.client = self.session.client('sagemaker')
    
    def initialize_resource(self) -> None:
        """
        Initialize the SageMaker resource (not applicable).
        """
        pass  # SageMaker doesn't have a resource interface
    
    def validate_config(self) -> None:
        """
        Validate the SageMaker configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_params = ['role_arn']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
    
    def test_connection(self) -> bool:
        """
        Test the connection to SageMaker.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.client.list_training_jobs(MaxResults=1)
            return True
        except botocore.exceptions.ClientError:
            return False
    
    @abstractmethod
    def create_training_job(self, 
                           job_name: str,
                           algorithm_specification: Dict[str, Any],
                           input_data_config: List[Dict[str, Any]],
                           output_data_config: Dict[str, Any],
                           resource_config: Dict[str, Any],
                           stopping_condition: Dict[str, Any],
                           hyperparameters: Optional[Dict[str, str]] = None) -> str:
        """
        Create a SageMaker training job.
        
        Args:
            job_name: Name of the training job
            algorithm_specification: Algorithm specification
            input_data_config: Input data configuration
            output_data_config: Output data configuration
            resource_config: Resource configuration
            stopping_condition: Stopping condition
            hyperparameters: Optional hyperparameters
            
        Returns:
            ARN of the created training job
        """
        pass
    
    @abstractmethod
    def deploy_model(self, 
                    model_name: str,
                    model_data_url: str,
                    image_uri: str,
                    instance_type: str,
                    initial_instance_count: int,
                    endpoint_name: Optional[str] = None,
                    environment: Optional[Dict[str, str]] = None) -> str:
        """
        Deploy a model to a SageMaker endpoint.
        
        Args:
            model_name: Name of the model
            model_data_url: URL of the model data
            image_uri: URI of the container image
            instance_type: Type of instance to deploy to
            initial_instance_count: Initial number of instances
            endpoint_name: Optional endpoint name
            environment: Optional environment variables
            
        Returns:
            ARN of the created endpoint
        """
        pass
    
    @abstractmethod
    def invoke_endpoint(self, 
                       endpoint_name: str,
                       data: Union[str, bytes],
                       content_type: str,
                       accept: str) -> Dict[str, Any]:
        """
        Invoke a SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            data: Input data
            content_type: Content type of the input
            accept: Expected content type of the output
            
        Returns:
            Response from the endpoint
        """
        pass


class LambdaService(AWSServiceBase):
    """
    Interface for AWS Lambda service integration.
    """
    
    def initialize_client(self) -> None:
        """
        Initialize the Lambda client.
        """
        self.client = self.session.client('lambda')
    
    def initialize_resource(self) -> None:
        """
        Initialize the Lambda resource (not applicable).
        """
        pass  # Lambda doesn't have a resource interface
    
    def validate_config(self) -> None:
        """
        Validate the Lambda configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass  # No specific required parameters for basic Lambda operations
    
    def test_connection(self) -> bool:
        """
        Test the connection to Lambda.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.client.list_functions(MaxItems=1)
            return True
        except botocore.exceptions.ClientError:
            return False
    
    @abstractmethod
    def invoke_function(self, 
                       function_name: str,
                       payload: Dict[str, Any],
                       invocation_type: str = 'RequestResponse') -> Dict[str, Any]:
        """
        Invoke a Lambda function.
        
        Args:
            function_name: Name or ARN of the function
            payload: Input payload
            invocation_type: Type of invocation (RequestResponse, Event, DryRun)
            
        Returns:
            Response from the function
        """
        pass
    
    @abstractmethod
    def create_function(self, 
                       function_name: str,
                       runtime: str,
                       role: str,
                       handler: str,
                       code: Dict[str, Any],
                       memory_size: int = 128,
                       timeout: int = 3,
                       environment: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Create a Lambda function.
        
        Args:
            function_name: Name of the function
            runtime: Runtime identifier
            role: ARN of the execution role
            handler: Function handler
            code: Function code
            memory_size: Memory size in MB
            timeout: Timeout in seconds
            environment: Optional environment variables
            
        Returns:
            Information about the created function
        """
        pass


class CloudWatchService(AWSServiceBase):
    """
    Interface for Amazon CloudWatch service integration.
    """
    
    def initialize_client(self) -> None:
        """
        Initialize the CloudWatch client.
        """
        self.client = self.session.client('cloudwatch')
        self.logs_client = self.session.client('logs')
    
    def initialize_resource(self) -> None:
        """
        Initialize the CloudWatch resource (not applicable).
        """
        pass  # CloudWatch doesn't have a resource interface
    
    def validate_config(self) -> None:
        """
        Validate the CloudWatch configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass  # No specific required parameters for basic CloudWatch operations
    
    def test_connection(self) -> bool:
        """
        Test the connection to CloudWatch.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.client.list_metrics(MetricName='CPUUtilization', Namespace='AWS/EC2')
            return True
        except botocore.exceptions.ClientError:
            return False


class BatchService(AWSServiceBase):
    """
    Interface for AWS Batch service integration.
    """
    
    def initialize_client(self) -> None:
        """
        Initialize the Batch client.
        """
        self.client = self.session.client('batch')
    
    def initialize_resource(self) -> None:
        """
        Initialize the Batch resource (not applicable).
        """
        pass  # Batch doesn't have a resource interface
    
    def validate_config(self) -> None:
        """
        Validate the Batch configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_params = ['job_queue']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
    
    def test_connection(self) -> bool:
        """
        Test the connection to Batch.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.client.describe_job_queues(jobQueues=[self.config['job_queue']])
            return True
        except botocore.exceptions.ClientError:
            return False
    
    @abstractmethod
    def submit_job(self,
                  job_name: str,
                  job_definition: str,
                  job_queue: Optional[str] = None,
                  container_overrides: Optional[Dict[str, Any]] = None,
                  parameters: Optional[Dict[str, str]] = None,
                  depends_on: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Submit a job to AWS Batch.
        
        Args:
            job_name: Name of the job
            job_definition: Job definition name or ARN
            job_queue: Optional job queue name or ARN (uses default from config if not provided)
            container_overrides: Optional container overrides
            parameters: Optional job parameters
            depends_on: Optional list of job dependencies
            
        Returns:
            Response from the submit_job API call
        """
        pass
    
    @abstractmethod
    def describe_job(self, job_id: str) -> Dict[str, Any]:
        """
        Describe a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job description
        """
        pass
    
    @abstractmethod
    def list_jobs(self,
                 job_queue: Optional[str] = None,
                 job_status: Optional[str] = None,
                 max_results: int = 100) -> List[Dict[str, Any]]:
        """
        List jobs in a job queue.
        
        Args:
            job_queue: Optional job queue name or ARN (uses default from config if not provided)
            job_status: Optional job status filter
            max_results: Maximum number of results to return
            
        Returns:
            List of jobs
        """
        pass
    
    @abstractmethod
    def put_metric_data(self, 
                       namespace: str,
                       metric_name: str,
                       value: float,
                       unit: str,
                       dimensions: Optional[List[Dict[str, str]]] = None,
                       timestamp: Optional[datetime] = None) -> bool:
        """
        Put metric data to CloudWatch.
        
        Args:
            namespace: Metric namespace
            metric_name: Metric name
            value: Metric value
            unit: Metric unit
            dimensions: Optional metric dimensions
            timestamp: Optional timestamp
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def create_alarm(self, 
                    alarm_name: str,
                    metric_name: str,
                    namespace: str,
                    statistic: str,
                    period: int,
                    evaluation_periods: int,
                    threshold: float,
                    comparison_operator: str,
                    alarm_actions: Optional[List[str]] = None,
                    dimensions: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Create a CloudWatch alarm.
        
        Args:
            alarm_name: Name of the alarm
            metric_name: Name of the metric
            namespace: Metric namespace
            statistic: Statistic to apply
            period: Period in seconds
            evaluation_periods: Number of periods to evaluate
            threshold: Alarm threshold
            comparison_operator: Comparison operator
            alarm_actions: Optional actions to take when alarm triggers
            dimensions: Optional metric dimensions
            
        Returns:
            Information about the created alarm
        """
        pass
    
    @abstractmethod
    def put_log_events(self, 
                      log_group_name: str,
                      log_stream_name: str,
                      events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Put log events to CloudWatch Logs.
        
        Args:
            log_group_name: Name of the log group
            log_stream_name: Name of the log stream
            events: List of log events
            
        Returns:
            Response from the API
        """
        pass