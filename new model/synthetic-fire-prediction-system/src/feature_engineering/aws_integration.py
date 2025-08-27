"""
AWS Integration for the feature extraction framework.

This module provides functionality for integrating the feature extraction framework
with AWS services such as Glue, S3, DynamoDB, Step Functions, and CloudWatch.
"""

from typing import Dict, Any, List, Optional, Union, BinaryIO
import os
import json
import logging
import boto3
import botocore
from datetime import datetime
import time
import uuid

# AWS integration (optional)
try:
    from ..aws.s3.service import S3ServiceImpl
    from ..aws.batch.service import BatchServiceImpl
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    S3ServiceImpl = None
    BatchServiceImpl = None


class AWSFeatureExtractionIntegration:
    """
    Integration with AWS services for feature extraction.
    
    This class provides functionality for using AWS Glue for feature extraction jobs,
    storing features in S3 and/or DynamoDB, using AWS Step Functions for orchestration,
    and implementing AWS CloudWatch for monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AWS integration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize AWS services
        self.s3_service = S3ServiceImpl(config.get('s3_config', {}))
        self.batch_service = BatchServiceImpl(config.get('batch_config', {}))
        
        # Initialize other AWS clients
        self.glue_client = None
        self.dynamodb_client = None
        self.dynamodb_resource = None
        self.step_functions_client = None
        self.cloudwatch_client = None
        
        # Validate configuration
        self._validate_config()
        
        # Initialize AWS clients
        self._initialize_aws_clients()
    
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required configuration sections
        required_sections = ['region_name']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration parameter: {section}")
    
    def _initialize_aws_clients(self) -> None:
        """
        Initialize AWS clients.
        """
        # Create a session
        session = boto3.Session(
            region_name=self.config['region_name'],
            profile_name=self.config.get('profile_name')
        )
        
        # Initialize clients
        if self.config.get('use_glue', False):
            self.glue_client = session.client('glue')
        
        if self.config.get('use_dynamodb', False):
            self.dynamodb_client = session.client('dynamodb')
            self.dynamodb_resource = session.resource('dynamodb')
        
        if self.config.get('use_step_functions', False):
            self.step_functions_client = session.client('stepfunctions')
        
        if self.config.get('use_cloudwatch', False):
            self.cloudwatch_client = session.client('cloudwatch')
    
    def create_glue_job(self, 
                       job_name: str, 
                       script_location: str,
                       role_arn: str,
                       arguments: Optional[Dict[str, str]] = None) -> str:
        """
        Create an AWS Glue job for feature extraction.
        
        Args:
            job_name: Name of the Glue job
            script_location: S3 location of the script
            role_arn: IAM role ARN for the job
            arguments: Optional arguments for the job
            
        Returns:
            Job name
        """
        if self.glue_client is None:
            raise ValueError("Glue client is not initialized")
        
        # Default job arguments
        default_args = {
            '--job-language': 'python',
            '--job-bookmark-option': 'job-bookmark-disable'
        }
        
        # Merge with provided arguments
        if arguments:
            default_args.update(arguments)
        
        # Create job
        try:
            response = self.glue_client.create_job(
                Name=job_name,
                Role=role_arn,
                Command={
                    'Name': 'glueetl',
                    'ScriptLocation': script_location,
                    'PythonVersion': '3'
                },
                DefaultArguments=default_args,
                GlueVersion='3.0',
                MaxRetries=2,
                Timeout=2880,  # 48 hours
                Tags={
                    'CreatedBy': 'FeatureExtractionFramework',
                    'CreatedAt': datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Created Glue job: {job_name}")
            return job_name
        
        except botocore.exceptions.ClientError as e:
            self.logger.error(f"Error creating Glue job: {str(e)}")
            raise
    
    def start_glue_job(self, 
                      job_name: str, 
                      arguments: Optional[Dict[str, str]] = None) -> str:
        """
        Start an AWS Glue job.
        
        Args:
            job_name: Name of the Glue job
            arguments: Optional arguments for the job run
            
        Returns:
            Job run ID
        """
        if self.glue_client is None:
            raise ValueError("Glue client is not initialized")
        
        try:
            response = self.glue_client.start_job_run(
                JobName=job_name,
                Arguments=arguments or {}
            )
            
            job_run_id = response['JobRunId']
            self.logger.info(f"Started Glue job {job_name} with run ID: {job_run_id}")
            return job_run_id
        
        except botocore.exceptions.ClientError as e:
            self.logger.error(f"Error starting Glue job: {str(e)}")
            raise
    
    def get_glue_job_status(self, job_name: str, job_run_id: str) -> Dict[str, Any]:
        """
        Get the status of a Glue job run.
        
        Args:
            job_name: Name of the Glue job
            job_run_id: Job run ID
            
        Returns:
            Dictionary containing job status information
        """
        if self.glue_client is None:
            raise ValueError("Glue client is not initialized")
        
        try:
            response = self.glue_client.get_job_run(
                JobName=job_name,
                RunId=job_run_id
            )
            
            return response['JobRun']
        
        except botocore.exceptions.ClientError as e:
            self.logger.error(f"Error getting Glue job status: {str(e)}")
            raise
    
    def wait_for_glue_job(self, 
                         job_name: str, 
                         job_run_id: str, 
                         poll_interval: int = 30,
                         timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Wait for a Glue job to complete.
        
        Args:
            job_name: Name of the Glue job
            job_run_id: Job run ID
            poll_interval: Polling interval in seconds
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary containing job status information
        """
        start_time = time.time()
        
        while True:
            # Check if timeout has been reached
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for Glue job {job_name} run {job_run_id} to complete")
            
            # Get job status
            job_run = self.get_glue_job_status(job_name, job_run_id)
            status = job_run['JobRunState']
            
            # Check if job has completed
            if status in ['SUCCEEDED', 'FAILED', 'STOPPED', 'TIMEOUT']:
                return job_run
            
            # Wait before polling again
            time.sleep(poll_interval)
    
    def create_dynamodb_table(self, 
                             table_name: str, 
                             key_schema: List[Dict[str, str]],
                             attribute_definitions: List[Dict[str, str]],
                             provisioned_throughput: Optional[Dict[str, int]] = None) -> str:
        """
        Create a DynamoDB table for storing features.
        
        Args:
            table_name: Name of the table
            key_schema: Key schema for the table
            attribute_definitions: Attribute definitions for the table
            provisioned_throughput: Optional provisioned throughput
            
        Returns:
            Table name
        """
        if self.dynamodb_client is None:
            raise ValueError("DynamoDB client is not initialized")
        
        # Default provisioned throughput
        if provisioned_throughput is None:
            provisioned_throughput = {
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        
        try:
            response = self.dynamodb_client.create_table(
                TableName=table_name,
                KeySchema=key_schema,
                AttributeDefinitions=attribute_definitions,
                ProvisionedThroughput=provisioned_throughput
            )
            
            # Wait for table to be created
            self.dynamodb_client.get_waiter('table_exists').wait(TableName=table_name)
            
            self.logger.info(f"Created DynamoDB table: {table_name}")
            return table_name
        
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ResourceInUseException':
                self.logger.info(f"DynamoDB table {table_name} already exists")
                return table_name
            else:
                self.logger.error(f"Error creating DynamoDB table: {str(e)}")
                raise
    
    def store_features_in_dynamodb(self, 
                                 table_name: str, 
                                 feature_id: str,
                                 features: Dict[str, Any],
                                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store features in DynamoDB.
        
        Args:
            table_name: Name of the DynamoDB table
            feature_id: Feature identifier
            features: Features to store
            metadata: Optional metadata
            
        Returns:
            True if successful, False otherwise
        """
        if self.dynamodb_resource is None:
            raise ValueError("DynamoDB resource is not initialized")
        
        try:
            table = self.dynamodb_resource.Table(table_name)
            
            # Prepare item
            item = {
                'feature_id': feature_id,
                'timestamp': datetime.now().isoformat(),
                'features': json.dumps(features)
            }
            
            # Add metadata if provided
            if metadata:
                item['metadata'] = json.dumps(metadata)
            
            # Store item
            table.put_item(Item=item)
            
            self.logger.info(f"Stored features in DynamoDB table {table_name} with ID: {feature_id}")
            return True
        
        except botocore.exceptions.ClientError as e:
            self.logger.error(f"Error storing features in DynamoDB: {str(e)}")
            return False
    
    def retrieve_features_from_dynamodb(self, 
                                      table_name: str, 
                                      feature_id: str) -> Dict[str, Any]:
        """
        Retrieve features from DynamoDB.
        
        Args:
            table_name: Name of the DynamoDB table
            feature_id: Feature identifier
            
        Returns:
            Dictionary containing features and metadata
        """
        if self.dynamodb_resource is None:
            raise ValueError("DynamoDB resource is not initialized")
        
        try:
            table = self.dynamodb_resource.Table(table_name)
            
            # Get item
            response = table.get_item(Key={'feature_id': feature_id})
            
            if 'Item' not in response:
                raise ValueError(f"Features not found for ID: {feature_id}")
            
            item = response['Item']
            
            # Parse features and metadata
            features = json.loads(item['features'])
            result = {
                'feature_id': feature_id,
                'timestamp': item.get('timestamp'),
                'features': features
            }
            
            # Add metadata if available
            if 'metadata' in item:
                result['metadata'] = json.loads(item['metadata'])
            
            return result
        
        except botocore.exceptions.ClientError as e:
            self.logger.error(f"Error retrieving features from DynamoDB: {str(e)}")
            raise
    
    def create_step_function(self, 
                           state_machine_name: str, 
                           definition: Dict[str, Any],
                           role_arn: str) -> str:
        """
        Create an AWS Step Function for orchestration.
        
        Args:
            state_machine_name: Name of the state machine
            definition: State machine definition
            role_arn: IAM role ARN for the state machine
            
        Returns:
            State machine ARN
        """
        if self.step_functions_client is None:
            raise ValueError("Step Functions client is not initialized")
        
        try:
            response = self.step_functions_client.create_state_machine(
                name=state_machine_name,
                definition=json.dumps(definition),
                roleArn=role_arn,
                type='STANDARD',
                tags=[
                    {
                        'key': 'CreatedBy',
                        'value': 'FeatureExtractionFramework'
                    },
                    {
                        'key': 'CreatedAt',
                        'value': datetime.now().isoformat()
                    }
                ]
            )
            
            self.logger.info(f"Created Step Function: {state_machine_name}")
            return response['stateMachineArn']
        
        except botocore.exceptions.ClientError as e:
            self.logger.error(f"Error creating Step Function: {str(e)}")
            raise
    
    def start_step_function_execution(self, 
                                    state_machine_arn: str, 
                                    input_data: Dict[str, Any],
                                    name: Optional[str] = None) -> str:
        """
        Start a Step Function execution.
        
        Args:
            state_machine_arn: ARN of the state machine
            input_data: Input data for the execution
            name: Optional name for the execution
            
        Returns:
            Execution ARN
        """
        if self.step_functions_client is None:
            raise ValueError("Step Functions client is not initialized")
        
        # Generate execution name if not provided
        if name is None:
            name = f"execution-{uuid.uuid4()}"
        
        try:
            response = self.step_functions_client.start_execution(
                stateMachineArn=state_machine_arn,
                name=name,
                input=json.dumps(input_data)
            )
            
            self.logger.info(f"Started Step Function execution: {name}")
            return response['executionArn']
        
        except botocore.exceptions.ClientError as e:
            self.logger.error(f"Error starting Step Function execution: {str(e)}")
            raise
    
    def get_step_function_execution_status(self, execution_arn: str) -> Dict[str, Any]:
        """
        Get the status of a Step Function execution.
        
        Args:
            execution_arn: Execution ARN
            
        Returns:
            Dictionary containing execution status information
        """
        if self.step_functions_client is None:
            raise ValueError("Step Functions client is not initialized")
        
        try:
            response = self.step_functions_client.describe_execution(
                executionArn=execution_arn
            )
            
            return response
        
        except botocore.exceptions.ClientError as e:
            self.logger.error(f"Error getting Step Function execution status: {str(e)}")
            raise
    
    def wait_for_step_function_execution(self, 
                                       execution_arn: str, 
                                       poll_interval: int = 30,
                                       timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Wait for a Step Function execution to complete.
        
        Args:
            execution_arn: Execution ARN
            poll_interval: Polling interval in seconds
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary containing execution status information
        """
        start_time = time.time()
        
        while True:
            # Check if timeout has been reached
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for Step Function execution {execution_arn} to complete")
            
            # Get execution status
            execution = self.get_step_function_execution_status(execution_arn)
            status = execution['status']
            
            # Check if execution has completed
            if status in ['SUCCEEDED', 'FAILED', 'TIMED_OUT', 'ABORTED']:
                return execution
            
            # Wait before polling again
            time.sleep(poll_interval)
    
    def put_cloudwatch_metric(self, 
                            namespace: str,
                            metric_name: str,
                            value: float,
                            unit: str,
                            dimensions: Optional[List[Dict[str, str]]] = None) -> bool:
        """
        Put a metric data point in CloudWatch.
        
        Args:
            namespace: Metric namespace
            metric_name: Metric name
            value: Metric value
            unit: Metric unit
            dimensions: Optional metric dimensions
            
        Returns:
            True if successful, False otherwise
        """
        if self.cloudwatch_client is None:
            raise ValueError("CloudWatch client is not initialized")
        
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.now()
            }
            
            if dimensions:
                metric_data['Dimensions'] = dimensions
            
            self.cloudwatch_client.put_metric_data(
                Namespace=namespace,
                MetricData=[metric_data]
            )
            
            self.logger.debug(f"Put CloudWatch metric: {namespace}/{metric_name} = {value} {unit}")
            return True
        
        except botocore.exceptions.ClientError as e:
            self.logger.error(f"Error putting CloudWatch metric: {str(e)}")
            return False
    
    def create_cloudwatch_alarm(self, 
                              alarm_name: str,
                              namespace: str,
                              metric_name: str,
                              comparison_operator: str,
                              threshold: float,
                              evaluation_periods: int,
                              period: int,
                              statistic: str,
                              dimensions: Optional[List[Dict[str, str]]] = None,
                              alarm_actions: Optional[List[str]] = None) -> str:
        """
        Create a CloudWatch alarm.
        
        Args:
            alarm_name: Name of the alarm
            namespace: Metric namespace
            metric_name: Metric name
            comparison_operator: Comparison operator
            threshold: Alarm threshold
            evaluation_periods: Number of periods to evaluate
            period: Period in seconds
            statistic: Statistic to apply
            dimensions: Optional metric dimensions
            alarm_actions: Optional actions to take when alarm triggers
            
        Returns:
            Alarm ARN
        """
        if self.cloudwatch_client is None:
            raise ValueError("CloudWatch client is not initialized")
        
        try:
            alarm_args = {
                'AlarmName': alarm_name,
                'AlarmDescription': f"Alarm for {namespace}/{metric_name}",
                'ActionsEnabled': True,
                'MetricName': metric_name,
                'Namespace': namespace,
                'Statistic': statistic,
                'Period': period,
                'EvaluationPeriods': evaluation_periods,
                'Threshold': threshold,
                'ComparisonOperator': comparison_operator
            }
            
            if dimensions:
                alarm_args['Dimensions'] = dimensions
            
            if alarm_actions:
                alarm_args['AlarmActions'] = alarm_actions
            
            response = self.cloudwatch_client.put_metric_alarm(**alarm_args)
            
            self.logger.info(f"Created CloudWatch alarm: {alarm_name}")
            return alarm_name
        
        except botocore.exceptions.ClientError as e:
            self.logger.error(f"Error creating CloudWatch alarm: {str(e)}")
            raise
    
    def setup_feature_extraction_monitoring(self, 
                                          namespace: str = 'FeatureExtraction',
                                          create_alarms: bool = True) -> Dict[str, Any]:
        """
        Set up CloudWatch monitoring for feature extraction.
        
        Args:
            namespace: CloudWatch namespace
            create_alarms: Whether to create alarms
            
        Returns:
            Dictionary containing monitoring information
        """
        if self.cloudwatch_client is None:
            raise ValueError("CloudWatch client is not initialized")
        
        monitoring_info = {
            'namespace': namespace,
            'metrics': [
                {
                    'name': 'ProcessingTime',
                    'unit': 'Milliseconds',
                    'description': 'Time taken to process features'
                },
                {
                    'name': 'FeatureCount',
                    'unit': 'Count',
                    'description': 'Number of features extracted'
                },
                {
                    'name': 'ErrorCount',
                    'unit': 'Count',
                    'description': 'Number of errors during feature extraction'
                },
                {
                    'name': 'MemoryUsage',
                    'unit': 'Megabytes',
                    'description': 'Memory usage during feature extraction'
                }
            ],
            'alarms': []
        }
        
        # Create alarms if requested
        if create_alarms:
            # Create alarm for high error count
            error_alarm_name = f"{namespace}-HighErrorCount"
            self.create_cloudwatch_alarm(
                alarm_name=error_alarm_name,
                namespace=namespace,
                metric_name='ErrorCount',
                comparison_operator='GreaterThanThreshold',
                threshold=5,
                evaluation_periods=1,
                period=60,
                statistic='Sum'
            )
            monitoring_info['alarms'].append(error_alarm_name)
            
            # Create alarm for high processing time
            time_alarm_name = f"{namespace}-HighProcessingTime"
            self.create_cloudwatch_alarm(
                alarm_name=time_alarm_name,
                namespace=namespace,
                metric_name='ProcessingTime',
                comparison_operator='GreaterThanThreshold',
                threshold=10000,  # 10 seconds
                evaluation_periods=3,
                period=60,
                statistic='Average'
            )
            monitoring_info['alarms'].append(time_alarm_name)
        
        self.logger.info(f"Set up CloudWatch monitoring for feature extraction in namespace: {namespace}")
        return monitoring_info