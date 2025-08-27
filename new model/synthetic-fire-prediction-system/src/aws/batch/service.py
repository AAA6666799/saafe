"""
Amazon Batch service implementation.

This module provides concrete implementations of the BatchService interface
for submitting and managing jobs on Amazon Batch.
"""

from typing import Dict, Any, List, Optional
import boto3
import botocore
import time
import logging
from datetime import datetime

from ..base import AWSServiceBase


class BatchService(AWSServiceBase):
    """
    Abstract base class for AWS Batch service integration.
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
        # Use default job queue from config if not provided
        if job_queue is None:
            job_queue = self.config['job_queue']
        
        # Prepare submit job request
        submit_job_kwargs = {
            'jobName': job_name,
            'jobQueue': job_queue,
            'jobDefinition': job_definition
        }
        
        # Add container overrides if provided
        if container_overrides is not None:
            submit_job_kwargs['containerOverrides'] = container_overrides
        
        # Add parameters if provided
        if parameters is not None:
            submit_job_kwargs['parameters'] = parameters
        
        # Add dependencies if provided
        if depends_on is not None:
            submit_job_kwargs['dependsOn'] = depends_on
        
        # Submit job
        try:
            response = self.client.submit_job(**submit_job_kwargs)
            return response
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error submitting job to AWS Batch: {e}")
            raise
    
    def describe_job(self, job_id: str) -> Dict[str, Any]:
        """
        Describe a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job description
        """
        try:
            response = self.client.describe_jobs(jobs=[job_id])
            if 'jobs' in response and len(response['jobs']) > 0:
                return response['jobs'][0]
            else:
                raise ValueError(f"Job not found: {job_id}")
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error describing job: {e}")
            raise
    
    def terminate_job(self, job_id: str, reason: str) -> Dict[str, Any]:
        """
        Terminate a job.
        
        Args:
            job_id: Job ID
            reason: Reason for termination
            
        Returns:
            Response from the terminate_job API call
        """
        try:
            response = self.client.terminate_job(jobId=job_id, reason=reason)
            return response
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error terminating job: {e}")
            raise
    
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
        # Use default job queue from config if not provided
        if job_queue is None:
            job_queue = self.config['job_queue']
        
        # Prepare list jobs request
        list_jobs_kwargs = {
            'jobQueue': job_queue,
            'maxResults': max_results
        }
        
        # Add job status filter if provided
        if job_status is not None:
            list_jobs_kwargs['jobStatus'] = job_status
        
        try:
            response = self.client.list_jobs(**list_jobs_kwargs)
            return response.get('jobSummaryList', [])
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error listing jobs: {e}")
            raise
    
    def wait_for_job_completion(self, 
                              job_id: str,
                              poll_interval: int = 30,
                              timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Wait for a job to complete.
        
        Args:
            job_id: Job ID
            poll_interval: Polling interval in seconds
            timeout: Optional timeout in seconds
            
        Returns:
            Final job description
        """
        start_time = time.time()
        while True:
            # Check if timeout has been reached
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for job {job_id} to complete")
            
            # Get job description
            job = self.describe_job(job_id)
            status = job.get('status')
            
            # Check if job has completed
            if status in ['SUCCEEDED', 'FAILED']:
                return job
            
            # Wait before polling again
            time.sleep(poll_interval)
    
    def create_compute_environment(self,
                                 compute_environment_name: str,
                                 compute_resources: Dict[str, Any],
                                 service_role: str,
                                 state: str = 'ENABLED',
                                 tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create a compute environment.
        
        Args:
            compute_environment_name: Name of the compute environment
            compute_resources: Compute resources configuration
            service_role: Service role ARN
            state: State of the compute environment
            tags: Optional tags
            
        Returns:
            Response from the create_compute_environment API call
        """
        # Prepare create compute environment request
        create_env_kwargs = {
            'computeEnvironmentName': compute_environment_name,
            'type': 'MANAGED',
            'state': state,
            'computeResources': compute_resources,
            'serviceRole': service_role
        }
        
        # Add tags if provided
        if tags is not None:
            create_env_kwargs['tags'] = tags
        
        try:
            response = self.client.create_compute_environment(**create_env_kwargs)
            return response
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error creating compute environment: {e}")
            raise
    
    def create_job_queue(self,
                        job_queue_name: str,
                        compute_environment_order: List[Dict[str, Any]],
                        priority: int = 1,
                        state: str = 'ENABLED',
                        tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create a job queue.
        
        Args:
            job_queue_name: Name of the job queue
            compute_environment_order: List of compute environments and their order
            priority: Priority of the job queue
            state: State of the job queue
            tags: Optional tags
            
        Returns:
            Response from the create_job_queue API call
        """
        # Prepare create job queue request
        create_queue_kwargs = {
            'jobQueueName': job_queue_name,
            'state': state,
            'priority': priority,
            'computeEnvironmentOrder': compute_environment_order
        }
        
        # Add tags if provided
        if tags is not None:
            create_queue_kwargs['tags'] = tags
        
        try:
            response = self.client.create_job_queue(**create_queue_kwargs)
            return response
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error creating job queue: {e}")
            raise
    
    def register_job_definition(self,
                              job_definition_name: str,
                              container_properties: Dict[str, Any],
                              type: str = 'container',
                              parameters: Optional[Dict[str, str]] = None,
                              tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Register a job definition.
        
        Args:
            job_definition_name: Name of the job definition
            container_properties: Container properties
            type: Type of job definition
            parameters: Optional parameters
            tags: Optional tags
            
        Returns:
            Response from the register_job_definition API call
        """
        # Prepare register job definition request
        register_job_def_kwargs = {
            'jobDefinitionName': job_definition_name,
            'type': type,
            'containerProperties': container_properties
        }
        
        # Add parameters if provided
        if parameters is not None:
            register_job_def_kwargs['parameters'] = parameters
        
        # Add tags if provided
        if tags is not None:
            register_job_def_kwargs['tags'] = tags
        
        try:
            response = self.client.register_job_definition(**register_job_def_kwargs)
            return response
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error registering job definition: {e}")
            raise


class BatchServiceImpl(BatchService):
    """
    Implementation of the BatchService interface for AWS Batch operations.
    
    This class provides concrete implementations of the abstract methods
    defined in the BatchService base class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Batch service implementation.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
    
    def submit_dataset_generation_job(self,
                                    job_name: str,
                                    output_dir: str,
                                    job_type: str,
                                    job_params: Dict[str, Any],
                                    depends_on: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Submit a dataset generation job to AWS Batch.
        
        Args:
            job_name: Name of the job
            output_dir: Output directory for the generated data
            job_type: Type of job ('normal', 'fire', 'false_positive')
            job_params: Job-specific parameters
            depends_on: Optional list of job dependencies
            
        Returns:
            Response from the submit_job API call
        """
        # Get job definition from config
        job_definition = self.config.get('job_definition', 'fire-prediction-dataset-generation')
        
        # Prepare container overrides based on job type
        container_overrides = {
            'command': []
        }
        
        if job_type == 'normal':
            container_overrides['command'] = [
                'python', '-m', 'src.data_generation.batch_jobs.generate_normal_data',
                '--output-dir', output_dir,
                '--num-hours', str(job_params.get('num_hours', 1000)),
                '--seed', str(job_params.get('seed', 42))
            ]
        elif job_type == 'fire':
            container_overrides['command'] = [
                'python', '-m', 'src.data_generation.batch_jobs.generate_fire_data',
                '--output-dir', output_dir,
                '--fire-type', job_params.get('fire_type', 'electrical'),
                '--num-scenarios', str(job_params.get('num_scenarios', 100)),
                '--seed', str(job_params.get('seed', 42))
            ]
        elif job_type == 'false_positive':
            container_overrides['command'] = [
                'python', '-m', 'src.data_generation.batch_jobs.generate_false_positive_data',
                '--output-dir', output_dir,
                '--false-positive-type', job_params.get('false_positive_type', 'cooking'),
                '--num-scenarios', str(job_params.get('num_scenarios', 50)),
                '--seed', str(job_params.get('seed', 42))
            ]
        else:
            raise ValueError(f"Unsupported job type: {job_type}")
        
        # Add S3 bucket if available
        if 'default_bucket' in self.config:
            container_overrides['command'].extend(['--s3-bucket', self.config['default_bucket']])
        
        # Add environment variables if provided
        if 'environment' in self.config:
            container_overrides['environment'] = [
                {'name': name, 'value': value}
                for name, value in self.config['environment'].items()
            ]
        
        # Submit job
        response = self.submit_job(
            job_name=job_name,
            job_definition=job_definition,
            container_overrides=container_overrides,
            depends_on=depends_on
        )
        
        self.logger.info(f"Submitted {job_type} dataset generation job: {job_name} (ID: {response.get('jobId')})")
        return response
    
    def submit_complete_dataset_generation(self,
                                         base_job_name: str,
                                         output_dir: str,
                                         num_normal_hours: int = 1000,
                                         num_fire_scenarios: int = 100,
                                         num_false_positive_scenarios: int = 50,
                                         seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Submit a complete set of dataset generation jobs to AWS Batch.
        
        Args:
            base_job_name: Base name for the jobs
            output_dir: Base output directory for the generated data
            num_normal_hours: Number of hours of normal operation data to generate
            num_fire_scenarios: Number of fire scenarios to generate per fire type
            num_false_positive_scenarios: Number of false positive scenarios to generate per type
            seed: Optional random seed
            
        Returns:
            Dictionary with information about the submitted jobs
        """
        # Generate timestamp for job names
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Set random seed if not provided
        if seed is None:
            seed = int(timestamp) % 10000
        
        # Submit normal operation data generation job
        normal_job_name = f"{base_job_name}-normal-{timestamp}"
        normal_job = self.submit_dataset_generation_job(
            job_name=normal_job_name,
            output_dir=f"{output_dir}/normal",
            job_type='normal',
            job_params={
                'num_hours': num_normal_hours,
                'seed': seed
            }
        )
        
        # Submit fire scenario generation jobs
        fire_jobs = {}
        fire_types = ['electrical', 'chemical', 'smoldering', 'rapid_combustion']
        for fire_type in fire_types:
            job_name = f"{base_job_name}-{fire_type}-{timestamp}"
            job = self.submit_dataset_generation_job(
                job_name=job_name,
                output_dir=f"{output_dir}/{fire_type}",
                job_type='fire',
                job_params={
                    'fire_type': fire_type,
                    'num_scenarios': num_fire_scenarios,
                    'seed': seed + hash(fire_type) % 10000
                }
            )
            fire_jobs[fire_type] = {
                'job_id': job.get('jobId'),
                'job_name': job_name
            }
        
        # Submit false positive scenario generation jobs
        fp_jobs = {}
        fp_types = ['cooking', 'heating', 'other']
        for fp_type in fp_types:
            job_name = f"{base_job_name}-fp-{fp_type}-{timestamp}"
            job = self.submit_dataset_generation_job(
                job_name=job_name,
                output_dir=f"{output_dir}/false_positive_{fp_type}",
                job_type='false_positive',
                job_params={
                    'false_positive_type': fp_type,
                    'num_scenarios': num_false_positive_scenarios,
                    'seed': seed + hash(fp_type) % 10000
                }
            )
            fp_jobs[fp_type] = {
                'job_id': job.get('jobId'),
                'job_name': job_name
            }
        
        # Return information about all submitted jobs
        return {
            'base_job_name': base_job_name,
            'timestamp': timestamp,
            'output_dir': output_dir,
            'normal_job': {
                'job_id': normal_job.get('jobId'),
                'job_name': normal_job_name
            },
            'fire_jobs': fire_jobs,
            'false_positive_jobs': fp_jobs
        }
    
    def monitor_dataset_generation_jobs(self, jobs_info: Dict[str, Any], poll_interval: int = 60) -> Dict[str, Any]:
        """
        Monitor the status of dataset generation jobs.
        
        Args:
            jobs_info: Information about the submitted jobs
            poll_interval: Polling interval in seconds
            
        Returns:
            Dictionary with job status information
        """
        # Initialize job status dictionary
        job_status = {
            'base_job_name': jobs_info.get('base_job_name'),
            'timestamp': jobs_info.get('timestamp'),
            'output_dir': jobs_info.get('output_dir'),
            'status': 'IN_PROGRESS',
            'jobs': {}
        }
        
        # Add normal job to status dictionary
        if 'normal_job' in jobs_info:
            normal_job = jobs_info['normal_job']
            job_status['jobs'][normal_job['job_name']] = {
                'job_id': normal_job['job_id'],
                'status': 'SUBMITTED'
            }
        
        # Add fire jobs to status dictionary
        if 'fire_jobs' in jobs_info:
            for fire_type, job in jobs_info['fire_jobs'].items():
                job_status['jobs'][job['job_name']] = {
                    'job_id': job['job_id'],
                    'status': 'SUBMITTED'
                }
        
        # Add false positive jobs to status dictionary
        if 'false_positive_jobs' in jobs_info:
            for fp_type, job in jobs_info['false_positive_jobs'].items():
                job_status['jobs'][job['job_name']] = {
                    'job_id': job['job_id'],
                    'status': 'SUBMITTED'
                }
        
        # Monitor job status until all jobs are complete
        all_complete = False
        while not all_complete:
            all_complete = True
            
            # Check status of each job
            for job_name, job_info in job_status['jobs'].items():
                job_id = job_info['job_id']
                
                # Skip jobs that are already complete
                if job_info['status'] in ['SUCCEEDED', 'FAILED']:
                    continue
                
                # Get job description
                try:
                    job = self.describe_job(job_id)
                    status = job.get('status')
                    
                    # Update job status
                    job_status['jobs'][job_name]['status'] = status
                    
                    # Add additional job information
                    if 'statusReason' in job:
                        job_status['jobs'][job_name]['status_reason'] = job['statusReason']
                    
                    if 'container' in job and 'exitCode' in job['container']:
                        job_status['jobs'][job_name]['exit_code'] = job['container']['exitCode']
                    
                    # Check if job is still in progress
                    if status not in ['SUCCEEDED', 'FAILED']:
                        all_complete = False
                except Exception as e:
                    self.logger.error(f"Error getting status for job {job_id}: {str(e)}")
                    all_complete = False
            
            # Wait before polling again if not all jobs are complete
            if not all_complete:
                time.sleep(poll_interval)
        
        # Set overall status based on job statuses
        failed_jobs = [job for job in job_status['jobs'].values() if job['status'] == 'FAILED']
        if failed_jobs:
            job_status['status'] = 'FAILED'
        else:
            job_status['status'] = 'SUCCEEDED'
        
        return job_status