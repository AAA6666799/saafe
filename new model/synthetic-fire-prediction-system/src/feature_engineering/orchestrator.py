"""
Feature Extraction Orchestrator for the synthetic fire prediction system.

This module provides the orchestration layer for coordinating feature extraction jobs,
managing dependencies, and handling parallel processing.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import os
import json
import logging
import concurrent.futures
import time
from datetime import datetime
import uuid
from tqdm import tqdm

from .framework import FeatureExtractionFramework
from .job_manager import FeatureExtractionJobManager
# AWS integration (optional)
try:
    from ..aws.batch.service import BatchServiceImpl
    AWS_BATCH_AVAILABLE = True
except ImportError:
    AWS_BATCH_AVAILABLE = False
    BatchServiceImpl = None


class FeatureExtractionOrchestrator:
    """
    Orchestrator for feature extraction jobs.
    
    This class coordinates the execution of feature extraction jobs,
    manages dependencies between feature extraction steps, handles parallel
    processing, and provides monitoring and logging capabilities.
    """
    
    def __init__(self, config: Dict[str, Any], framework: Optional[FeatureExtractionFramework] = None):
        """
        Initialize the feature extraction orchestrator.
        
        Args:
            config: Dictionary containing configuration parameters
            framework: Optional feature extraction framework instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize framework if not provided
        if framework is None:
            self.framework = FeatureExtractionFramework(config.get('framework_config', {}))
        else:
            self.framework = framework
        
        # Initialize job manager
        self.job_manager = FeatureExtractionJobManager(config.get('job_manager_config', {}))
        
        # Initialize AWS Batch service if AWS integration is enabled
        self.batch_service = None
        if self.config.get('aws_integration', False):
            self.batch_service = BatchServiceImpl(self.config.get('aws_config', {}))
        
        # Initialize state
        self.running_jobs = {}
        self.completed_jobs = {}
        self.failed_jobs = {}
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_params = ['output_dir', 'max_concurrent_jobs']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def process_dataset(self, 
                       dataset_path: str, 
                       output_path: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       use_aws: bool = False) -> Dict[str, Any]:
        """
        Process a dataset by orchestrating feature extraction jobs.
        
        Args:
            dataset_path: Path to the dataset
            output_path: Optional path to save extracted features
            metadata: Optional metadata about the dataset
            use_aws: Whether to use AWS services for processing
            
        Returns:
            Dictionary containing processing results
        """
        self.logger.info(f"Processing dataset: {dataset_path}")
        
        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.config['output_dir'], 
                                      f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize processing metadata
        process_id = str(uuid.uuid4())
        process_metadata = {
            'process_id': process_id,
            'dataset_path': dataset_path,
            'output_path': output_path,
            'start_time': datetime.now().isoformat(),
            'status': 'RUNNING',
            'jobs': []
        }
        
        # Add provided metadata if available
        if metadata:
            process_metadata.update(metadata)
        
        # Save initial metadata
        metadata_path = os.path.join(output_path, "process_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(process_metadata, f, indent=2)
        
        try:
            if use_aws and self.batch_service is not None:
                # Process using AWS services
                results = self._process_with_aws(dataset_path, output_path, process_id)
            else:
                # Process locally
                results = self._process_locally(dataset_path, output_path, process_id)
            
            # Update and save final metadata
            process_metadata.update({
                'end_time': datetime.now().isoformat(),
                'status': 'COMPLETED',
                'results': results
            })
            
            with open(metadata_path, 'w') as f:
                json.dump(process_metadata, f, indent=2)
            
            self.logger.info(f"Dataset processing completed: {process_id}")
            return process_metadata
        
        except Exception as e:
            # Update metadata with error information
            process_metadata.update({
                'end_time': datetime.now().isoformat(),
                'status': 'FAILED',
                'error': str(e)
            })
            
            with open(metadata_path, 'w') as f:
                json.dump(process_metadata, f, indent=2)
            
            self.logger.error(f"Dataset processing failed: {str(e)}")
            raise
    
    def _process_locally(self, 
                        dataset_path: str, 
                        output_path: str,
                        process_id: str) -> Dict[str, Any]:
        """
        Process a dataset locally using parallel processing.
        
        Args:
            dataset_path: Path to the dataset
            output_path: Path to save extracted features
            process_id: Unique process identifier
            
        Returns:
            Dictionary containing processing results
        """
        self.logger.info(f"Processing dataset locally: {dataset_path}")
        
        # Discover scenarios in the dataset
        scenarios = self._discover_scenarios(dataset_path)
        
        if not scenarios:
            self.logger.warning(f"No scenarios found in dataset: {dataset_path}")
            return {'warning': 'No scenarios found'}
        
        # Create jobs for each scenario
        jobs = []
        for scenario in scenarios:
            job_id = f"{process_id}_{scenario['id']}"
            job_config = {
                'job_id': job_id,
                'scenario_path': scenario['path'],
                'output_path': os.path.join(output_path, f"scenario_{scenario['id']}"),
                'metadata': {
                    'scenario_id': scenario['id'],
                    'scenario_type': scenario.get('type', 'unknown')
                }
            }
            jobs.append(job_config)
        
        # Process jobs in parallel
        max_workers = min(self.config['max_concurrent_jobs'], len(jobs))
        results = {
            'total_jobs': len(jobs),
            'completed_jobs': 0,
            'failed_jobs': 0,
            'job_results': {}
        }
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit jobs
            future_to_job = {
                executor.submit(self._process_job, job): job
                for job in jobs
            }
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_job), 
                              total=len(jobs), 
                              desc="Processing scenarios"):
                job = future_to_job[future]
                job_id = job['job_id']
                
                try:
                    job_result = future.result()
                    results['job_results'][job_id] = job_result
                    results['completed_jobs'] += 1
                    self.completed_jobs[job_id] = job_result
                except Exception as e:
                    results['job_results'][job_id] = {'error': str(e)}
                    results['failed_jobs'] += 1
                    self.failed_jobs[job_id] = {'error': str(e)}
                    self.logger.error(f"Job failed {job_id}: {str(e)}")
        
        # Save results summary
        results_path = os.path.join(output_path, "processing_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Local processing completed: {results['completed_jobs']} succeeded, {results['failed_jobs']} failed")
        return results
    
    def _process_with_aws(self, 
                         dataset_path: str, 
                         output_path: str,
                         process_id: str) -> Dict[str, Any]:
        """
        Process a dataset using AWS services.
        
        Args:
            dataset_path: Path to the dataset
            output_path: Path to save extracted features
            process_id: Unique process identifier
            
        Returns:
            Dictionary containing processing results
        """
        self.logger.info(f"Processing dataset with AWS: {dataset_path}")
        
        if self.batch_service is None:
            raise ValueError("AWS Batch service is not initialized")
        
        # Discover scenarios in the dataset
        scenarios = self._discover_scenarios(dataset_path)
        
        if not scenarios:
            self.logger.warning(f"No scenarios found in dataset: {dataset_path}")
            return {'warning': 'No scenarios found'}
        
        # Get AWS configuration
        aws_config = self.config.get('aws_config', {})
        if 'job_definition' not in aws_config:
            raise ValueError("Missing 'job_definition' in AWS configuration")
        
        # Submit AWS Batch jobs for each scenario
        batch_jobs = []
        for scenario in scenarios:
            job_name = f"feature-extraction-{scenario['id']}-{process_id[:8]}"
            
            # Prepare container overrides
            container_overrides = {
                'command': [
                    'python', '-m', 'src.feature_engineering.batch_jobs.extract_features',
                    '--scenario-path', scenario['path'],
                    '--output-path', f"s3://{aws_config.get('default_bucket', 'fire-prediction')}/features/{process_id}/scenario_{scenario['id']}",
                    '--config-path', self.config.get('config_path', 'config/feature_extraction_config.json')
                ]
            }
            
            # Submit job
            job_response = self.batch_service.submit_job(
                job_name=job_name,
                job_definition=aws_config['job_definition'],
                container_overrides=container_overrides
            )
            
            batch_jobs.append({
                'job_id': job_response['jobId'],
                'job_name': job_name,
                'scenario_id': scenario['id'],
                'scenario_path': scenario['path']
            })
            
            self.logger.info(f"Submitted AWS Batch job: {job_name} (ID: {job_response['jobId']})")
        
        # Save batch jobs metadata
        batch_jobs_metadata = {
            'process_id': process_id,
            'total_jobs': len(batch_jobs),
            'jobs': batch_jobs
        }
        
        batch_jobs_path = os.path.join(output_path, "aws_batch_jobs.json")
        with open(batch_jobs_path, 'w') as f:
            json.dump(batch_jobs_metadata, f, indent=2)
        
        self.logger.info(f"Submitted {len(batch_jobs)} AWS Batch jobs")
        return batch_jobs_metadata
    
    def _process_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single feature extraction job.
        
        Args:
            job_config: Job configuration
            
        Returns:
            Dictionary containing job results
        """
        job_id = job_config['job_id']
        scenario_path = job_config['scenario_path']
        output_path = job_config['output_path']
        metadata = job_config.get('metadata', {})
        
        self.logger.debug(f"Processing job {job_id} for scenario: {scenario_path}")
        
        # Register job with job manager
        self.job_manager.register_job(job_id, job_config)
        
        # Update job status
        self.job_manager.update_job_status(job_id, 'RUNNING')
        self.running_jobs[job_id] = job_config
        
        try:
            # Extract features using the framework
            start_time = time.time()
            features = self.framework.extract_features(
                dataset_path=scenario_path,
                output_path=output_path,
                metadata=metadata
            )
            end_time = time.time()
            
            # Calculate processing time
            processing_time = end_time - start_time
            
            # Prepare job results
            job_results = {
                'job_id': job_id,
                'status': 'COMPLETED',
                'processing_time': processing_time,
                'output_path': output_path,
                'feature_count': sum(len(features.get('features', {}).get(extractor_type, {})) 
                                   for extractor_type in features.get('features', {})),
                'completion_time': datetime.now().isoformat()
            }
            
            # Update job status
            self.job_manager.update_job_status(job_id, 'COMPLETED', job_results)
            
            # Remove from running jobs
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            
            self.logger.debug(f"Job {job_id} completed successfully")
            return job_results
        
        except Exception as e:
            # Update job status with error
            error_info = {
                'error': str(e),
                'error_time': datetime.now().isoformat()
            }
            
            self.job_manager.update_job_status(job_id, 'FAILED', error_info)
            
            # Remove from running jobs
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            
            self.logger.error(f"Job {job_id} failed: {str(e)}")
            raise
    
    def _discover_scenarios(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Discover scenarios in a dataset.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            List of scenario information dictionaries
        """
        scenarios = []
        
        # Check if dataset path is a directory
        if not os.path.isdir(dataset_path):
            self.logger.warning(f"Dataset path is not a directory: {dataset_path}")
            return scenarios
        
        # Look for metadata file
        metadata_path = os.path.join(dataset_path, "dataset_metadata.json")
        if os.path.exists(metadata_path):
            # Parse metadata to find scenarios
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                if 'scenarios' in metadata:
                    for i, scenario in enumerate(metadata['scenarios']):
                        scenario_id = scenario.get('scenario_id', i)
                        scenario_dir = scenario.get('scenario_dir')
                        
                        if scenario_dir and os.path.isdir(scenario_dir):
                            scenarios.append({
                                'id': scenario_id,
                                'path': scenario_dir,
                                'type': scenario.get('scenario_type', 'unknown')
                            })
                
                self.logger.info(f"Found {len(scenarios)} scenarios in dataset metadata")
                return scenarios
            
            except Exception as e:
                self.logger.warning(f"Error parsing dataset metadata: {str(e)}")
        
        # If no metadata or parsing failed, try to discover scenarios by directory structure
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            
            # Check if it's a directory and looks like a scenario
            if os.path.isdir(item_path) and (
                item.startswith('scenario_') or 
                item.startswith('normal_scenario_') or
                item.startswith('fire_scenario_') or
                item.startswith('false_positive_scenario_')
            ):
                # Extract scenario ID from directory name
                scenario_id = item.split('_')[-1]
                
                # Determine scenario type
                scenario_type = 'unknown'
                if 'normal' in item:
                    scenario_type = 'normal'
                elif 'fire' in item:
                    scenario_type = 'fire'
                elif 'false_positive' in item:
                    scenario_type = 'false_positive'
                
                scenarios.append({
                    'id': scenario_id,
                    'path': item_path,
                    'type': scenario_type
                })
        
        self.logger.info(f"Discovered {len(scenarios)} scenarios by directory structure")
        return scenarios
    
    def monitor_aws_jobs(self, process_id: str, poll_interval: int = 60) -> Dict[str, Any]:
        """
        Monitor AWS Batch jobs for a process.
        
        Args:
            process_id: Process identifier
            poll_interval: Polling interval in seconds
            
        Returns:
            Dictionary with job status information
        """
        if self.batch_service is None:
            raise ValueError("AWS Batch service is not initialized")
        
        # Find the process metadata
        output_dir = self.config['output_dir']
        process_dirs = [d for d in os.listdir(output_dir) 
                       if os.path.isdir(os.path.join(output_dir, d)) and process_id in d]
        
        if not process_dirs:
            raise ValueError(f"Process directory not found for process ID: {process_id}")
        
        process_dir = os.path.join(output_dir, process_dirs[0])
        batch_jobs_path = os.path.join(process_dir, "aws_batch_jobs.json")
        
        if not os.path.exists(batch_jobs_path):
            raise ValueError(f"AWS Batch jobs metadata not found: {batch_jobs_path}")
        
        # Load batch jobs metadata
        with open(batch_jobs_path, 'r') as f:
            batch_jobs_metadata = json.load(f)
        
        # Initialize job status dictionary
        job_status = {
            'process_id': process_id,
            'status': 'IN_PROGRESS',
            'jobs': {}
        }
        
        # Add jobs to status dictionary
        for job in batch_jobs_metadata.get('jobs', []):
            job_status['jobs'][job['job_id']] = {
                'job_name': job['job_name'],
                'scenario_id': job['scenario_id'],
                'status': 'SUBMITTED'
            }
        
        # Monitor job status until all jobs are complete
        all_complete = False
        while not all_complete:
            all_complete = True
            
            # Check status of each job
            for job_id, job_info in job_status['jobs'].items():
                # Skip jobs that are already complete
                if job_info['status'] in ['SUCCEEDED', 'FAILED']:
                    continue
                
                # Get job description
                try:
                    job = self.batch_service.describe_job(job_id)
                    status = job.get('status')
                    
                    # Update job status
                    job_status['jobs'][job_id]['status'] = status
                    
                    # Add additional job information
                    if 'statusReason' in job:
                        job_status['jobs'][job_id]['status_reason'] = job['statusReason']
                    
                    if 'container' in job and 'exitCode' in job['container']:
                        job_status['jobs'][job_id]['exit_code'] = job['container']['exitCode']
                    
                    # Check if job is still in progress
                    if status not in ['SUCCEEDED', 'FAILED']:
                        all_complete = False
                except Exception as e:
                    self.logger.error(f"Error getting status for job {job_id}: {str(e)}")
                    all_complete = False
            
            # Save current status
            status_path = os.path.join(process_dir, "aws_jobs_status.json")
            with open(status_path, 'w') as f:
                json.dump(job_status, f, indent=2)
            
            # Wait before polling again if not all jobs are complete
            if not all_complete:
                time.sleep(poll_interval)
        
        # Set overall status based on job statuses
        failed_jobs = [job for job in job_status['jobs'].values() if job['status'] == 'FAILED']
        if failed_jobs:
            job_status['status'] = 'FAILED'
        else:
            job_status['status'] = 'SUCCEEDED'
        
        # Save final status
        status_path = os.path.join(process_dir, "aws_jobs_status.json")
        with open(status_path, 'w') as f:
            json.dump(job_status, f, indent=2)
        
        self.logger.info(f"AWS job monitoring completed: {len(job_status['jobs']) - len(failed_jobs)} succeeded, {len(failed_jobs)} failed")
        return job_status
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a specific job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary with job status information
        """
        return self.job_manager.get_job_status(job_id)
    
    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all jobs.
        
        Returns:
            Dictionary mapping job IDs to job status information
        """
        return self.job_manager.get_all_jobs()