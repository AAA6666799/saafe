"""
Feature Extraction Job Manager for the synthetic fire prediction system.

This module provides functionality for managing feature extraction jobs,
including job scheduling, resource allocation, status tracking, and error handling.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import os
import json
import logging
import threading
import time
from datetime import datetime
import uuid
import heapq
from enum import Enum


class JobStatus(Enum):
    """Enum for job status values."""
    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class FeatureExtractionJobManager:
    """
    Manager for feature extraction jobs.
    
    This class creates and manages feature extraction jobs, handles job scheduling
    and resource allocation, provides job status tracking and error handling, and
    implements retry mechanisms for failed jobs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature extraction job manager.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize job storage
        self.jobs = {}
        self.job_status = {}
        self.job_results = {}
        self.job_errors = {}
        
        # Initialize job queues
        self.pending_jobs = []  # Priority queue for pending jobs
        self.running_jobs = set()
        self.completed_jobs = set()
        self.failed_jobs = set()
        
        # Initialize locks for thread safety
        self.jobs_lock = threading.RLock()
        self.queue_lock = threading.RLock()
        
        # Initialize job scheduler thread
        self.scheduler_running = False
        self.scheduler_thread = None
        
        # Validate configuration
        self._validate_config()
        
        # Start scheduler if auto-start is enabled
        if self.config.get('auto_start_scheduler', False):
            self.start_scheduler()
    
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Set default values if not provided
        if 'max_concurrent_jobs' not in self.config:
            self.config['max_concurrent_jobs'] = 4
        
        if 'max_retries' not in self.config:
            self.config['max_retries'] = 3
        
        if 'retry_delay_seconds' not in self.config:
            self.config['retry_delay_seconds'] = 60
        
        if 'job_timeout_seconds' not in self.config:
            self.config['job_timeout_seconds'] = 3600  # 1 hour
        
        if 'status_update_interval_seconds' not in self.config:
            self.config['status_update_interval_seconds'] = 10
        
        # Create output directory if specified
        if 'output_dir' in self.config:
            os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def register_job(self, job_id: str, job_config: Dict[str, Any]) -> str:
        """
        Register a new feature extraction job.
        
        Args:
            job_id: Unique job identifier
            job_config: Job configuration parameters
            
        Returns:
            Job ID
        """
        with self.jobs_lock:
            # Check if job already exists
            if job_id in self.jobs:
                self.logger.warning(f"Job {job_id} already exists, updating configuration")
            
            # Set default values
            if 'priority' not in job_config:
                job_config['priority'] = 0
            
            if 'dependencies' not in job_config:
                job_config['dependencies'] = []
            
            if 'retry_count' not in job_config:
                job_config['retry_count'] = 0
            
            # Store job configuration
            self.jobs[job_id] = job_config
            
            # Initialize job status
            self.job_status[job_id] = {
                'status': JobStatus.PENDING.value,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Add to pending queue if no dependencies or all dependencies are completed
            if not job_config['dependencies'] or all(
                dep_id in self.job_status and 
                self.job_status[dep_id]['status'] == JobStatus.COMPLETED.value
                for dep_id in job_config['dependencies']
            ):
                self._add_to_pending_queue(job_id, job_config['priority'])
            
            self.logger.info(f"Registered job: {job_id}")
            return job_id
    
    def create_job(self, job_config: Dict[str, Any]) -> str:
        """
        Create a new feature extraction job with a generated ID.
        
        Args:
            job_config: Job configuration parameters
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        return self.register_job(job_id, job_config)
    
    def update_job_status(self, job_id: str, status: str, additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the status of a job.
        
        Args:
            job_id: Job identifier
            status: New job status
            additional_info: Optional additional information to store with the status
        """
        with self.jobs_lock:
            if job_id not in self.jobs:
                self.logger.warning(f"Cannot update status for unknown job: {job_id}")
                return
            
            # Update job status
            self.job_status[job_id] = {
                'status': status,
                'updated_at': datetime.now().isoformat(),
                **(self.job_status.get(job_id, {})),
                **(additional_info or {})
            }
            
            # Update job collections based on status
            if status == JobStatus.RUNNING.value:
                with self.queue_lock:
                    if job_id in self.pending_jobs:
                        self._remove_from_pending_queue(job_id)
                    self.running_jobs.add(job_id)
            
            elif status == JobStatus.COMPLETED.value:
                with self.queue_lock:
                    if job_id in self.running_jobs:
                        self.running_jobs.remove(job_id)
                    self.completed_jobs.add(job_id)
                
                # Store job results if provided
                if additional_info:
                    self.job_results[job_id] = additional_info
                
                # Check if any pending jobs depend on this job
                self._check_dependencies()
            
            elif status == JobStatus.FAILED.value:
                with self.queue_lock:
                    if job_id in self.running_jobs:
                        self.running_jobs.remove(job_id)
                    self.failed_jobs.add(job_id)
                
                # Store error information if provided
                if additional_info:
                    self.job_errors[job_id] = additional_info
                
                # Check if job should be retried
                self._handle_job_failure(job_id)
            
            self.logger.debug(f"Updated job {job_id} status to {status}")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a specific job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary with job status information
        """
        with self.jobs_lock:
            if job_id not in self.jobs:
                return {'error': f"Job not found: {job_id}"}
            
            return {
                'job_id': job_id,
                'config': self.jobs[job_id],
                'status': self.job_status.get(job_id, {}),
                'results': self.job_results.get(job_id, {}),
                'errors': self.job_errors.get(job_id, {})
            }
    
    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all jobs.
        
        Returns:
            Dictionary mapping job IDs to job status information
        """
        with self.jobs_lock:
            return {
                job_id: {
                    'config': self.jobs[job_id],
                    'status': self.job_status.get(job_id, {}),
                    'results': self.job_results.get(job_id, {}),
                    'errors': self.job_errors.get(job_id, {})
                }
                for job_id in self.jobs
            }
    
    def get_jobs_by_status(self, status: str) -> List[str]:
        """
        Get IDs of jobs with a specific status.
        
        Args:
            status: Job status to filter by
            
        Returns:
            List of job IDs
        """
        with self.jobs_lock:
            return [
                job_id for job_id in self.jobs
                if job_id in self.job_status and self.job_status[job_id]['status'] == status
            ]
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job if it's not already completed or running.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled, False otherwise
        """
        with self.jobs_lock:
            if job_id not in self.jobs:
                self.logger.warning(f"Cannot cancel unknown job: {job_id}")
                return False
            
            current_status = self.job_status.get(job_id, {}).get('status')
            
            if current_status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
                self.logger.warning(f"Cannot cancel job {job_id} with status {current_status}")
                return False
            
            # Remove from queues
            with self.queue_lock:
                if job_id in self.pending_jobs:
                    self._remove_from_pending_queue(job_id)
                
                if job_id in self.running_jobs:
                    self.logger.warning(f"Cancelling running job: {job_id}")
                    # Note: In a real system, this would need to signal the worker to stop
                    self.running_jobs.remove(job_id)
            
            # Update status
            self.update_job_status(job_id, JobStatus.CANCELLED.value, {
                'cancelled_at': datetime.now().isoformat()
            })
            
            self.logger.info(f"Cancelled job: {job_id}")
            return True
    
    def retry_job(self, job_id: str, reset_retry_count: bool = False) -> bool:
        """
        Retry a failed job.
        
        Args:
            job_id: Job identifier
            reset_retry_count: Whether to reset the retry count
            
        Returns:
            True if job was queued for retry, False otherwise
        """
        with self.jobs_lock:
            if job_id not in self.jobs:
                self.logger.warning(f"Cannot retry unknown job: {job_id}")
                return False
            
            current_status = self.job_status.get(job_id, {}).get('status')
            
            if current_status != JobStatus.FAILED.value:
                self.logger.warning(f"Cannot retry job {job_id} with status {current_status}")
                return False
            
            # Update retry count
            if reset_retry_count:
                self.jobs[job_id]['retry_count'] = 0
            else:
                self.jobs[job_id]['retry_count'] += 1
            
            # Add to pending queue
            with self.queue_lock:
                self._add_to_pending_queue(job_id, self.jobs[job_id].get('priority', 0))
            
            # Update status
            self.update_job_status(job_id, JobStatus.PENDING.value, {
                'retried_at': datetime.now().isoformat(),
                'retry_count': self.jobs[job_id]['retry_count']
            })
            
            self.logger.info(f"Queued job {job_id} for retry (attempt {self.jobs[job_id]['retry_count']})")
            return True
    
    def start_scheduler(self) -> None:
        """
        Start the job scheduler thread.
        """
        if self.scheduler_running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        self.logger.info("Job scheduler started")
    
    def stop_scheduler(self) -> None:
        """
        Stop the job scheduler thread.
        """
        if not self.scheduler_running:
            self.logger.warning("Scheduler is not running")
            return
        
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        self.logger.info("Job scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """
        Main loop for the job scheduler.
        """
        while self.scheduler_running:
            try:
                # Check for jobs that can be scheduled
                self._schedule_pending_jobs()
                
                # Check for timed-out jobs
                self._check_job_timeouts()
                
                # Sleep for a short interval
                time.sleep(self.config.get('status_update_interval_seconds', 10))
            
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {str(e)}")
    
    def _schedule_pending_jobs(self) -> None:
        """
        Schedule pending jobs based on available resources and priorities.
        """
        with self.queue_lock:
            # Check if we can schedule more jobs
            available_slots = self.config['max_concurrent_jobs'] - len(self.running_jobs)
            
            if available_slots <= 0 or not self.pending_jobs:
                return
            
            # Schedule jobs until no more slots or no more pending jobs
            scheduled_count = 0
            while available_slots > 0 and self.pending_jobs:
                # Get highest priority job
                _, job_id = heapq.heappop(self.pending_jobs)
                
                # Check if job still exists and is still pending
                if (job_id not in self.jobs or 
                    job_id not in self.job_status or 
                    self.job_status[job_id]['status'] != JobStatus.PENDING.value):
                    continue
                
                # Update job status to scheduled
                self.update_job_status(job_id, JobStatus.SCHEDULED.value, {
                    'scheduled_at': datetime.now().isoformat()
                })
                
                # In a real system, this would dispatch the job to a worker
                # For this implementation, we'll just mark it as running
                self.update_job_status(job_id, JobStatus.RUNNING.value, {
                    'started_at': datetime.now().isoformat()
                })
                
                available_slots -= 1
                scheduled_count += 1
            
            if scheduled_count > 0:
                self.logger.info(f"Scheduled {scheduled_count} jobs")
    
    def _check_job_timeouts(self) -> None:
        """
        Check for jobs that have exceeded their timeout.
        """
        timeout_seconds = self.config.get('job_timeout_seconds', 3600)
        current_time = datetime.now()
        
        with self.jobs_lock:
            for job_id in list(self.running_jobs):
                if job_id not in self.job_status:
                    continue
                
                status_info = self.job_status[job_id]
                
                if 'started_at' in status_info:
                    started_at = datetime.fromisoformat(status_info['started_at'])
                    elapsed_seconds = (current_time - started_at).total_seconds()
                    
                    if elapsed_seconds > timeout_seconds:
                        self.logger.warning(f"Job {job_id} timed out after {elapsed_seconds:.1f} seconds")
                        
                        # Mark job as failed
                        self.update_job_status(job_id, JobStatus.FAILED.value, {
                            'error': f"Job timed out after {elapsed_seconds:.1f} seconds",
                            'timed_out_at': current_time.isoformat()
                        })
    
    def _handle_job_failure(self, job_id: str) -> None:
        """
        Handle a job failure, potentially scheduling a retry.
        
        Args:
            job_id: Job identifier
        """
        if job_id not in self.jobs:
            return
        
        job_config = self.jobs[job_id]
        retry_count = job_config.get('retry_count', 0)
        max_retries = self.config.get('max_retries', 3)
        
        if retry_count < max_retries:
            # Schedule retry after delay
            retry_delay = self.config.get('retry_delay_seconds', 60)
            
            def delayed_retry():
                time.sleep(retry_delay)
                self.retry_job(job_id)
            
            threading.Thread(target=delayed_retry).start()
            
            self.logger.info(f"Scheduled retry for job {job_id} in {retry_delay} seconds (attempt {retry_count + 1}/{max_retries})")
        else:
            self.logger.warning(f"Job {job_id} failed after {retry_count} retry attempts")
    
    def _check_dependencies(self) -> None:
        """
        Check if any pending jobs have their dependencies satisfied.
        """
        with self.jobs_lock:
            for job_id, job_config in self.jobs.items():
                # Skip jobs that are not pending or have no dependencies
                if (job_id not in self.job_status or 
                    self.job_status[job_id]['status'] != JobStatus.PENDING.value or
                    not job_config.get('dependencies')):
                    continue
                
                # Check if all dependencies are completed
                dependencies_met = all(
                    dep_id in self.job_status and 
                    self.job_status[dep_id]['status'] == JobStatus.COMPLETED.value
                    for dep_id in job_config['dependencies']
                )
                
                if dependencies_met:
                    # Add to pending queue
                    with self.queue_lock:
                        self._add_to_pending_queue(job_id, job_config.get('priority', 0))
                    
                    self.logger.info(f"Dependencies satisfied for job {job_id}, added to pending queue")
    
    def _add_to_pending_queue(self, job_id: str, priority: int) -> None:
        """
        Add a job to the pending queue with the specified priority.
        
        Args:
            job_id: Job identifier
            priority: Job priority (lower value = higher priority)
        """
        # Use negative priority for the heap (so lower values have higher priority)
        heapq.heappush(self.pending_jobs, (-priority, job_id))
        
        # Update status if needed
        if (job_id in self.job_status and 
            self.job_status[job_id]['status'] not in [JobStatus.PENDING.value, JobStatus.SCHEDULED.value]):
            self.update_job_status(job_id, JobStatus.PENDING.value)
    
    def _remove_from_pending_queue(self, job_id: str) -> None:
        """
        Remove a job from the pending queue.
        
        Args:
            job_id: Job identifier
        """
        # Create a new queue without the specified job
        self.pending_jobs = [(p, j) for p, j in self.pending_jobs if j != job_id]
        heapq.heapify(self.pending_jobs)
    
    def save_state(self, filepath: str) -> None:
        """
        Save the current state of the job manager to a file.
        
        Args:
            filepath: Path to save the state
        """
        with self.jobs_lock:
            state = {
                'jobs': self.jobs,
                'job_status': self.job_status,
                'job_results': self.job_results,
                'job_errors': self.job_errors,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Saved job manager state to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """
        Load the state of the job manager from a file.
        
        Args:
            filepath: Path to load the state from
        """
        if not os.path.exists(filepath):
            self.logger.warning(f"State file not found: {filepath}")
            return
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            with self.jobs_lock, self.queue_lock:
                # Load job data
                self.jobs = state.get('jobs', {})
                self.job_status = state.get('job_status', {})
                self.job_results = state.get('job_results', {})
                self.job_errors = state.get('job_errors', {})
                
                # Rebuild job collections
                self.pending_jobs = []
                self.running_jobs = set()
                self.completed_jobs = set()
                self.failed_jobs = set()
                
                for job_id, status_info in self.job_status.items():
                    status = status_info.get('status')
                    
                    if status == JobStatus.PENDING.value:
                        priority = self.jobs.get(job_id, {}).get('priority', 0)
                        self._add_to_pending_queue(job_id, priority)
                    
                    elif status == JobStatus.RUNNING.value:
                        self.running_jobs.add(job_id)
                    
                    elif status == JobStatus.COMPLETED.value:
                        self.completed_jobs.add(job_id)
                    
                    elif status == JobStatus.FAILED.value:
                        self.failed_jobs.add(job_id)
            
            self.logger.info(f"Loaded job manager state from {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error loading state from {filepath}: {str(e)}")