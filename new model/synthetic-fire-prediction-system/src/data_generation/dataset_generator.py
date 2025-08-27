"""
Dataset generator for synthetic fire data.

This module provides functionality for generating comprehensive synthetic datasets
that combine thermal, gas, and environmental data for various fire scenarios.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import logging
import concurrent.futures
import boto3
import botocore
from tqdm import tqdm
import time
import uuid

from .base import DataGenerator
from .thermal.thermal_image_generator import ThermalImageGenerator
from .gas.gas_concentration_generator import GasConcentrationGenerator
from .environmental.environmental_data_generator import EnvironmentalDataGenerator
from .scenarios.scenario_generator import ScenarioGenerator
from .scenarios.specific_generators import (
    ElectricalFireGenerator,
    ChemicalFireGenerator,
    SmolderingFireGenerator,
    RapidCombustionFireGenerator
)
from .scenarios.false_positive_generator import FalsePositiveGenerator

from ..aws.s3.service import S3ServiceImpl


class DatasetGenerator:
    """
    Class for generating comprehensive synthetic datasets.
    
    This class coordinates the generation of large-scale datasets using the
    scenario generation system and other data generators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset generator with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.validate_config()
        
        # Initialize component generators
        self.thermal_generator = ThermalImageGenerator(self.config.get('thermal_config', {}))
        self.gas_generator = GasConcentrationGenerator(self.config.get('gas_config', {}))
        self.environmental_generator = EnvironmentalDataGenerator(self.config.get('environmental_config', {}))
        
        # Initialize scenario generator
        self.scenario_generator = ScenarioGenerator(
            thermal_generator=self.thermal_generator,
            gas_generator=self.gas_generator,
            environmental_generator=self.environmental_generator,
            config=self.config.get('scenario_config', {})
        )
        
        # Initialize specific scenario generators
        self.electrical_fire_generator = ElectricalFireGenerator(
            thermal_generator=self.thermal_generator,
            gas_generator=self.gas_generator,
            environmental_generator=self.environmental_generator,
            config=self.config.get('electrical_fire_config', {})
        )
        
        self.chemical_fire_generator = ChemicalFireGenerator(
            thermal_generator=self.thermal_generator,
            gas_generator=self.gas_generator,
            environmental_generator=self.environmental_generator,
            config=self.config.get('chemical_fire_config', {})
        )
        
        self.smoldering_fire_generator = SmolderingFireGenerator(
            thermal_generator=self.thermal_generator,
            gas_generator=self.gas_generator,
            environmental_generator=self.environmental_generator,
            config=self.config.get('smoldering_fire_config', {})
        )
        
        self.rapid_combustion_generator = RapidCombustionFireGenerator(
            thermal_generator=self.thermal_generator,
            gas_generator=self.gas_generator,
            environmental_generator=self.environmental_generator,
            config=self.config.get('rapid_combustion_config', {})
        )
        
        self.false_positive_generator = FalsePositiveGenerator(
            thermal_generator=self.thermal_generator,
            gas_generator=self.gas_generator,
            environmental_generator=self.environmental_generator,
            config=self.config.get('false_positive_config', {})
        )
        
        # Initialize S3 service if AWS integration is enabled
        self.s3_service = None
        if self.config.get('aws_integration', False):
            self.s3_service = S3ServiceImpl(self.config.get('aws_config', {}))
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """
        Set up logging configuration.
        """
        log_level = self.config.get('log_level', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required configuration sections
        required_sections = ['output_dir']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Check AWS integration
        if self.config.get('aws_integration', False):
            if 'aws_config' not in self.config:
                raise ValueError("aws_config is required when aws_integration is enabled")
            
            if 'default_bucket' not in self.config['aws_config']:
                raise ValueError("default_bucket is required in aws_config")
        
        # Set default values if not provided
        if 'num_normal_hours' not in self.config:
            self.config['num_normal_hours'] = 1000
        
        if 'num_fire_scenarios' not in self.config:
            self.config['num_fire_scenarios'] = 100
        
        if 'num_false_positive_scenarios' not in self.config:
            self.config['num_false_positive_scenarios'] = 50
        
        if 'scenario_duration' not in self.config:
            self.config['scenario_duration'] = 3600  # 1 hour in seconds
        
        if 'sample_rate' not in self.config:
            self.config['sample_rate'] = 1.0  # 1 Hz
    
    def generate_normal_operation_data(self, 
                                      num_hours: int, 
                                      output_dir: str,
                                      parallel: bool = False,
                                      max_workers: int = 4,
                                      seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate normal operation data.
        
        Args:
            num_hours: Number of hours of normal operation data to generate
            output_dir: Directory to save the generated data
            parallel: Whether to generate data in parallel
            max_workers: Maximum number of parallel workers
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary with metadata about the generated dataset
        """
        self.logger.info(f"Generating {num_hours} hours of normal operation data")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate number of scenarios to generate (1 scenario per hour)
        num_scenarios = num_hours
        
        # Define scenario parameters
        scenario_params = {
            'room_params': {
                'room_volume': 50.0,
                'ventilation_rate': 0.5,
                'initial_temperature': 20.0,
                'initial_humidity': 50.0
            }
        }
        
        # Generate scenarios
        if parallel and num_scenarios > 1:
            return self._generate_scenarios_parallel(
                scenario_type='normal',
                num_scenarios=num_scenarios,
                scenario_params=scenario_params,
                output_dir=output_dir,
                max_workers=max_workers,
                seed=seed
            )
        else:
            return self._generate_scenarios_sequential(
                scenario_type='normal',
                num_scenarios=num_scenarios,
                scenario_params=scenario_params,
                output_dir=output_dir,
                seed=seed
            )
    
    def generate_fire_scenarios(self, 
                              fire_type: str,
                              num_scenarios: int, 
                              output_dir: str,
                              parallel: bool = False,
                              max_workers: int = 4,
                              seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate fire scenarios of a specific type.
        
        Args:
            fire_type: Type of fire ('electrical', 'chemical', 'smoldering', 'rapid_combustion')
            num_scenarios: Number of scenarios to generate
            output_dir: Directory to save the generated data
            parallel: Whether to generate data in parallel
            max_workers: Maximum number of parallel workers
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary with metadata about the generated dataset
        """
        self.logger.info(f"Generating {num_scenarios} {fire_type} fire scenarios")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Select appropriate generator based on fire type
        if fire_type == 'electrical':
            generator = self.electrical_fire_generator
        elif fire_type == 'chemical':
            generator = self.chemical_fire_generator
        elif fire_type == 'smoldering':
            generator = self.smoldering_fire_generator
        elif fire_type == 'rapid_combustion':
            generator = self.rapid_combustion_generator
        else:
            raise ValueError(f"Unsupported fire type: {fire_type}")
        
        # Generate scenarios
        if parallel and num_scenarios > 1:
            return self._generate_scenarios_parallel(
                scenario_type=fire_type,
                num_scenarios=num_scenarios,
                scenario_params={},  # Use default params from specific generator
                output_dir=output_dir,
                max_workers=max_workers,
                seed=seed,
                generator=generator
            )
        else:
            return self._generate_scenarios_sequential(
                scenario_type=fire_type,
                num_scenarios=num_scenarios,
                scenario_params={},  # Use default params from specific generator
                output_dir=output_dir,
                seed=seed,
                generator=generator
            )
    
    def generate_false_positive_scenarios(self, 
                                        false_positive_type: str,
                                        num_scenarios: int, 
                                        output_dir: str,
                                        parallel: bool = False,
                                        max_workers: int = 4,
                                        seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate false positive scenarios of a specific type.
        
        Args:
            false_positive_type: Type of false positive ('cooking', 'heating', 'other')
            num_scenarios: Number of scenarios to generate
            output_dir: Directory to save the generated data
            parallel: Whether to generate data in parallel
            max_workers: Maximum number of parallel workers
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary with metadata about the generated dataset
        """
        self.logger.info(f"Generating {num_scenarios} {false_positive_type} false positive scenarios")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Define scenario parameters
        scenario_params = {
            'false_positive_type': false_positive_type
        }
        
        # Generate scenarios
        if parallel and num_scenarios > 1:
            return self._generate_scenarios_parallel(
                scenario_type='false_positive',
                num_scenarios=num_scenarios,
                scenario_params=scenario_params,
                output_dir=output_dir,
                max_workers=max_workers,
                seed=seed,
                generator=self.false_positive_generator
            )
        else:
            return self._generate_scenarios_sequential(
                scenario_type='false_positive',
                num_scenarios=num_scenarios,
                scenario_params=scenario_params,
                output_dir=output_dir,
                seed=seed,
                generator=self.false_positive_generator
            )
    
    def _generate_scenarios_sequential(self,
                                     scenario_type: str,
                                     num_scenarios: int,
                                     scenario_params: Dict[str, Any],
                                     output_dir: str,
                                     seed: Optional[int] = None,
                                     generator: Optional[ScenarioGenerator] = None) -> Dict[str, Any]:
        """
        Generate scenarios sequentially.
        
        Args:
            scenario_type: Type of scenario
            num_scenarios: Number of scenarios to generate
            scenario_params: Parameters for the scenarios
            output_dir: Directory to save the generated data
            seed: Optional random seed for reproducibility
            generator: Optional specific generator to use
            
        Returns:
            Dictionary with metadata about the generated dataset
        """
        # Use default generator if not provided
        if generator is None:
            generator = self.scenario_generator
        
        # Get scenario duration and sample rate from config
        duration_seconds = self.config.get('scenario_duration', 3600)
        sample_rate_hz = self.config.get('sample_rate', 1.0)
        
        # Generate dataset metadata
        dataset_metadata = {
            'dataset_name': f"{scenario_type}_scenarios",
            'num_scenarios': num_scenarios,
            'scenario_type': scenario_type,
            'duration_seconds': duration_seconds,
            'sample_rate_hz': sample_rate_hz,
            'creation_date': datetime.now().isoformat(),
            'scenarios': []
        }
        
        # Generate scenarios with progress bar
        for i in tqdm(range(num_scenarios), desc=f"Generating {scenario_type} scenarios"):
            try:
                # Generate start time (random time in the past week)
                start_time = datetime.now() - timedelta(days=np.random.randint(0, 7))
                
                # Generate scenario
                scenario_data = generator.generate_scenario(
                    start_time=start_time,
                    duration_seconds=duration_seconds,
                    sample_rate_hz=sample_rate_hz,
                    scenario_type=scenario_type,
                    scenario_params=scenario_params,
                    seed=seed + i if seed is not None else None
                )
                
                # Save scenario
                scenario_dir = os.path.join(output_dir, f'{scenario_type}_scenario_{i:04d}')
                generator.save_scenario(scenario_data, scenario_dir)
                
                # Add to dataset metadata
                scenario_metadata = scenario_data['metadata'].copy()
                scenario_metadata['scenario_id'] = i
                scenario_metadata['scenario_dir'] = scenario_dir
                dataset_metadata['scenarios'].append(scenario_metadata)
                
                self.logger.debug(f"Generated {scenario_type} scenario {i+1}/{num_scenarios}")
            except Exception as e:
                self.logger.error(f"Error generating {scenario_type} scenario {i}: {str(e)}")
        
        # Save dataset metadata
        dataset_metadata_path = os.path.join(output_dir, f'{scenario_type}_dataset_metadata.json')
        with open(dataset_metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            s3_key = f"datasets/{scenario_type}/{os.path.basename(output_dir)}_metadata.json"
            self.s3_service.upload_file(dataset_metadata_path, s3_key)
            self.logger.info(f"Uploaded dataset metadata to S3: {s3_key}")
        
        self.logger.info(f"Generated {num_scenarios} {scenario_type} scenarios")
        return dataset_metadata
    
    def _generate_scenarios_parallel(self,
                                   scenario_type: str,
                                   num_scenarios: int,
                                   scenario_params: Dict[str, Any],
                                   output_dir: str,
                                   max_workers: int = 4,
                                   seed: Optional[int] = None,
                                   generator: Optional[ScenarioGenerator] = None) -> Dict[str, Any]:
        """
        Generate scenarios in parallel.
        
        Args:
            scenario_type: Type of scenario
            num_scenarios: Number of scenarios to generate
            scenario_params: Parameters for the scenarios
            output_dir: Directory to save the generated data
            max_workers: Maximum number of parallel workers
            seed: Optional random seed for reproducibility
            generator: Optional specific generator to use
            
        Returns:
            Dictionary with metadata about the generated dataset
        """
        # Use default generator if not provided
        if generator is None:
            generator = self.scenario_generator
        
        # Get scenario duration and sample rate from config
        duration_seconds = self.config.get('scenario_duration', 3600)
        sample_rate_hz = self.config.get('sample_rate', 1.0)
        
        # Generate dataset metadata
        dataset_metadata = {
            'dataset_name': f"{scenario_type}_scenarios",
            'num_scenarios': num_scenarios,
            'scenario_type': scenario_type,
            'duration_seconds': duration_seconds,
            'sample_rate_hz': sample_rate_hz,
            'creation_date': datetime.now().isoformat(),
            'scenarios': []
        }
        
        # Define worker function
        def generate_scenario_worker(i):
            try:
                # Generate start time (random time in the past week)
                start_time = datetime.now() - timedelta(days=np.random.randint(0, 7))
                
                # Generate scenario
                scenario_data = generator.generate_scenario(
                    start_time=start_time,
                    duration_seconds=duration_seconds,
                    sample_rate_hz=sample_rate_hz,
                    scenario_type=scenario_type,
                    scenario_params=scenario_params,
                    seed=seed + i if seed is not None else None
                )
                
                # Save scenario
                scenario_dir = os.path.join(output_dir, f'{scenario_type}_scenario_{i:04d}')
                generator.save_scenario(scenario_data, scenario_dir)
                
                # Return scenario metadata
                scenario_metadata = scenario_data['metadata'].copy()
                scenario_metadata['scenario_id'] = i
                scenario_metadata['scenario_dir'] = scenario_dir
                return scenario_metadata
            except Exception as e:
                self.logger.error(f"Error generating {scenario_type} scenario {i}: {str(e)}")
                return None
        
        # Generate scenarios in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_scenario_worker, i) for i in range(num_scenarios)]
            
            # Process results with progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=num_scenarios, 
                              desc=f"Generating {scenario_type} scenarios"):
                scenario_metadata = future.result()
                if scenario_metadata is not None:
                    dataset_metadata['scenarios'].append(scenario_metadata)
        
        # Save dataset metadata
        dataset_metadata_path = os.path.join(output_dir, f'{scenario_type}_dataset_metadata.json')
        with open(dataset_metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            s3_key = f"datasets/{scenario_type}/{os.path.basename(output_dir)}_metadata.json"
            self.s3_service.upload_file(dataset_metadata_path, s3_key)
            self.logger.info(f"Uploaded dataset metadata to S3: {s3_key}")
        
        self.logger.info(f"Generated {len(dataset_metadata['scenarios'])}/{num_scenarios} {scenario_type} scenarios")
        return dataset_metadata
    
    def generate_complete_dataset(self, 
                                output_dir: str,
                                use_aws_batch: bool = False,
                                seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a complete dataset with normal operation, fire scenarios, and false positives.
        
        Args:
            output_dir: Directory to save the generated data
            use_aws_batch: Whether to use AWS Batch for parallel generation
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary with metadata about the generated dataset
        """
        self.logger.info("Generating complete dataset")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Get configuration parameters
        num_normal_hours = self.config['num_normal_hours']
        num_fire_scenarios = self.config['num_fire_scenarios']
        num_false_positive_scenarios = self.config['num_false_positive_scenarios']
        
        # Generate dataset metadata
        dataset_metadata = {
            'dataset_name': "complete_fire_prediction_dataset",
            'creation_date': datetime.now().isoformat(),
            'components': []
        }
        
        # If using AWS Batch, submit jobs and return
        if use_aws_batch and self.config.get('aws_integration', False):
            return self._generate_dataset_with_aws_batch(
                output_dir=output_dir,
                num_normal_hours=num_normal_hours,
                num_fire_scenarios=num_fire_scenarios,
                num_false_positive_scenarios=num_false_positive_scenarios,
                seed=seed
            )
        
        # Generate normal operation data
        normal_dir = os.path.join(output_dir, 'normal')
        normal_metadata = self.generate_normal_operation_data(
            num_hours=num_normal_hours,
            output_dir=normal_dir,
            parallel=True,
            seed=seed
        )
        dataset_metadata['components'].append({
            'type': 'normal',
            'metadata_path': os.path.join(normal_dir, 'normal_dataset_metadata.json'),
            'num_scenarios': num_normal_hours
        })
        
        # Generate fire scenarios
        fire_types = ['electrical', 'chemical', 'smoldering', 'rapid_combustion']
        for fire_type in fire_types:
            fire_dir = os.path.join(output_dir, fire_type)
            fire_metadata = self.generate_fire_scenarios(
                fire_type=fire_type,
                num_scenarios=num_fire_scenarios,
                output_dir=fire_dir,
                parallel=True,
                seed=seed + hash(fire_type) if seed is not None else None
            )
            dataset_metadata['components'].append({
                'type': fire_type,
                'metadata_path': os.path.join(fire_dir, f'{fire_type}_dataset_metadata.json'),
                'num_scenarios': num_fire_scenarios
            })
        
        # Generate false positive scenarios
        false_positive_types = ['cooking', 'heating', 'other']
        for fp_type in false_positive_types:
            fp_dir = os.path.join(output_dir, f'false_positive_{fp_type}')
            fp_metadata = self.generate_false_positive_scenarios(
                false_positive_type=fp_type,
                num_scenarios=num_false_positive_scenarios,
                output_dir=fp_dir,
                parallel=True,
                seed=seed + hash(fp_type) if seed is not None else None
            )
            dataset_metadata['components'].append({
                'type': f'false_positive_{fp_type}',
                'metadata_path': os.path.join(fp_dir, 'false_positive_dataset_metadata.json'),
                'num_scenarios': num_false_positive_scenarios
            })
        
        # Save dataset metadata
        dataset_metadata_path = os.path.join(output_dir, 'complete_dataset_metadata.json')
        with open(dataset_metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            s3_key = f"datasets/complete/{os.path.basename(output_dir)}_metadata.json"
            self.s3_service.upload_file(dataset_metadata_path, s3_key)
            self.logger.info(f"Uploaded complete dataset metadata to S3: {s3_key}")
        
        self.logger.info("Complete dataset generation finished")
        return dataset_metadata
    
    def _generate_dataset_with_aws_batch(self,
                                       output_dir: str,
                                       num_normal_hours: int,
                                       num_fire_scenarios: int,
                                       num_false_positive_scenarios: int,
                                       seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a complete dataset using AWS Batch for parallel processing.
        
        Args:
            output_dir: Directory to save the generated data
            num_normal_hours: Number of hours of normal operation data
            num_fire_scenarios: Number of fire scenarios per type
            num_false_positive_scenarios: Number of false positive scenarios per type
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary with metadata about the submitted AWS Batch jobs
        """
        self.logger.info("Submitting AWS Batch jobs for dataset generation")
        
        # Check if AWS Batch configuration is available
        if 'aws_batch_config' not in self.config:
            raise ValueError("aws_batch_config is required for AWS Batch integration")
        
        batch_config = self.config['aws_batch_config']
        required_batch_params = ['job_queue', 'job_definition', 'container_overrides']
        for param in required_batch_params:
            if param not in batch_config:
                raise ValueError(f"Missing required AWS Batch parameter: {param}")
        
        # Initialize AWS Batch client
        batch_client = boto3.client('batch')
        
        # Generate dataset metadata
        dataset_metadata = {
            'dataset_name': "complete_fire_prediction_dataset",
            'creation_date': datetime.now().isoformat(),
            'aws_batch_jobs': []
        }
        
        # Generate a unique dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Submit job for normal operation data
        normal_job_name = f"normal-data-{dataset_id}"
        normal_job = batch_client.submit_job(
            jobName=normal_job_name,
            jobQueue=batch_config['job_queue'],
            jobDefinition=batch_config['job_definition'],
            containerOverrides={
                **batch_config['container_overrides'],
                'command': [
                    'python', '-m', 'src.data_generation.batch_jobs.generate_normal_data',
                    '--output-dir', f"{output_dir}/normal",
                    '--num-hours', str(num_normal_hours),
                    '--seed', str(seed) if seed is not None else '42',
                    '--s3-bucket', self.config['aws_config']['default_bucket']
                ]
            }
        )
        dataset_metadata['aws_batch_jobs'].append({
            'type': 'normal',
            'job_id': normal_job['jobId'],
            'job_name': normal_job_name,
            'output_dir': f"{output_dir}/normal"
        })
        
        # Submit jobs for fire scenarios
        fire_types = ['electrical', 'chemical', 'smoldering', 'rapid_combustion']
        for fire_type in fire_types:
            job_name = f"{fire_type}-fire-{dataset_id}"
            job = batch_client.submit_job(
                jobName=job_name,
                jobQueue=batch_config['job_queue'],
                jobDefinition=batch_config['job_definition'],
                containerOverrides={
                    **batch_config['container_overrides'],
                    'command': [
                        'python', '-m', 'src.data_generation.batch_jobs.generate_fire_data',
                        '--output-dir', f"{output_dir}/{fire_type}",
                        '--fire-type', fire_type,
                        '--num-scenarios', str(num_fire_scenarios),
                        '--seed', str(seed + hash(fire_type)) if seed is not None else str(hash(fire_type)),
                        '--s3-bucket', self.config['aws_config']['default_bucket']
                    ]
                }
            )
            dataset_metadata['aws_batch_jobs'].append({
                'type': fire_type,
                'job_id': job['jobId'],
                'job_name': job_name,
                'output_dir': f"{output_dir}/{fire_type}"
            })
        
        # Submit jobs for false positive scenarios
        false_positive_types = ['cooking', 'heating', 'other']
        for fp_type in false_positive_types:
            job_name = f"false-positive-{fp_type}-{dataset_id}"
            job = batch_client.submit_job(
                jobName=job_name,
                jobQueue=batch_config['job_queue'],
                jobDefinition=batch_config['job_definition'],
                containerOverrides={
                    **batch_config['container_overrides'],
                    'command': [
                        'python', '-m', 'src.data_generation.batch_jobs.generate_false_positive_data',
                        '--output-dir', f"{output_dir}/false_positive_{fp_type}",
                        '--false-positive-type', fp_type,
                        '--num-scenarios', str(num_false_positive_scenarios),
                        '--seed', str(seed + hash(fp_type)) if seed is not None else str(hash(fp_type)),
                        '--s3-bucket', self.config['aws_config']['default_bucket']
                    ]
                }
            )
            dataset_metadata['aws_batch_jobs'].append({
                'type': f'false_positive_{fp_type}',
                'job_id': job['jobId'],
                'job_name': job_name,
                'output_dir': f"{output_dir}/false_positive_{fp_type}"
            })
        
        # Save dataset metadata
        dataset_metadata_path = os.path.join(output_dir, 'aws_batch_jobs_metadata.json')
        with open(dataset_metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            s3_key = f"datasets/aws_batch/{os.path.basename(output_dir)}_jobs_metadata.json"
            self.s3_service.upload_file(dataset_metadata_path, s3_key)
            self.logger.info(f"Uploaded AWS Batch jobs metadata to S3: {s3_key}")
        
        self.logger.info(f"Submitted {len(dataset_metadata['aws_batch_jobs'])} AWS Batch jobs")
        return dataset_metadata
    
    def monitor_aws_batch_jobs(self, jobs_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor the status of AWS Batch jobs.
        
        Args:
            jobs_metadata: Metadata about the submitted AWS Batch jobs
            
        Returns:
            Dictionary