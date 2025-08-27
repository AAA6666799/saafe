"""
Data validator for synthetic fire data.

This module provides functionality for validating the quality and consistency
of generated synthetic datasets.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..aws.s3.service import S3ServiceImpl


class DataValidator:
    """
    Class for validating synthetic datasets.
    
    This class provides methods for performing quality control on generated datasets,
    validating physical consistency, checking for completeness, and generating
    validation reports.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data validator with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.validate_config()
        
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
        # Check AWS integration
        if self.config.get('aws_integration', False):
            if 'aws_config' not in self.config:
                raise ValueError("aws_config is required when aws_integration is enabled")
            
            if 'default_bucket' not in self.config['aws_config']:
                raise ValueError("default_bucket is required in aws_config")
        
        # Set default values if not provided
        if 'validation_thresholds' not in self.config:
            self.config['validation_thresholds'] = {
                'temperature_min': -20.0,  # Celsius
                'temperature_max': 1000.0,  # Celsius
                'humidity_min': 0.0,  # Percentage
                'humidity_max': 100.0,  # Percentage
                'pressure_min': 800.0,  # hPa
                'pressure_max': 1100.0,  # hPa
                'gas_concentration_min': 0.0,  # PPM
                'gas_concentration_max': 10000.0,  # PPM
                'voc_min': 0.0,  # PPB
                'voc_max': 10000.0,  # PPB
                'max_missing_data_percentage': 5.0,  # Percentage
                'max_outlier_percentage': 2.0,  # Percentage
            }
    
    def validate_dataset(self, 
                        dataset_dir: str, 
                        output_dir: Optional[str] = None,
                        generate_plots: bool = True) -> Dict[str, Any]:
        """
        Validate a dataset and generate a validation report.
        
        Args:
            dataset_dir: Directory containing the dataset
            output_dir: Directory to save the validation report (default: dataset_dir/validation)
            generate_plots: Whether to generate validation plots
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Validating dataset in {dataset_dir}")
        
        # Set output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(dataset_dir, 'validation')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset metadata
        metadata_path = os.path.join(dataset_dir, 'complete_dataset_metadata.json')
        if not os.path.exists(metadata_path):
            # Try to find any metadata file
            metadata_files = [f for f in os.listdir(dataset_dir) if f.endswith('_metadata.json')]
            if metadata_files:
                metadata_path = os.path.join(dataset_dir, metadata_files[0])
            else:
                raise FileNotFoundError(f"No metadata file found in {dataset_dir}")
        
        with open(metadata_path, 'r') as f:
            dataset_metadata = json.load(f)
        
        # Initialize validation results
        validation_results = {
            'dataset_name': dataset_metadata.get('dataset_name', 'unknown'),
            'dataset_dir': dataset_dir,
            'validation_date': datetime.now().isoformat(),
            'overall_status': 'PENDING',
            'summary': {},
            'components': [],
            'issues': []
        }
        
        # Collect all scenarios from all components
        all_scenarios = []
        scenario_types = {}
        
        if 'components' in dataset_metadata:
            # Process complete dataset metadata
            for component in dataset_metadata['components']:
                component_type = component['type']
                component_metadata_path = component['metadata_path']
                
                if os.path.exists(component_metadata_path):
                    with open(component_metadata_path, 'r') as f:
                        component_metadata = json.load(f)
                        
                    if 'scenarios' in component_metadata:
                        for scenario in component_metadata['scenarios']:
                            scenario['scenario_type'] = component_type
                            all_scenarios.append(scenario)
                            
                            # Track scenario types
                            if component_type not in scenario_types:
                                scenario_types[component_type] = []
                            scenario_types[component_type].append(scenario)
                
                # Validate component
                component_results = self._validate_component(component_type, component_metadata_path)
                validation_results['components'].append(component_results)
        elif 'scenarios' in dataset_metadata:
            # Process single component metadata
            component_type = dataset_metadata.get('scenario_type', 'unknown')
            for scenario in dataset_metadata['scenarios']:
                scenario['scenario_type'] = component_type
                all_scenarios.append(scenario)
                
                # Track scenario types
                if component_type not in scenario_types:
                    scenario_types[component_type] = []
                scenario_types[component_type].append(scenario)
            
            # Validate component
            component_results = self._validate_component(component_type, metadata_path)
            validation_results['components'].append(component_results)
        
        self.logger.info(f"Found {len(all_scenarios)} scenarios of {len(scenario_types)} types")
        
        # Validate individual scenarios
        scenario_results = []
        for scenario in tqdm(all_scenarios, desc="Validating scenarios"):
            if 'scenario_dir' in scenario:
                scenario_dir = scenario['scenario_dir']
                scenario_type = scenario.get('scenario_type', 'unknown')
                scenario_id = scenario.get('scenario_id', 'unknown')
                
                # Validate scenario
                result = self._validate_scenario(scenario_dir, scenario_type, scenario_id)
                scenario_results.append(result)
        
        # Compute summary statistics
        total_scenarios = len(scenario_results)
        valid_scenarios = sum(1 for r in scenario_results if r['status'] == 'VALID')
        invalid_scenarios = total_scenarios - valid_scenarios
        
        validation_results['summary'] = {
            'total_scenarios': total_scenarios,
            'valid_scenarios': valid_scenarios,
            'invalid_scenarios': invalid_scenarios,
            'valid_percentage': (valid_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0,
            'scenario_type_counts': {t: len(s) for t, s in scenario_types.items()},
            'issues_by_type': self._count_issues_by_type(scenario_results)
        }
        
        # Set overall status
        if invalid_scenarios == 0:
            validation_results['overall_status'] = 'VALID'
        elif invalid_scenarios / total_scenarios <= 0.05:  # Less than 5% invalid
            validation_results['overall_status'] = 'VALID_WITH_ISSUES'
        else:
            validation_results['overall_status'] = 'INVALID'
        
        # Add scenario results
        validation_results['scenario_results'] = scenario_results
        
        # Generate validation plots if requested
        if generate_plots:
            plot_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            # Generate plots
            self._generate_validation_plots(validation_results, plot_dir)
            validation_results['plots_dir'] = plot_dir
        
        # Save validation results
        results_path = os.path.join(output_dir, 'validation_results.json')
        with open(results_path, 'w') as f:
            # Remove scenario_results to keep the file size manageable
            results_to_save = validation_results.copy()
            results_to_save.pop('scenario_results', None)
            json.dump(results_to_save, f, indent=2)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            s3_key = f"validation/{os.path.basename(dataset_dir)}_validation_results.json"
            self.s3_service.upload_file(results_path, s3_key)
            self.logger.info(f"Uploaded validation results to S3: {s3_key}")
        
        self.logger.info(f"Dataset validation complete: {validation_results['overall_status']}")
        return validation_results
    
    def _validate_component(self, component_type: str, metadata_path: str) -> Dict[str, Any]:
        """
        Validate a dataset component.
        
        Args:
            component_type: Type of component
            metadata_path: Path to component metadata file
            
        Returns:
            Dictionary with component validation results
        """
        self.logger.debug(f"Validating component: {component_type}")
        
        # Initialize component results
        component_results = {
            'component_type': component_type,
            'metadata_path': metadata_path,
            'status': 'PENDING',
            'issues': []
        }
        
        try:
            # Check if metadata file exists
            if not os.path.exists(metadata_path):
                component_results['status'] = 'INVALID'
                component_results['issues'].append({
                    'type': 'MISSING_METADATA',
                    'message': f"Metadata file not found: {metadata_path}"
                })
                return component_results
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check required metadata fields
            required_fields = ['scenarios']
            for field in required_fields:
                if field not in metadata:
                    component_results['status'] = 'INVALID'
                    component_results['issues'].append({
                        'type': 'MISSING_METADATA_FIELD',
                        'message': f"Required metadata field missing: {field}"
                    })
            
            # Check scenarios
            if 'scenarios' in metadata:
                if not isinstance(metadata['scenarios'], list):
                    component_results['status'] = 'INVALID'
                    component_results['issues'].append({
                        'type': 'INVALID_METADATA_FORMAT',
                        'message': f"Scenarios field is not a list"
                    })
                elif len(metadata['scenarios']) == 0:
                    component_results['status'] = 'INVALID'
                    component_results['issues'].append({
                        'type': 'EMPTY_COMPONENT',
                        'message': f"Component has no scenarios"
                    })
            
            # Set status based on issues
            if not component_results['issues']:
                component_results['status'] = 'VALID'
            
            return component_results
        
        except Exception as e:
            component_results['status'] = 'INVALID'
            component_results['issues'].append({
                'type': 'VALIDATION_ERROR',
                'message': f"Error validating component: {str(e)}"
            })
            return component_results
    
    def _validate_scenario(self, scenario_dir: str, scenario_type: str, scenario_id: Any) -> Dict[str, Any]:
        """
        Validate a single scenario.
        
        Args:
            scenario_dir: Directory containing the scenario
            scenario_type: Type of scenario
            scenario_id: Scenario ID
            
        Returns:
            Dictionary with scenario validation results
        """
        # Initialize scenario results
        scenario_results = {
            'scenario_type': scenario_type,
            'scenario_id': scenario_id,
            'scenario_dir': scenario_dir,
            'status': 'PENDING',
            'issues': []
        }
        
        try:
            # Check if scenario directory exists
            if not os.path.exists(scenario_dir):
                scenario_results['status'] = 'INVALID'
                scenario_results['issues'].append({
                    'type': 'MISSING_DIRECTORY',
                    'message': f"Scenario directory not found: {scenario_dir}"
                })
                return scenario_results
            
            # Check for required files
            required_files = ['metadata.json', 'scenario_definition.json']
            for file in required_files:
                file_path = os.path.join(scenario_dir, file)
                if not os.path.exists(file_path):
                    scenario_results['status'] = 'INVALID'
                    scenario_results['issues'].append({
                        'type': 'MISSING_FILE',
                        'message': f"Required file missing: {file}"
                    })
            
            # Load metadata
            metadata_path = os.path.join(scenario_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check required metadata fields
                required_fields = ['scenario_type', 'start_time', 'end_time', 'duration']
                for field in required_fields:
                    if field not in metadata:
                        scenario_results['status'] = 'INVALID'
                        scenario_results['issues'].append({
                            'type': 'MISSING_METADATA_FIELD',
                            'message': f"Required metadata field missing: {field}"
                        })
            
            # Check for data directories
            data_dirs = ['thermal', 'gas', 'environmental']
            for data_dir in data_dirs:
                dir_path = os.path.join(scenario_dir, data_dir)
                if not os.path.exists(dir_path):
                    scenario_results['status'] = 'INVALID'
                    scenario_results['issues'].append({
                        'type': 'MISSING_DATA_DIRECTORY',
                        'message': f"Data directory missing: {data_dir}"
                    })
            
            # Validate combined data file
            combined_data_path = os.path.join(scenario_dir, 'combined_data.csv')
            if os.path.exists(combined_data_path):
                try:
                    df = pd.read_csv(combined_data_path)
                    
                    # Check for required columns
                    required_columns = ['timestamp']
                    for col in required_columns:
                        if col not in df.columns:
                            scenario_results['issues'].append({
                                'type': 'MISSING_DATA_COLUMN',
                                'message': f"Required column missing in combined data: {col}"
                            })
                    
                    # Check for missing values
                    missing_percentage = df.isnull().mean().max() * 100
                    if missing_percentage > self.config['validation_thresholds']['max_missing_data_percentage']:
                        scenario_results['issues'].append({
                            'type': 'EXCESSIVE_MISSING_DATA',
                            'message': f"Excessive missing data: {missing_percentage:.2f}%"
                        })
                    
                    # Validate physical consistency
                    self._validate_physical_consistency(df, scenario_results)
                    
                except Exception as e:
                    scenario_results['issues'].append({
                        'type': 'DATA_LOADING_ERROR',
                        'message': f"Error loading combined data: {str(e)}"
                    })
            else:
                scenario_results['issues'].append({
                    'type': 'MISSING_COMBINED_DATA',
                    'message': f"Combined data file missing: {combined_data_path}"
                })
            
            # Set status based on issues
            if not scenario_results['issues']:
                scenario_results['status'] = 'VALID'
            else:
                scenario_results['status'] = 'INVALID'
            
            return scenario_results
        
        except Exception as e:
            scenario_results['status'] = 'INVALID'
            scenario_results['issues'].append({
                'type': 'VALIDATION_ERROR',
                'message': f"Error validating scenario: {str(e)}"
            })
            return scenario_results
    
    def _validate_physical_consistency(self, df: pd.DataFrame, scenario_results: Dict[str, Any]) -> None:
        """
        Validate physical consistency of the data.
        
        Args:
            df: DataFrame containing the data
            scenario_results: Dictionary to update with validation results
        """
        thresholds = self.config['validation_thresholds']
        
        # Check temperature range
        if any(col.startswith('temperature') or col.endswith('temperature') for col in df.columns):
            temp_cols = [col for col in df.columns if col.startswith('temperature') or col.endswith('temperature')]
            for col in temp_cols:
                if df[col].min() < thresholds['temperature_min']:
                    scenario_results['issues'].append({
                        'type': 'TEMPERATURE_BELOW_THRESHOLD',
                        'message': f"Temperature below threshold in {col}: {df[col].min():.2f} < {thresholds['temperature_min']}"
                    })
                
                if df[col].max() > thresholds['temperature_max']:
                    scenario_results['issues'].append({
                        'type': 'TEMPERATURE_ABOVE_THRESHOLD',
                        'message': f"Temperature above threshold in {col}: {df[col].max():.2f} > {thresholds['temperature_max']}"
                    })
        
        # Check humidity range
        if any(col.startswith('humidity') or col.endswith('humidity') for col in df.columns):
            humidity_cols = [col for col in df.columns if col.startswith('humidity') or col.endswith('humidity')]
            for col in humidity_cols:
                if df[col].min() < thresholds['humidity_min']:
                    scenario_results['issues'].append({
                        'type': 'HUMIDITY_BELOW_THRESHOLD',
                        'message': f"Humidity below threshold in {col}: {df[col].min():.2f} < {thresholds['humidity_min']}"
                    })
                
                if df[col].max() > thresholds['humidity_max']:
                    scenario_results['issues'].append({
                        'type': 'HUMIDITY_ABOVE_THRESHOLD',
                        'message': f"Humidity above threshold in {col}: {df[col].max():.2f} > {thresholds['humidity_max']}"
                    })
        
        # Check pressure range
        if any(col.startswith('pressure') or col.endswith('pressure') for col in df.columns):
            pressure_cols = [col for col in df.columns if col.startswith('pressure') or col.endswith('pressure')]
            for col in pressure_cols:
                if df[col].min() < thresholds['pressure_min']:
                    scenario_results['issues'].append({
                        'type': 'PRESSURE_BELOW_THRESHOLD',
                        'message': f"Pressure below threshold in {col}: {df[col].min():.2f} < {thresholds['pressure_min']}"
                    })
                
                if df[col].max() > thresholds['pressure_max']:
                    scenario_results['issues'].append({
                        'type': 'PRESSURE_ABOVE_THRESHOLD',
                        'message': f"Pressure above threshold in {col}: {df[col].max():.2f} > {thresholds['pressure_max']}"
                    })
        
        # Check gas concentration range
        if any('concentration' in col for col in df.columns):
            gas_cols = [col for col in df.columns if 'concentration' in col]
            for col in gas_cols:
                if df[col].min() < thresholds['gas_concentration_min']:
                    scenario_results['issues'].append({
                        'type': 'GAS_CONCENTRATION_BELOW_THRESHOLD',
                        'message': f"Gas concentration below threshold in {col}: {df[col].min():.2f} < {thresholds['gas_concentration_min']}"
                    })
                
                if df[col].max() > thresholds['gas_concentration_max']:
                    scenario_results['issues'].append({
                        'type': 'GAS_CONCENTRATION_ABOVE_THRESHOLD',
                        'message': f"Gas concentration above threshold in {col}: {df[col].max():.2f} > {thresholds['gas_concentration_max']}"
                    })
        
        # Check VOC range
        if any(col.startswith('voc') or col.endswith('voc') for col in df.columns):
            voc_cols = [col for col in df.columns if col.startswith('voc') or col.endswith('voc')]
            for col in voc_cols:
                if df[col].min() < thresholds['voc_min']:
                    scenario_results['issues'].append({
                        'type': 'VOC_BELOW_THRESHOLD',
                        'message': f"VOC below threshold in {col}: {df[col].min():.2f} < {thresholds['voc_min']}"
                    })
                
                if df[col].max() > thresholds['voc_max']:
                    scenario_results['issues'].append({
                        'type': 'VOC_ABOVE_THRESHOLD',
                        'message': f"VOC above threshold in {col}: {df[col].max():.2f} > {thresholds['voc_max']}"
                    })
        
        # Check for outliers
        for col in df.select_dtypes(include=[np.number]).columns:
            # Skip timestamp columns
            if col == 'timestamp' or 'time' in col.lower():
                continue
            
            # Calculate IQR
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            # Define outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_percentage = len(outliers) / len(df) * 100
            
            if outlier_percentage > thresholds['max_outlier_percentage']:
                scenario_results['issues'].append({
                    'type': 'EXCESSIVE_OUTLIERS',
                    'message': f"Excessive outliers in {col}: {outlier_percentage:.2f}%"
                })
    
    def _count_issues_by_type(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Count issues by type across all scenarios.
        
        Args:
            scenario_results: List of scenario validation results
            
        Returns:
            Dictionary mapping issue types to counts
        """
        issue_counts = {}
        for result in scenario_results:
            for issue in result.get('issues', []):
                issue_type = issue['type']
                if issue_type not in issue_counts:
                    issue_counts[issue_type] = 0
                issue_counts[issue_type] += 1
        return issue_counts
    
    def _generate_validation_plots(self, validation_results: Dict[str, Any], plot_dir: str) -> None:
        """
        Generate validation plots.
        
        Args:
            validation_results: Validation results
            plot_dir: Directory to save plots
        """
        self.logger.info("Generating validation plots")
        
        # Set plot style
        sns.set(style="whitegrid")
        
        # Plot scenario type distribution
        if 'summary' in validation_results and 'scenario_type_counts' in validation_results['summary']:
            scenario_types = validation_results['summary']['scenario_type_counts']
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x=list(scenario_types.keys()), y=list(scenario_types.values()))
            plt.title('Scenario Type Distribution')
            plt.xlabel('Scenario Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'scenario_type_distribution.png'))
            plt.close()
        
        # Plot validation status distribution
        if 'scenario_results' in validation_results:
            status_counts = {}
            for result in validation_results['scenario_results']:
                status = result['status']
                if status not in status_counts:
                    status_counts[status] = 0
                status_counts[status] += 1
            
            plt.figure(figsize=(8, 6))
            sns.barplot(x=list(status_counts.keys()), y=list(status_counts.values()))
            plt.title('Validation Status Distribution')
            plt.xlabel('Status')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'validation_status_distribution.png'))
            plt.close()
        
        # Plot issue type distribution
        if 'summary' in validation_results and 'issues_by_type' in validation_results['summary']:
            issues = validation_results['summary']['issues_by_type']
            
            if issues:
                plt.figure(figsize=(14, 6))
                sns.barplot(x=list(issues.keys()), y=list(issues.values()))
                plt.title('Issue Type Distribution')
                plt.xlabel('Issue Type')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'issue_type_distribution.png'))
                plt.close()
        
        # Upload plots to S3 if AWS integration is enabled
        if self.s3_service is not None:
            for filename in os.listdir(plot_dir):
                if filename.endswith('.png'):
                    file_path = os.path.join(plot_dir, filename)
                    s3_key = f"validation/plots/{os.path.basename(plot_dir)}_{filename}"
                    self.s3_service.upload_file(file_path, s3_key)
    
    def generate_validation_report(self, validation_results: Dict[str, Any], output_path: str) -> None:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_results: Validation results
            output_path: Path to save the report
        """
        self.logger.info(f"Generating validation report: {output_path}")
        
        # Create report directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Generate report
        with open(output_path, 'w') as f:
            f.write(f"# Validation Report: {validation_results.get('dataset_name', 'Unknown Dataset')}\n\n")
            f.write(f"**Date:** {validation_results.get('validation_date', datetime.now().isoformat())}\n\n")
            f.write(f"**Overall Status:** {validation_results.get('overall_status', 'UNKNOWN')}\n\n")
            
            # Write summary
            if 'summary' in validation_results:
                summary = validation_results['summary']
                f.write("## Summary\n\n")
                f.write(f"- Total Scenarios: {summary.get('total_scenarios', 0)}\n")
                f.write(f"- Valid Scenarios: {summary.get('valid_scenarios', 0)} ({summary.get('valid_percentage', 0):.2f}%)\n")
                f.write(f"- Invalid Scenarios: {summary.get('invalid_scenarios', 0)}\n\n")
                
                # Write scenario type counts
                if 'scenario_type_counts' in summary:
                    f.write("### Scenario Type Distribution\n\n")
                    f.write("| Scenario Type | Count |\n")
                    f.write("|--------------|-------|\n")
                    for scenario_type, count in summary['scenario_type_counts'].items():
                        f.write(f"| {scenario_type} | {count} |\n")
                    f.write("\n")
                
                # Write issue counts
                if 'issues_by_type' in summary and summary['issues_by_type']:
                    f.write("### Issue Distribution\n\n")
                    f.write("| Issue Type | Count |\n")
                    f.write("|------------|-------|\n")
                    for issue_type, count in summary['issues_by_type'].items():
                        f.write(f"| {issue_type} | {count} |\n")
                    f.write("\n")
            
            # Write component results
            if 'components' in validation_results:
                f.write("## Component Validation\n\n")
                for component in validation_results['components']:
                    f.write(f"### {component.get('component_type', 'Unknown Component')}\n\n")
                    f.write(f"**Status:** {component.get('status', 'UNKNOWN')}\n\n")
                    
                    if 'issues' in component and component['issues']:
                        f.write("**Issues:**\n\n")
                        for issue in component['issues']:
                            f.write(f"- **{issue.get('type', 'Unknown Issue')}:** {issue.get('message', '')}\n")
                        f.write("\n")
            
            # Write plots section if available
            if 'plots_dir' in validation_results:
                f.write("## Validation Plots\n\n")
                plots_dir = validation_results['plots_dir']
                for filename in os.listdir(plots_dir):
                    if filename.endswith('.png'):
                        plot_path = os.path.join(plots_dir, filename)
                        plot_name = os.path.splitext(filename)[0].replace('_', ' ').title()
                        f.write(f"### {plot_name}\n\n")
                        f.write(f"![{plot_name}]({os.path.relpath(plot_path, os.path.dirname(output_path))})\n\n")
            # Write recommendations
            f.write("## Recommendations\n\n")
            
            # Add recommendations based on validation results
            if validation_results.get('overall_status') == 'VALID':
                f.write("- The dataset is valid and ready for use in training and evaluation.\n")
            elif validation_results.get('overall_status') == 'VALID_WITH_ISSUES':
                f.write("- The dataset is generally valid but has some issues that should be addressed:\n")
                if 'summary' in validation_results and 'issues_by_type' in validation_results['summary']:
                    for issue_type, count in validation_results['summary']['issues_by_type'].items():
                        f.write(f"  - Fix {count} instances of {issue_type}\n")
            else:
                f.write("- The dataset has significant issues that must be addressed before use:\n")
                if 'summary' in validation_results and 'issues_by_type' in validation_results['summary']:
                    for issue_type, count in validation_results['summary']['issues_by_type'].items():
                        f.write(f"  - Fix {count} instances of {issue_type}\n")
                f.write("- Consider regenerating problematic scenarios or adjusting generation parameters.\n")
            
            # Add general recommendations
            f.write("\n### General Recommendations\n\n")
            f.write("- Regularly validate datasets after generation to ensure quality.\n")
            f.write("- Monitor physical consistency across different sensor types.\n")
            f.write("- Ensure balanced representation of different scenario types in training data.\n")
            f.write("- Consider data augmentation for underrepresented scenario types.\n")
        
        # Upload report to S3 if AWS integration is enabled
        if self.s3_service is not None:
            s3_key = f"validation/reports/{os.path.basename(output_path)}"
            self.s3_service.upload_file(output_path, s3_key)
            self.logger.info(f"Uploaded validation report to S3: {s3_key}")