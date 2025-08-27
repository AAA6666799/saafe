"""
Implementation of the run_pipeline method for the ModelTrainingPipeline class.
"""

from typing import Dict, Any, Type, Optional, Union
import time
import os
from datetime import datetime

from ..base import FireModel


def run_pipeline(self, 
                data_path: str,
                model_classes: Dict[str, Type[FireModel]],
                model_configs: Dict[str, Dict[str, Any]],
                save_models: bool = True,
                to_s3: bool = False,
                deploy_best_model: bool = False,
                instance_type: str = 'ml.m5.large',
                initial_instance_count: int = 1) -> Dict[str, Any]:
    """
    Run the complete model training pipeline.
    
    Args:
        data_path: Path to the data file
        model_classes: Dictionary mapping model names to model classes
        model_configs: Dictionary mapping model names to model configurations
        save_models: Whether to save trained models
        to_s3: Whether to save models to S3
        deploy_best_model: Whether to deploy the best model to SageMaker
        instance_type: SageMaker instance type
        initial_instance_count: Number of initial instances
        
    Returns:
        Dictionary containing pipeline results
    """
    self.logger.info(f"Starting model training pipeline with run ID: {self.current_run_id}")
    
    # Record pipeline start time
    pipeline_start_time = time.time()
    
    # Step 1: Load data
    from_s3 = data_path.startswith('s3://')
    features, labels = self.load_data(data_path, from_s3=from_s3)
    
    # Step 2: Preprocess data
    processed_features, processed_labels = self.preprocess_data(features, labels)
    
    # Step 3: Split data
    split_data = self.split_data(processed_features, processed_labels)
    
    # Step 4: Train models
    for model_name, model_class in model_classes.items():
        self.logger.info(f"Processing model: {model_name}")
        
        if model_name not in model_configs:
            self.logger.warning(f"No configuration found for model {model_name}, using default")
            model_configs[model_name] = {}
        
        # Train the model
        model = self.train_model(model_class, model_configs[model_name], split_data)
    
    # Step 5: Evaluate models
    evaluation_results = self.evaluate_models(split_data)
    
    # Step 6: Select best model
    best_model_name, best_model = self.select_best_model(evaluation_results)
    
    # Step 7: Save models if requested
    saved_model_paths = {}
    if save_models:
        for model_name, model in self.trained_models.items():
            model_path = self.save_model(model, model_name, to_s3=to_s3)
            saved_model_paths[model_name] = model_path
    
    # Step 8: Deploy best model to SageMaker if requested
    endpoint_name = None
    if deploy_best_model and self.use_aws:
        endpoint_name = self.deploy_to_sagemaker(
            best_model, 
            best_model_name,
            instance_type=instance_type,
            initial_instance_count=initial_instance_count
        )
    
    # Record pipeline end time
    pipeline_end_time = time.time()
    pipeline_duration = pipeline_end_time - pipeline_start_time
    
    # Create results summary
    results = {
        'run_id': self.current_run_id,
        'timestamp': self.run_timestamp,
        'duration': pipeline_duration,
        'models_trained': list(self.trained_models.keys()),
        'best_model': best_model_name,
        'evaluation_results': evaluation_results,
        'saved_model_paths': saved_model_paths
    }
    
    if endpoint_name:
        results['sagemaker_endpoint'] = endpoint_name
    
    # Save results to file
    results_path = os.path.join(self.results_dir, f"pipeline_results_{self.current_run_id}.json")
    with open(results_path, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    self.logger.info(f"Completed model training pipeline in {pipeline_duration:.2f} seconds")
    self.logger.info(f"Best model: {best_model_name}")
    
    return results