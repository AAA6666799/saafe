"""
Model Training Pipeline for the fire prediction system.

This module implements the ModelTrainingPipeline class that handles the overall
model training workflow, including data loading, preprocessing, model training,
validation, hyperparameter tuning, model evaluation, and model versioning.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Type, Callable
from datetime import datetime
from pathlib import Path
import pickle
import uuid
import time
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import boto3
import joblib

from ..base import FireModel


class ModelTrainingPipeline:
    """
    Pipeline for training machine learning models for fire prediction.
    
    This class handles the entire model training workflow, including:
    - Data loading and preprocessing
    - Model training and validation
    - Hyperparameter tuning
    - Model evaluation and selection
    - Model versioning and tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model training pipeline.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Set up directories
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.models_dir = Path(config.get('models_dir', 'models'))
        self.results_dir = Path(config.get('results_dir', 'results'))
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up AWS integration if enabled
        self.use_aws = config.get('use_aws', False)
        if self.use_aws:
            self._setup_aws()
        
        # Initialize tracking variables
        self.current_run_id = str(uuid.uuid4())
        self.run_timestamp = datetime.now().isoformat()
        self.metrics = {}
        self.trained_models = {}
        
        # Set up preprocessing options
        self.preprocessing_steps = config.get('preprocessing_steps', ['scaling'])
        self.scaling_method = config.get('scaling_method', 'standard')
        
        # Set up validation options
        self.validation_method = config.get('validation_method', 'train_test_split')
        self.test_size = config.get('test_size', 0.2)
        self.validation_size = config.get('validation_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.n_splits = config.get('n_splits', 5)
        
        # Set up hyperparameter tuning options
        self.tuning_method = config.get('tuning_method', 'grid')
        self.n_iter = config.get('n_iter', 10)
        self.cv = config.get('cv', 3)
        self.n_jobs = config.get('n_jobs', -1)
        
        # Set up model selection options
        self.selection_metric = config.get('selection_metric', 'f1_score')
        
        self.logger.info(f"Initialized ModelTrainingPipeline with run ID: {self.current_run_id}")
    
    def _setup_aws(self) -> None:
        """
        Set up AWS integration.
        """
        self.aws_region = self.config.get('aws_region', 'us-east-1')
        self.s3_bucket = self.config.get('s3_bucket')
        self.sagemaker_role = self.config.get('sagemaker_role')
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3', region_name=self.aws_region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.aws_region)
        
        self.logger.info(f"Set up AWS integration with region: {self.aws_region}")
    
    def load_data(self, 
                 data_path: Optional[str] = None, 
                 from_s3: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data for model training.
        
        Args:
            data_path: Path to the data file
            from_s3: Whether to load data from S3
            
        Returns:
            Tuple of (features, labels)
        """
        if data_path is None:
            data_path = self.config.get('data_path')
            if data_path is None:
                raise ValueError("No data path provided")
        
        self.logger.info(f"Loading data from {'S3' if from_s3 else 'local file system'}: {data_path}")
        
        if from_s3 and self.use_aws:
            # Parse S3 path
            s3_parts = data_path.replace('s3://', '').split('/')
            bucket = s3_parts[0]
            key = '/'.join(s3_parts[1:])
            
            # Download file to temporary location
            local_path = os.path.join(self.data_dir, os.path.basename(data_path))
            self.s3_client.download_file(bucket, key, local_path)
            data_path = local_path
        
        # Load data based on file extension
        file_ext = os.path.splitext(data_path)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(data_path)
        elif file_ext == '.parquet':
            df = pd.read_parquet(data_path)
        elif file_ext == '.json':
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Extract features and labels
        label_column = self.config.get('label_column', 'label')
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")
        
        features = df.drop(columns=[label_column])
        labels = df[label_column]
        
        self.logger.info(f"Loaded data with {features.shape[0]} samples and {features.shape[1]} features")
        
        return features, labels
    
    def preprocess_data(self, 
                       features: pd.DataFrame, 
                       labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data for model training.
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            
        Returns:
            Tuple of (preprocessed_features, preprocessed_labels)
        """
        self.logger.info("Preprocessing data")
        
        # Make a copy to avoid modifying the original data
        processed_features = features.copy()
        processed_labels = labels.copy()
        
        # Apply preprocessing steps
        for step in self.preprocessing_steps:
            if step == 'scaling':
                processed_features = self._apply_scaling(processed_features)
            elif step == 'encoding':
                processed_features = self._apply_encoding(processed_features)
            elif step == 'imputation':
                processed_features = self._apply_imputation(processed_features)
            elif step == 'feature_selection':
                processed_features = self._apply_feature_selection(processed_features, processed_labels)
        
        self.logger.info(f"Completed data preprocessing with {processed_features.shape[1]} features")
        
        return processed_features, processed_labels
    
    def _apply_scaling(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling to features.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Scaled features
        """
        numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) == 0:
            return features
        
        if self.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {self.scaling_method}")
        
        # Store the scaler for later use
        self.scaler = scaler
        
        # Apply scaling only to numeric columns
        features_scaled = features.copy()
        features_scaled[numeric_cols] = scaler.fit_transform(features[numeric_cols])
        
        return features_scaled
    
    def _apply_encoding(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply encoding to categorical features.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Encoded features
        """
        # Get categorical columns
        cat_cols = features.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols) == 0:
            return features
        
        # Apply one-hot encoding
        encoded_features = pd.get_dummies(features, columns=cat_cols, drop_first=True)
        
        return encoded_features
    
    def _apply_imputation(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation to handle missing values.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Imputed features
        """
        # Simple imputation strategy - replace with mean/mode
        numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = features.select_dtypes(include=['object', 'category']).columns
        
        imputed_features = features.copy()
        
        # Impute numeric columns with mean
        for col in numeric_cols:
            imputed_features[col].fillna(imputed_features[col].mean(), inplace=True)
        
        # Impute categorical columns with mode
        for col in cat_cols:
            imputed_features[col].fillna(imputed_features[col].mode()[0], inplace=True)
        
        return imputed_features
    
    def _apply_feature_selection(self, 
                               features: pd.DataFrame, 
                               labels: pd.Series) -> pd.DataFrame:
        """
        Apply feature selection.
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            
        Returns:
            Selected features
        """
        # Simple feature selection based on correlation with target
        if not self.config.get('use_feature_selection', False):
            return features
        
        # Only applicable for numeric features
        numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) == 0:
            return features
        
        # Calculate correlation with target for numeric features
        correlations = {}
        for col in numeric_cols:
            correlations[col] = abs(np.corrcoef(features[col], labels)[0, 1])
        
        # Select top N features or features with correlation above threshold
        threshold = self.config.get('feature_selection_threshold', 0.1)
        selected_features = [col for col, corr in correlations.items() if corr > threshold]
        
        # Add non-numeric columns back
        non_numeric_cols = [col for col in features.columns if col not in numeric_cols]
        selected_features.extend(non_numeric_cols)
        
        return features[selected_features]
    
    def split_data(self, 
                  features: pd.DataFrame, 
                  labels: pd.Series) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            
        Returns:
            Dictionary containing split data
        """
        self.logger.info("Splitting data into train, validation, and test sets")
        
        if self.validation_method == 'train_test_split':
            # First split into train+val and test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                features, labels, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=labels if self.config.get('stratify', True) else None
            )
            
            # Then split train+val into train and val
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=self.validation_size / (1 - self.test_size),
                random_state=self.random_state,
                stratify=y_train_val if self.config.get('stratify', True) else None
            )
            
            split_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test
            }
            
            self.logger.info(f"Data split: train={X_train.shape[0]} samples, "
                           f"validation={X_val.shape[0]} samples, "
                           f"test={X_test.shape[0]} samples")
            
            return split_data
        else:
            raise ValueError(f"Unsupported validation method: {self.validation_method}")
    
    def train_model(self, 
                   model_class: Type[FireModel], 
                   model_config: Dict[str, Any],
                   data: Dict[str, Union[pd.DataFrame, pd.Series]],
                   tune_hyperparameters: bool = True) -> FireModel:
        """
        Train a model with the given configuration.
        
        Args:
            model_class: Class of the model to train
            model_config: Configuration for the model
            data: Dictionary containing split data
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        model_name = model_class.__name__
        self.logger.info(f"Training model: {model_name}")
        
        # Extract data
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        
        # Perform hyperparameter tuning if requested
        if tune_hyperparameters and 'hyperparameter_grid' in model_config:
            self.logger.info(f"Performing hyperparameter tuning for {model_name}")
            best_params = self._tune_hyperparameters(model_class, model_config, X_train, y_train)
            
            # Update model config with best parameters
            model_config.update(best_params)
            self.logger.info(f"Best hyperparameters for {model_name}: {best_params}")
        
        # Initialize and train the model
        model = model_class(model_config)
        
        # Record training start time
        start_time = time.time()
        
        # Train the model
        training_metrics = model.train(X_train, y_train, validation_data=(X_val, y_val))
        
        # Record training end time
        end_time = time.time()
        training_time = end_time - start_time
        
        # Evaluate the model
        val_metrics = model.evaluate(X_val, y_val)
        
        # Store metrics
        self.metrics[model_name] = {
            'training': training_metrics,
            'validation': val_metrics,
            'training_time': training_time
        }
        
        # Store the trained model
        self.trained_models[model_name] = model
        
        self.logger.info(f"Completed training {model_name} in {training_time:.2f} seconds")
        self.logger.info(f"Validation metrics for {model_name}: {val_metrics}")
        
        return model
    
    def _tune_hyperparameters(self, 
                            model_class: Type[FireModel], 
                            model_config: Dict[str, Any],
                            X_train: pd.DataFrame,
                            y_train: pd.Series) -> Dict[str, Any]:
        """
        Tune hyperparameters for a model.
        
        Args:
            model_class: Class of the model to tune
            model_config: Configuration for the model
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of best hyperparameters
        """
        # Extract hyperparameter grid
        param_grid = model_config['hyperparameter_grid']
        
        # Create a wrapper for the model to use with scikit-learn
        class ModelWrapper:
            def __init__(self, model_class, base_config):
                self.model_class = model_class
                self.base_config = base_config
                self.model = None
            
            def set_params(self, **params):
                self.params = params
                return self
            
            def fit(self, X, y):
                # Create a new config with base config and hyperparameters
                config = {**self.base_config}
                for key, value in self.params.items():
                    config[key] = value
                
                # Initialize and train the model
                self.model = self.model_class(config)
                self.model.train(X, y)
                return self
            
            def predict(self, X):
                return self.model.predict(X)
            
            def score(self, X, y):
                # Use the primary metric for scoring
                predictions = self.model.predict(X)
                metric_func = self._get_metric_function()
                return metric_func(y, predictions)
            
            def _get_metric_function(self):
                metric = self.base_config.get('primary_metric', 'accuracy')
                if metric == 'accuracy':
                    return accuracy_score
                elif metric == 'precision':
                    return lambda y, p: precision_score(y, p, average='weighted')
                elif metric == 'recall':
                    return lambda y, p: recall_score(y, p, average='weighted')
                elif metric == 'f1':
                    return lambda y, p: f1_score(y, p, average='weighted')
                else:
                    return accuracy_score
        
        # Create the model wrapper
        model_wrapper = ModelWrapper(model_class, model_config)
        
        # Perform hyperparameter tuning
        if self.tuning_method == 'grid':
            search = GridSearchCV(
                model_wrapper,
                param_grid,
                cv=self.cv,
                n_jobs=self.n_jobs,
                verbose=1
            )
        elif self.tuning_method == 'random':
            search = RandomizedSearchCV(
                model_wrapper,
                param_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                n_jobs=self.n_jobs,
                verbose=1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported tuning method: {self.tuning_method}")
        
        # Fit the search
        search.fit(X_train, y_train)
        
        # Return the best parameters
        return search.best_params_
    
    def evaluate_models(self, 
                       data: Dict[str, Union[pd.DataFrame, pd.Series]],
                       models: Optional[Dict[str, FireModel]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate models on test data.
        
        Args:
            data: Dictionary containing split data
            models: Dictionary of models to evaluate (if None, use trained_models)
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        self.logger.info("Evaluating models on test data")
        
        # Use trained models if none provided
        if models is None:
            models = self.trained_models
        
        # Extract test data
        X_test = data['X_test']
        y_test = data['y_test']
        
        evaluation_results = {}
        
        # Evaluate each model
        for model_name, model in models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Evaluate the model
            test_metrics = model.evaluate(X_test, y_test)
            
            # Store metrics
            if model_name not in self.metrics:
                self.metrics[model_name] = {}
            
            self.metrics[model_name]['test'] = test_metrics
            
            # Add to results
            evaluation_results[model_name] = test_metrics
            
            self.logger.info(f"Test metrics for {model_name}: {test_metrics}")
        
        return evaluation_results
    
    def select_best_model(self, 
                         evaluation_results: Dict[str, Dict[str, Any]],
                         metric: Optional[str] = None) -> Tuple[str, FireModel]:
        """
        Select the best model based on evaluation metrics.
        
        Args:
            evaluation_results: Dictionary of evaluation metrics for each model
            metric: Metric to use for selection (if None, use selection_metric from config)
            
        Returns:
            Tuple of (best_model_name, best_model)
        """
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        if metric is None:
            metric = self.selection_metric
        
        self.logger.info(f"Selecting best model based on {metric}")
        
        # Find the model with the best metric value
        best_model_name = None
        best_metric_value = -float('inf')
        
        for model_name, metrics in evaluation_results.items():
            if metric not in metrics:
                self.logger.warning(f"Metric {metric} not found for model {model_name}")
                continue
            
            metric_value = metrics[metric]
            
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"Could not select best model based on metric {metric}")
        
        best_model = self.trained_models[best_model_name]
        
        self.logger.info(f"Selected best model: {best_model_name} with {metric} = {best_metric_value}")
        
        return best_model_name, best_model
    
    def save_model(self, 
                  model: FireModel, 
                  model_name: str,
                  to_s3: bool = False) -> str:
        """
        Save a trained model.
        
        Args:
            model: Model to save
            model_name: Name of the model
            to_s3: Whether to save to S3
            
        Returns:
            Path where the model was saved
        """
        # Create a version string
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, f"{model_name}_{version}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(model_dir, "model.pkl")
        model.save(model_path)
        
        # Save metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        model.save_metadata(metadata_path)
        
        # Save metrics if available
        if model_name in self.metrics:
            metrics_path = os.path.join(model_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics[model_name], f, indent=2)
        
        # Save to S3 if requested
        if to_s3 and self.use_aws:
            self._save_model_to_s3(model_dir, model_name, version)
        
        self.logger.info(f"Saved model {model_name} to {model_dir}")
        
        return model_dir
    
    def _save_model_to_s3(self, 
                         model_dir: str, 
                         model_name: str,
                         version: str) -> None:
        """
        Save a model to S3.
        
        Args:
            model_dir: Directory containing the model files
            model_name: Name of the model
            version: Version string
        """
        if not self.use_aws:
            raise ValueError("AWS integration is not enabled")
        
        # Create S3 prefix
        s3_prefix = f"models/{model_name}/{version}"
        
        # Upload all files in the model directory
        for root, _, files in os.walk(model_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, model_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                
                self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
        
        self.logger.info(f"Uploaded model {model_name} to S3 bucket {self.s3_bucket} with prefix {s3_prefix}")
    
    def load_model(self, 
                 model_path: str,
                 model_class: Type[FireModel],
                 from_s3: bool = False) -> FireModel:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the model
            model_class: Class of the model to load
            from_s3: Whether to load from S3
            
        Returns:
            Loaded model
        """
        self.logger.info(f"Loading model from {'S3' if from_s3 else 'local file system'}: {model_path}")
        
        if from_s3 and self.use_aws:
            # Parse S3 path
            s3_parts = model_path.replace('s3://', '').split('/')
            bucket = s3_parts[0]
            key = '/'.join(s3_parts[1:])
            
            # Download file to temporary location
            local_path = os.path.join(self.models_dir, os.path.basename(model_path))
            self.s3_client.download_file(bucket, key, local_path)
            model_path = local_path
        
        # Load metadata to get configuration
        metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Initialize model with configuration
        model = model_class(metadata['config'])
        
        # Load the model
        model.load(model_path)
        
        self.logger.info(f"Loaded model: {model.__class__.__name__}")
        
        return model
    
    def deploy_to_sagemaker(self, 
                          model: FireModel, 
                          model_name: str,
                          instance_type: str = 'ml.m5.large',
                          initial_instance_count: int = 1) -> str:
        """
        Deploy a model to SageMaker.
        
        Args:
            model: Model to deploy
            model_name: Name of the model
            instance_type: SageMaker instance type
            initial_instance_count: Number of initial instances
            
        Returns:
            SageMaker endpoint name
        """
        if not self.use_aws:
            raise ValueError("AWS integration is not enabled")
        
        self.logger.info(f"Deploying model {model_name} to SageMaker")
        
        # Create a version string
        version = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, f"{model_name}_{version}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(model_dir, "model.pkl")
        model.save(model_path)
        
        # Create a tar.gz file for SageMaker
        import tarfile
        tar_path = os.path.join(model_dir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_path, arcname=os.path.basename(model_path))
        
        # Upload to S3
        s3_key = f"sagemaker/{model_name}/{version}/model.tar.gz"
        self.s3_client.upload_file(tar_path, self.s3_bucket, s3_key)
        model_data = f"s3://{self.s3_bucket}/{s3_key}"
        
        # Create SageMaker model
        create_model_response = self.sagemaker_client.create_model(
            ModelName=f"{model_name}-{version}",
            PrimaryContainer={
                'Image': f"{boto3.client('sts').get_caller_identity()['Account']}.dkr.ecr.{self.aws_region}.amazonaws.com/sagemaker-scikit-learn:latest",
                'ModelDataUrl': model_data,
                'Environment': {
                    'SAGEMAKER_SUBMIT_DIRECTORY': model_data,
                    'SAGEMAKER_PROGRAM': 'inference.py'
                }
            },
            ExecutionRoleArn=self.sagemaker_role
        )
        
        # Create endpoint configuration
        endpoint_config_name = f"{model_name}-config-{version}"
        create_endpoint_config_response = self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': f"{model_name}-{version}",
                    'InitialInstanceCount': initial_instance_count,
                    'InstanceType': instance_type
                }
            ]
        )
        
        # Create endpoint
        endpoint_name = f"{model_name}-endpoint-{version}"
        create_endpoint_response = self.sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        self.logger.info(f"Created SageMaker endpoint: {endpoint_name}")
        
        return endpoint_name
        
    # Import the run_pipeline method from the separate file
    from .run_pipeline import run_pipeline