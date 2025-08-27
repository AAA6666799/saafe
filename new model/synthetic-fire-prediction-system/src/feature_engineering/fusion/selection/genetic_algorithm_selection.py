"""
Genetic Algorithm feature selection implementation for the synthetic fire prediction system.

This module provides an implementation of Genetic Algorithm-based feature selection,
which uses evolutionary algorithms to select an optimal subset of features.
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import random
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score

from ...base import FeatureFusion


class GeneticAlgorithmSelection(FeatureFusion):
    """
    Implementation of Genetic Algorithm-based feature selection.
    
    This class selects features using a genetic algorithm approach, which evolves
    a population of feature subsets to find an optimal solution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Genetic Algorithm feature selection component.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.best_features = []
        self.feature_names = []
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required parameters
        required_params = ['population_size', 'generations', 'crossover_prob', 'mutation_prob']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate population size
        if not isinstance(self.config['population_size'], int) or self.config['population_size'] <= 0:
            raise ValueError("'population_size' must be a positive integer")
        
        # Validate generations
        if not isinstance(self.config['generations'], int) or self.config['generations'] <= 0:
            raise ValueError("'generations' must be a positive integer")
        
        # Validate crossover probability
        if not isinstance(self.config['crossover_prob'], (int, float)) or \
           not 0 <= self.config['crossover_prob'] <= 1:
            raise ValueError("'crossover_prob' must be a number between 0 and 1")
        
        # Validate mutation probability
        if not isinstance(self.config['mutation_prob'], (int, float)) or \
           not 0 <= self.config['mutation_prob'] <= 1:
            raise ValueError("'mutation_prob' must be a number between 0 and 1")
        
        # Set default values for optional parameters
        if 'estimator_type' not in self.config:
            self.config['estimator_type'] = 'random_forest'
        
        if 'target_type' not in self.config:
            self.config['target_type'] = 'continuous'
        
        if 'target_variable' not in self.config:
            self.config['target_variable'] = 'risk_score'
        
        if 'cv_folds' not in self.config:
            self.config['cv_folds'] = 3
        
        if 'min_features' not in self.config:
            self.config['min_features'] = 1
        
        if 'max_features' not in self.config:
            self.config['max_features'] = None
    
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select features using a Genetic Algorithm approach.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the selected features
        """
        self.logger.info("Performing Genetic Algorithm feature selection")
        
        # Convert features to DataFrames
        thermal_df = self._to_dataframe(thermal_features, 'thermal')
        gas_df = self._to_dataframe(gas_features, 'gas')
        env_df = self._to_dataframe(environmental_features, 'environmental')
        
        # Combine all features into a single DataFrame
        all_features_df = pd.concat([thermal_df, gas_df, env_df], axis=1)
        
        # Create or extract target variable
        target_variable = self.config['target_variable']
        
        if target_variable in all_features_df.columns:
            # Use existing column as target
            target = all_features_df[target_variable]
            # Remove target from features
            all_features_df = all_features_df.drop(columns=[target_variable])
        else:
            # Create synthetic target variable based on available features
            target = self._create_synthetic_target(all_features_df)
        
        # Filter numeric columns
        numeric_df = all_features_df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            self.logger.warning("No numeric features available for Genetic Algorithm selection")
            return {'error': 'No numeric features available for Genetic Algorithm selection'}
        
        # Store original feature names
        self.feature_names = numeric_df.columns.tolist()
        
        # Handle missing values
        numeric_df = numeric_df.fillna(0)
        
        # Apply Genetic Algorithm feature selection
        selected_features, fitness_history = self._apply_genetic_algorithm(numeric_df, target)
        
        # Get selected feature values
        selected_feature_values = {}
        for feature in selected_features:
            if feature in all_features_df.columns:
                selected_feature_values[feature] = all_features_df[feature].iloc[0] if not all_features_df.empty else None
        
        # Create result dictionary
        result = {
            'selection_time': datetime.now().isoformat(),
            'population_size': self.config['population_size'],
            'generations': self.config['generations'],
            'crossover_prob': self.config['crossover_prob'],
            'mutation_prob': self.config['mutation_prob'],
            'estimator_type': self.config['estimator_type'],
            'target_type': self.config['target_type'],
            'target_variable': target_variable,
            'original_feature_count': len(self.feature_names),
            'selected_feature_count': len(selected_features),
            'selected_features': selected_features,
            'fitness_history': fitness_history,
            'selected_feature_values': selected_feature_values
        }
        
        self.logger.info(f"Selected {len(selected_features)} features out of {len(self.feature_names)}")
        return result
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # For feature selection, we calculate the risk score based on the selected features
        selected_feature_values = fused_features.get('selected_feature_values', {})
        
        if not selected_feature_values:
            self.logger.warning("No selected features available for risk score calculation")
            return 0.1
        
        # Calculate risk score based on selected features
        risk_indicators = []
        
        # Check for thermal indicators
        thermal_features = {k: v for k, v in selected_feature_values.items() if k.startswith('thermal_')}
        for key, value in thermal_features.items():
            if 'max_temperature' in key and isinstance(value, (int, float)):
                max_temp = value
                if max_temp > 100:  # Example threshold
                    risk_indicators.append(min(1.0, (max_temp - 100) / 100))
            
            if 'hotspot_count' in key and isinstance(value, (int, float)):
                hotspot_count = value
                if hotspot_count > 3:  # Example threshold
                    risk_indicators.append(min(1.0, hotspot_count / 10))
        
        # Check for gas indicators
        gas_features = {k: v for k, v in selected_feature_values.items() if k.startswith('gas_')}
        for key, value in gas_features.items():
            if 'concentration' in key and isinstance(value, (int, float)):
                concentration = value
                if concentration > 50:  # Example threshold
                    risk_indicators.append(min(1.0, (concentration - 50) / 100))
        
        # Check for environmental indicators
        env_features = {k: v for k, v in selected_feature_values.items() if k.startswith('environmental_')}
        for key, value in env_features.items():
            if 'temperature_rise' in key and isinstance(value, (int, float)):
                temp_rise = value
                if temp_rise > 5:  # Example threshold
                    risk_indicators.append(min(1.0, temp_rise / 20))
        
        # Calculate overall risk score
        if risk_indicators:
            # Use a weighted average of the top 3 risk indicators
            risk_indicators.sort(reverse=True)
            top_indicators = risk_indicators[:3]
            weights = [0.5, 0.3, 0.2][:len(top_indicators)]
            
            risk_score = sum(indicator * weight for indicator, weight in zip(top_indicators, weights))
        else:
            # If no risk indicators are available, use a default low risk score
            risk_score = 0.1
        
        self.logger.info(f"Calculated risk score: {risk_score}")
        return risk_score
    
    def _to_dataframe(self, features: Dict[str, Any], prefix: str) -> pd.DataFrame:
        """
        Convert features dictionary to a pandas DataFrame with prefixed column names.
        
        Args:
            features: Features dictionary
            prefix: Prefix for column names
            
        Returns:
            DataFrame with prefixed column names
        """
        # Flatten nested dictionary
        flat_dict = {}
        
        def flatten(d, parent_key=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    flatten(v, f"{parent_key}{k}_")
                elif isinstance(v, (list, tuple)) and len(v) > 0 and not isinstance(v[0], dict):
                    # For lists of simple types, use the first element
                    flat_dict[f"{parent_key}{k}"] = v[0] if v else None
                elif not isinstance(v, (list, tuple)):
                    flat_dict[f"{parent_key}{k}"] = v
        
        flatten(features)
        
        # Create DataFrame with prefixed column names
        df = pd.DataFrame({f"{prefix}_{k}": [v] for k, v in flat_dict.items()})
        
        return df
    
    def _create_synthetic_target(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Create a synthetic target variable based on available features.
        
        Args:
            features_df: DataFrame containing features
            
        Returns:
            Series containing the synthetic target variable
        """
        # This is a simplified implementation; in a real system, you would
        # use domain knowledge to create a more meaningful target variable
        
        # Check for known risk indicators
        risk_indicators = []
        
        # Check for thermal indicators
        thermal_cols = [col for col in features_df.columns if col.startswith('thermal_')]
        for col in thermal_cols:
            if 'max_temperature' in col and pd.api.types.is_numeric_dtype(features_df[col]):
                max_temp = features_df[col].iloc[0] if not features_df.empty else 0
                if max_temp > 100:  # Example threshold
                    risk_indicators.append(min(1.0, (max_temp - 100) / 100))
            
            if 'hotspot_count' in col and pd.api.types.is_numeric_dtype(features_df[col]):
                hotspot_count = features_df[col].iloc[0] if not features_df.empty else 0
                if hotspot_count > 3:  # Example threshold
                    risk_indicators.append(min(1.0, hotspot_count / 10))
        
        # Check for gas indicators
        gas_cols = [col for col in features_df.columns if col.startswith('gas_')]
        for col in gas_cols:
            if 'concentration' in col and pd.api.types.is_numeric_dtype(features_df[col]):
                concentration = features_df[col].iloc[0] if not features_df.empty else 0
                if concentration > 50:  # Example threshold
                    risk_indicators.append(min(1.0, (concentration - 50) / 100))
        
        # Check for environmental indicators
        env_cols = [col for col in features_df.columns if col.startswith('environmental_')]
        for col in env_cols:
            if 'temperature_rise' in col and pd.api.types.is_numeric_dtype(features_df[col]):
                temp_rise = features_df[col].iloc[0] if not features_df.empty else 0
                if temp_rise > 5:  # Example threshold
                    risk_indicators.append(min(1.0, temp_rise / 20))
        
        # Calculate synthetic target
        if risk_indicators:
            # Use a weighted average of the top 3 risk indicators
            risk_indicators.sort(reverse=True)
            top_indicators = risk_indicators[:3]
            weights = [0.5, 0.3, 0.2][:len(top_indicators)]
            
            synthetic_target = sum(indicator * weight for indicator, weight in zip(top_indicators, weights))
        else:
            # If no risk indicators are available, use a default low risk score
            synthetic_target = 0.1
        
        # For classification, convert to binary target
        if self.config['target_type'] == 'discrete':
            threshold = self.config.get('classification_threshold', 0.5)
            synthetic_target = 1 if synthetic_target >= threshold else 0
        
        return pd.Series([synthetic_target])
    
    def _apply_genetic_algorithm(self, features_df: pd.DataFrame, target: pd.Series) -> Tuple[List[str], List[float]]:
        """
        Apply Genetic Algorithm to select features.
        
        Args:
            features_df: DataFrame containing features
            target: Series containing the target variable
            
        Returns:
            Tuple of (selected_features, fitness_history)
        """
        # Extract feature values as a numpy array
        X = features_df.values
        y = target.values
        
        # Create estimator based on configuration
        estimator = self._create_estimator()
        
        # Define fitness function
        def fitness_function(chromosome: List[bool]) -> float:
            # If no features are selected, return a very low fitness
            if not any(chromosome):
                return -float('inf')
            
            # Get selected feature indices
            selected_indices = [i for i, selected in enumerate(chromosome) if selected]
            
            # If too few features are selected, return a low fitness
            min_features = self.config.get('min_features', 1)
            if len(selected_indices) < min_features:
                return -float('inf')
            
            # Extract selected features
            X_selected = X[:, selected_indices]
            
            try:
                # Evaluate model using cross-validation
                cv_folds = self.config.get('cv_folds', 3)
                
                if self.config['target_type'] == 'discrete':
                    # For classification, use accuracy
                    scores = cross_val_score(estimator, X_selected, y, cv=cv_folds, scoring='accuracy')
                else:
                    # For regression, use negative mean squared error
                    scores = cross_val_score(estimator, X_selected, y, cv=cv_folds, scoring='neg_mean_squared_error')
                
                # Calculate mean score
                mean_score = np.mean(scores)
                
                # Add penalty for using too many features
                feature_count_penalty = 0.001 * len(selected_indices)
                
                # Final fitness is mean score minus feature count penalty
                fitness = mean_score - feature_count_penalty
                
                return fitness
            
            except Exception as e:
                self.logger.error(f"Error evaluating fitness: {str(e)}")
                return -float('inf')
        
        # Initialize population
        population_size = self.config['population_size']
        n_features = features_df.shape[1]
        
        # Set maximum number of features if not specified
        max_features = self.config.get('max_features')
        if max_features is None or max_features > n_features:
            max_features = n_features
        
        # Initialize population with random chromosomes
        population = []
        for _ in range(population_size):
            # Randomly select between min_features and max_features
            min_features = self.config.get('min_features', 1)
            n_selected = random.randint(min_features, max_features)
            
            # Create chromosome with n_selected features set to True
            chromosome = [False] * n_features
            selected_indices = random.sample(range(n_features), n_selected)
            for idx in selected_indices:
                chromosome[idx] = True
            
            population.append(chromosome)
        
        # Run genetic algorithm
        generations = self.config['generations']
        crossover_prob = self.config['crossover_prob']
        mutation_prob = self.config['mutation_prob']
        
        fitness_history = []
        best_fitness = -float('inf')
        best_chromosome = None
        
        for generation in range(generations):
            # Evaluate fitness for each chromosome
            fitness_scores = [fitness_function(chromosome) for chromosome in population]
            
            # Track best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_chromosome = population[max_fitness_idx]
            
            # Add best fitness to history
            fitness_history.append(best_fitness)
            
            self.logger.info(f"Generation {generation + 1}/{generations}, Best Fitness: {best_fitness}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep the best chromosome
            new_population.append(population[max_fitness_idx])
            
            # Create rest of new population
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self._mutate(child1, mutation_prob)
                child2 = self._mutate(child2, mutation_prob)
                
                # Add children to new population
                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)
            
            # Update population
            population = new_population
        
        # Get selected features from best chromosome
        if best_chromosome is not None:
            selected_indices = [i for i, selected in enumerate(best_chromosome) if selected]
            selected_features = [features_df.columns[i] for i in selected_indices]
            self.best_features = selected_features
        else:
            selected_features = []
            self.best_features = []
        
        return selected_features, fitness_history
    
    def _create_estimator(self) -> Any:
        """
        Create an estimator based on configuration.
        
        Returns:
            Estimator object
        """
        estimator_type = self.config['estimator_type']
        target_type = self.config['target_type']
        
        if target_type == 'discrete':
            # Classification
            if estimator_type == 'random_forest':
                return RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                return LogisticRegression(random_state=42)
        else:
            # Regression
            if estimator_type == 'random_forest':
                return RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                return LinearRegression()
    
    def _tournament_selection(self, population: List[List[bool]], fitness_scores: List[float]) -> List[bool]:
        """
        Select a chromosome using tournament selection.
        
        Args:
            population: List of chromosomes
            fitness_scores: List of fitness scores
            
        Returns:
            Selected chromosome
        """
        # Select tournament participants
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        
        # Select the best chromosome from the tournament
        best_idx = tournament_indices[0]
        for idx in tournament_indices[1:]:
            if fitness_scores[idx] > fitness_scores[best_idx]:
                best_idx = idx
        
        return population[best_idx].copy()
    
    def _crossover(self, parent1: List[bool], parent2: List[bool]) -> Tuple[List[bool], List[bool]]:
        """
        Perform crossover between two parent chromosomes.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of (child1, child2)
        """
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def _mutate(self, chromosome: List[bool], mutation_prob: float) -> List[bool]:
        """
        Mutate a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            mutation_prob: Probability of mutation
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        
        for i in range(len(mutated)):
            if random.random() < mutation_prob:
                mutated[i] = not mutated[i]
        
        # Ensure at least one feature is selected
        if not any(mutated):
            # Randomly select one feature
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] = True
        
        return mutated
    
    def transform_new_data(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Transform new data using the selected features.
        
        Args:
            features: Features to transform
            
        Returns:
            DataFrame with selected features
        """
        if not self.best_features:
            self.logger.error("No features selected. Call fuse_features first.")
            return pd.DataFrame()
        
        # Convert features to DataFrame
        df = pd.DataFrame(features, index=[0])
        
        # Filter to include only the selected features
        common_features = [col for col in df.columns if col in self.best_features]
        selected_df = df[common_features]
        
        # Fill missing columns with zeros
        for feature in self.best_features:
            if feature not in selected_df.columns:
                selected_df[feature] = 0
        
        return selected_df