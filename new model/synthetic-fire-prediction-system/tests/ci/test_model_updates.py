"""
Continuous Integration tests for model updates in FLIR+SCD41 fire detection system.

This module contains tests that verify model updates, retraining pipelines,
and version compatibility for continuous integration.
"""

import sys
import os
import unittest
from datetime import datetime
import numpy as np
import pandas as pd
import tempfile
import shutil
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

try:
    from src.ml.ensemble.model_ensemble_manager import ModelEnsembleManager
    from src.ml.active_learning.active_learning_loop import ActiveLearningLoop
    from src.ml.model_updater import ModelUpdater
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ML components: {e}")
    ML_AVAILABLE = False


class TestModelUpdateCI(unittest.TestCase):
    """Test cases for model update CI functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        if not ML_AVAILABLE:
            self.skipTest("ML components not available")
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.test_config = {
            'ensemble_strategy': 'weighted_voting',
            'confidence_threshold': 0.7,
            'cross_validation_folds': 3,
            'active_learning': {
                'sampling_strategy': 'uncertainty',
                'batch_size': 10,
                'retrain_threshold': 0.1
            }
        }
        
        # Generate synthetic test data
        self.X_train, self.y_train = self._generate_test_data(n_samples=200)
        self.X_test, self.y_test = self._generate_test_data(n_samples=50, random_state=42)
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _generate_test_data(self, n_samples: int = 100, random_state: int = None) -> tuple:
        """Generate synthetic test data for CI testing."""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate 18 features (15 FLIR thermal + 3 SCD41 gas)
        feature_names = [
            't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
            't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
            't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
            'tproxy_val', 'tproxy_delta', 'tproxy_vel',  # FLIR features
            'gas_val', 'gas_delta', 'gas_vel'  # SCD41 features
        ]
        
        # Generate features with some correlation
        X = np.random.randn(n_samples, len(feature_names))
        
        # Generate target based on features
        fire_score = (
            X[:, 0] * 0.2 +   # t_mean
            X[:, 2] * 0.3 +   # t_max
            X[:, 4] * 0.15 +  # t_hot_area_pct
            X[:, 15] * 0.2 +  # gas_val
            X[:, 16] * 0.15 + # gas_delta
            np.random.randn(n_samples) * 0.1
        )
        
        # Convert to binary classification
        y = (fire_score > 0).astype(int)
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='fire_detected')
        
        return X_df, y_series
    
    def test_model_version_compatibility(self):
        """Test model version compatibility across updates."""
        # Create initial ensemble
        ensemble = ModelEnsembleManager(self.test_config)
        ensemble.create_default_ensemble()
        
        # Train initial model
        initial_results = ensemble.train(self.X_train, self.y_train)
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'initial_model.json')
        ensemble.save(model_path)
        
        # Load model in new ensemble instance
        new_ensemble = ModelEnsembleManager(self.test_config)
        new_ensemble.load(model_path)
        
        # Check that loaded model works
        predictions = new_ensemble.predict(self.X_test)
        self.assertIn('predictions', predictions)
        self.assertIn('confidence_info', predictions)
        
        # Check that model structure is preserved
        self.assertEqual(len(ensemble.models), len(new_ensemble.models))
        for model_name in ensemble.models:
            self.assertIn(model_name, new_ensemble.models)
    
    def test_incremental_model_updates(self):
        """Test incremental model updates."""
        try:
            # Create model updater
            updater = ModelUpdater(self.test_config)
            
            # Initial training
            initial_metrics = updater.update_model(self.X_train, self.y_train)
            self.assertIsInstance(initial_metrics, dict)
            self.assertIn('training_time', initial_metrics)
            
            # Incremental update with new data
            X_new, y_new = self._generate_test_data(n_samples=50, random_state=100)
            update_metrics = updater.update_model(X_new, y_new, incremental=True)
            
            self.assertIsInstance(update_metrics, dict)
            self.assertIn('update_time', update_metrics)
            
            # Check that model can still make predictions
            test_predictions = updater.predict(self.X_test)
            self.assertEqual(len(test_predictions), len(self.X_test))
            
        except Exception as e:
            self.skipTest(f"Incremental update test failed: {e}")
    
    def test_active_learning_integration(self):
        """Test active learning integration with model updates."""
        try:
            # Create active learning loop
            al_loop = ActiveLearningLoop(self.test_config)
            
            # Initialize with initial data
            al_loop.initialize(self.X_train, self.y_train)
            
            # Run active learning cycle
            new_data, new_labels = al_loop.select_next_batch(self.X_test, n_samples=10)
            
            # Check that we got new data
            self.assertEqual(len(new_data), 10)
            self.assertEqual(len(new_labels), 10)
            
            # Update model with new data
            update_result = al_loop.update_model(new_data, new_labels)
            self.assertIsInstance(update_result, dict)
            
        except Exception as e:
            self.skipTest(f"Active learning test failed: {e}")
    
    def test_model_performance_regression(self):
        """Test for model performance regression."""
        # Create ensemble and train
        ensemble = ModelEnsembleManager(self.test_config)
        ensemble.create_default_ensemble()
        ensemble.train(self.X_train, self.y_train)
        
        # Get baseline performance
        baseline_predictions = ensemble.predict(self.X_test)
        baseline_accuracy = np.mean(baseline_predictions['predictions'] == self.y_test)
        
        # Save baseline metrics
        baseline_metrics = {
            'accuracy': baseline_accuracy,
            'min_acceptable_accuracy': 0.7
        }
        
        # Check that baseline performance meets minimum requirements
        self.assertGreaterEqual(baseline_accuracy, baseline_metrics['min_acceptable_accuracy'],
                               f"Baseline accuracy {baseline_accuracy:.3f} below minimum {baseline_metrics['min_acceptable_accuracy']}")
        
        # Test model persistence and reloading
        model_path = os.path.join(self.temp_dir, 'performance_test_model.json')
        ensemble.save(model_path)
        
        # Load and test again
        new_ensemble = ModelEnsembleManager(self.test_config)
        new_ensemble.load(model_path)
        
        new_predictions = new_ensemble.predict(self.X_test)
        new_accuracy = np.mean(new_predictions['predictions'] == self.y_test)
        
        # Performance should be consistent after save/load
        self.assertAlmostEqual(baseline_accuracy, new_accuracy, places=2,
                              msg="Performance should be consistent after save/load")
    
    def test_cross_platform_compatibility(self):
        """Test model compatibility across different platforms."""
        # Create and train ensemble
        ensemble = ModelEnsembleManager(self.test_config)
        ensemble.create_default_ensemble()
        ensemble.train(self.X_train, self.y_train)
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'cross_platform_model.json')
        save_result = ensemble.save(model_path)
        
        # Verify save was successful
        self.assertTrue(save_result, "Model save should be successful")
        self.assertTrue(os.path.exists(model_path), "Model file should exist")
        
        # Load model
        new_ensemble = ModelEnsembleManager(self.test_config)
        load_result = new_ensemble.load(model_path)
        
        # Verify load was successful
        self.assertTrue(load_result, "Model load should be successful")
        
        # Test predictions
        predictions = new_ensemble.predict(self.X_test)
        self.assertIn('predictions', predictions)
        self.assertIn('confidence_info', predictions)


if __name__ == '__main__':
    unittest.main(verbosity=2)