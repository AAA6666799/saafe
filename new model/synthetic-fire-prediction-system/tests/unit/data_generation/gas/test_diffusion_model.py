"""
Unit tests for the DiffusionModel class.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta

from src.data_generation.gas.diffusion_model import DiffusionModel, DiffusionType


class TestDiffusionModel(unittest.TestCase):
    """Test cases for the DiffusionModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a basic configuration for testing
        self.config = {
            'diffusion_type': 'gaussian',
            'grid_resolution': (10, 10, 5),
            'time_step': 1.0,
            'boundary_conditions': 'reflective',
            'environmental_factors': {
                'temperature': 25.0,
                'pressure': 101.3,
                'humidity': 50.0,
                'air_flow': [0.1, 0.0, 0.0]
            }
        }
        
        # Create the diffusion model
        self.model = DiffusionModel(self.config)
    
    def test_initialization(self):
        """Test initialization of the diffusion model."""
        self.assertEqual(self.model.diffusion_type, DiffusionType.GAUSSIAN)
        self.assertEqual(self.model.grid_resolution, (10, 10, 5))
        self.assertEqual(self.model.time_step, 1.0)
        self.assertEqual(self.model.boundary_conditions, 'reflective')
        self.assertEqual(self.model.env_factors['temperature'], 25.0)
        self.assertEqual(self.model.env_factors['pressure'], 101.3)
        self.assertEqual(self.model.env_factors['humidity'], 50.0)
        self.assertEqual(self.model.env_factors['air_flow'], [0.1, 0.0, 0.0])
        self.assertIsNotNone(self.model.grid)
        self.assertEqual(self.model.grid.shape, (10, 10, 5))
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Test with valid config
        self.model.validate_config()
        
        # Test with invalid diffusion type
        config_invalid_type = self.config.copy()
        config_invalid_type['diffusion_type'] = 'invalid_type'
        with self.assertRaises(ValueError):
            DiffusionModel(config_invalid_type).validate_config()
        
        # Test with invalid grid resolution
        config_invalid_grid = self.config.copy()
        config_invalid_grid['grid_resolution'] = (10, 10)  # Missing z dimension
        with self.assertRaises(ValueError):
            DiffusionModel(config_invalid_grid).validate_config()
        
        # Test with negative time step
        config_negative_time = self.config.copy()
        config_negative_time['time_step'] = -1.0
        with self.assertRaises(ValueError):
            DiffusionModel(config_negative_time).validate_config()
        
        # Test with invalid boundary conditions
        config_invalid_bc = self.config.copy()
        config_invalid_bc['boundary_conditions'] = 'invalid_bc'
        with self.assertRaises(ValueError):
            DiffusionModel(config_invalid_bc).validate_config()
    
    def test_initialize_grid(self):
        """Test grid initialization."""
        # Test default grid
        self.model.initialize_grid()
        self.assertEqual(self.model.grid.shape, (10, 10, 5))
        self.assertTrue(np.all(self.model.grid == 0.0))
        
        # Test with different resolution
        self.model.grid_resolution = (5, 5, 5)
        self.model.initialize_grid()
        self.assertEqual(self.model.grid.shape, (5, 5, 5))
        
        # Test with advection-diffusion type
        self.model.diffusion_type = DiffusionType.ADVECTION_DIFFUSION
        self.model.initialize_grid()
        self.assertEqual(self.model.grid.shape, (5, 5, 5))
        self.assertTrue(hasattr(self.model, 'velocity_field'))
        self.assertEqual(self.model.velocity_field.shape, (5, 5, 5, 3))
    
    def test_adjust_diffusion_coefficient(self):
        """Test adjustment of diffusion coefficient based on temperature and pressure."""
        # Test with methane at standard conditions
        d_methane = self.model.adjust_diffusion_coefficient('methane', 25.0, 101.3)
        self.assertAlmostEqual(d_methane, 0.196, places=3)
        
        # Test with methane at higher temperature
        d_methane_hot = self.model.adjust_diffusion_coefficient('methane', 50.0, 101.3)
        self.assertGreater(d_methane_hot, d_methane)  # Should increase with temperature
        
        # Test with methane at higher pressure
        d_methane_high_p = self.model.adjust_diffusion_coefficient('methane', 25.0, 200.0)
        self.assertLess(d_methane_high_p, d_methane)  # Should decrease with pressure
        
        # Test with unknown gas type
        with self.assertRaises(ValueError):
            self.model.adjust_diffusion_coefficient('unknown_gas', 25.0, 101.3)
    
    def test_simulate_diffusion(self):
        """Test diffusion simulation."""
        # Create a simple source point
        source_points = [
            {
                'position': (5, 5, 2),
                'strength': 10.0,
                'start_time': 0,
                'duration': 10
            }
        ]
        
        # Simulate diffusion
        result = self.model.simulate_diffusion('methane', source_points, 10.0)
        
        # Check result
        self.assertEqual(result.shape, (10, 10, 5))
        self.assertGreater(np.max(result), 0.0)  # Should have some concentration
        self.assertGreaterEqual(np.min(result), 0.0)  # Should not have negative concentrations
        
        # Check that concentration is highest at source point
        source_x, source_y, source_z = source_points[0]['position']
        source_conc = result[source_x, source_y, source_z]
        self.assertGreaterEqual(source_conc, result[0, 0, 0])
    
    def test_simulate_gaussian_diffusion(self):
        """Test Gaussian diffusion simulation."""
        # Create a simple source point
        source_points = [
            {
                'position': (5, 5, 2),
                'strength': 10.0,
                'start_time': 0,
                'duration': 10
            }
        ]
        
        # Simulate Gaussian diffusion
        result = self.model._simulate_gaussian_diffusion('methane', source_points, 10.0)
        
        # Check result
        self.assertEqual(result.shape, (10, 10, 5))
        self.assertGreater(np.max(result), 0.0)
    
    def test_apply_boundary_conditions(self):
        """Test application of boundary conditions."""
        # Create a test grid
        test_grid = np.ones((10, 10, 5))
        
        # Test reflective boundary conditions
        self.model.boundary_conditions = 'reflective'
        result = self.model._apply_boundary_conditions(test_grid)
        self.assertTrue(np.all(result == 1.0))
        
        # Test absorbing boundary conditions
        self.model.boundary_conditions = 'absorbing'
        result = self.model._apply_boundary_conditions(test_grid)
        self.assertEqual(result[0, 0, 0], 0.0)  # Boundary should be zero
        self.assertEqual(result[5, 5, 2], 1.0)  # Interior should be unchanged
        
        # Test periodic boundary conditions
        self.model.boundary_conditions = 'periodic'
        result = self.model._apply_boundary_conditions(test_grid)
        self.assertEqual(result[0, 0, 0], result[9, 9, 4])  # Opposite corners should be equal
    
    def test_get_concentration_at_point(self):
        """Test getting concentration at a specific point."""
        # Set up a test grid
        self.model.grid = np.zeros((10, 10, 5))
        self.model.grid[5, 5, 2] = 10.0
        
        # Test valid point
        conc = self.model.get_concentration_at_point((5, 5, 2))
        self.assertEqual(conc, 10.0)
        
        # Test another valid point
        conc = self.model.get_concentration_at_point((0, 0, 0))
        self.assertEqual(conc, 0.0)
        
        # Test out-of-bounds point
        conc = self.model.get_concentration_at_point((20, 20, 20))
        self.assertEqual(conc, 0.0)


if __name__ == '__main__':
    unittest.main()