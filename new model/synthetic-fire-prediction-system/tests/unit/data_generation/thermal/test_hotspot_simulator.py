"""
Unit tests for the HotspotSimulator class.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta

from src.data_generation.thermal.hotspot_simulator import HotspotSimulator, HotspotShape, GrowthPattern


class TestHotspotSimulator(unittest.TestCase):
    """Test cases for the HotspotSimulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a basic configuration for testing
        self.config = {
            'min_temperature': 20.0,  # 20°C
            'max_temperature': 500.0,  # 500°C
            'temperature_unit': 'celsius',
            'default_shape': 'circular',
            'default_growth': 'exponential'
        }
        
        # Create a test image shape
        self.image_shape = (288, 384)
        
        # Create hotspot simulator
        self.hotspot_simulator = HotspotSimulator(self.config)
    
    def test_initialization(self):
        """Test initialization of HotspotSimulator."""
        self.assertIsNotNone(self.hotspot_simulator)
        self.assertEqual(self.hotspot_simulator.config, self.config)
        self.assertEqual(self.hotspot_simulator.min_temp, 20.0)
        self.assertEqual(self.hotspot_simulator.max_temp, 500.0)
        self.assertEqual(self.hotspot_simulator.temp_unit, 'celsius')
        self.assertEqual(self.hotspot_simulator.default_shape, HotspotShape.CIRCULAR)
        self.assertEqual(self.hotspot_simulator.default_growth, GrowthPattern.EXPONENTIAL)
    
    def test_validate_config_valid(self):
        """Test config validation with valid config."""
        # Should not raise an exception
        self.hotspot_simulator.validate_config()
    
    def test_validate_config_invalid(self):
        """Test config validation with invalid config."""
        # Invalid temperature range
        invalid_config = {
            'min_temperature': 500.0,
            'max_temperature': 20.0  # Max less than min
        }
        
        with self.assertRaises(ValueError):
            HotspotSimulator(invalid_config)
        
        # Invalid temperature unit
        invalid_config = {
            'temperature_unit': 'invalid_unit'
        }
        
        with self.assertRaises(ValueError):
            HotspotSimulator(invalid_config)
        
        # Invalid shape
        invalid_config = {
            'default_shape': 'invalid_shape'
        }
        
        with self.assertRaises(ValueError):
            HotspotSimulator(invalid_config)
        
        # Invalid growth pattern
        invalid_config = {
            'default_growth': 'invalid_growth'
        }
        
        with self.assertRaises(ValueError):
            HotspotSimulator(invalid_config)
    
    def test_generate_hotspot_default(self):
        """Test generating a hotspot with default parameters."""
        hotspot = self.hotspot_simulator.generate_hotspot(self.image_shape)
        
        # Check that the hotspot has the correct shape
        self.assertEqual(hotspot.shape, self.image_shape)
        
        # Check that the hotspot has values in the correct temperature range
        self.assertGreaterEqual(np.min(hotspot), self.config['min_temperature'])
        self.assertLessEqual(np.max(hotspot), self.config['max_temperature'])
        
        # Check that the hotspot has a peak in the center
        h, w = self.image_shape
        center_value = hotspot[h // 2, w // 2]
        edge_value = hotspot[0, 0]
        self.assertGreater(center_value, edge_value)
    
    def test_generate_hotspot_custom(self):
        """Test generating a hotspot with custom parameters."""
        # Custom parameters
        center = (100, 150)
        radius = 30
        intensity = 0.8
        shape = HotspotShape.ELLIPTICAL
        
        hotspot = self.hotspot_simulator.generate_hotspot(
            image_shape=self.image_shape,
            center=center,
            radius=radius,
            intensity=intensity,
            shape=shape
        )
        
        # Check that the hotspot has the correct shape
        self.assertEqual(hotspot.shape, self.image_shape)
        
        # Check that the hotspot has values in the correct temperature range
        self.assertGreaterEqual(np.min(hotspot), self.config['min_temperature'])
        self.assertLessEqual(np.max(hotspot), self.config['max_temperature'])
        
        # Check that the hotspot has a peak at the specified center
        center_value = hotspot[center[0], center[1]]
        edge_value = hotspot[0, 0]
        self.assertGreater(center_value, edge_value)
        
        # Check that the peak value is scaled by intensity
        max_temp_range = self.config['max_temperature'] - self.config['min_temperature']
        expected_max = self.config['min_temperature'] + intensity * max_temp_range
        self.assertAlmostEqual(np.max(hotspot), expected_max, delta=1.0)
    
    def test_generate_hotspot_shapes(self):
        """Test generating hotspots with different shapes."""
        for shape in HotspotShape:
            hotspot = self.hotspot_simulator.generate_hotspot(
                image_shape=self.image_shape,
                shape=shape
            )
            
            # Check that the hotspot has the correct shape
            self.assertEqual(hotspot.shape, self.image_shape)
            
            # Check that the hotspot has values in the correct temperature range
            self.assertGreaterEqual(np.min(hotspot), self.config['min_temperature'])
            self.assertLessEqual(np.max(hotspot), self.config['max_temperature'])
    
    def test_calculate_growth_factor(self):
        """Test calculating growth factors with different patterns."""
        # Test linear growth
        linear_factor = self.hotspot_simulator.calculate_growth_factor(
            time_elapsed=50,
            total_duration=100,
            growth_pattern=GrowthPattern.LINEAR
        )
        self.assertAlmostEqual(linear_factor, 0.5)
        
        # Test exponential growth
        exp_factor = self.hotspot_simulator.calculate_growth_factor(
            time_elapsed=50,
            total_duration=100,
            growth_pattern=GrowthPattern.EXPONENTIAL
        )
        self.assertAlmostEqual(exp_factor, 0.25)
        
        # Test logarithmic growth
        log_factor = self.hotspot_simulator.calculate_growth_factor(
            time_elapsed=50,
            total_duration=100,
            growth_pattern=GrowthPattern.LOGARITHMIC
        )
        self.assertGreater(log_factor, 0.5)  # Logarithmic grows faster initially
        
        # Test sigmoid growth
        sigmoid_factor = self.hotspot_simulator.calculate_growth_factor(
            time_elapsed=50,
            total_duration=100,
            growth_pattern=GrowthPattern.SIGMOID
        )
        self.assertAlmostEqual(sigmoid_factor, 0.5, delta=0.1)
        
        # Test time bounds
        zero_factor = self.hotspot_simulator.calculate_growth_factor(
            time_elapsed=0,
            total_duration=100,
            growth_pattern=GrowthPattern.LINEAR
        )
        self.assertAlmostEqual(zero_factor, 0.0)
        
        full_factor = self.hotspot_simulator.calculate_growth_factor(
            time_elapsed=100,
            total_duration=100,
            growth_pattern=GrowthPattern.LINEAR
        )
        self.assertAlmostEqual(full_factor, 1.0)
        
        # Test exceeding bounds
        exceed_factor = self.hotspot_simulator.calculate_growth_factor(
            time_elapsed=150,
            total_duration=100,
            growth_pattern=GrowthPattern.LINEAR
        )
        self.assertAlmostEqual(exceed_factor, 1.0)  # Should clamp to 1.0
    
    def test_generate_evolving_hotspot(self):
        """Test generating an evolving hotspot."""
        # Create parameters
        start_time = datetime.now()
        current_time = start_time + timedelta(seconds=50)
        duration = timedelta(seconds=100)
        
        params = {
            'center': (100, 150),
            'max_radius': 30,
            'max_intensity': 0.8,
            'shape': HotspotShape.CIRCULAR.value,
            'growth_pattern': GrowthPattern.LINEAR.value
        }
        
        hotspot = self.hotspot_simulator.generate_evolving_hotspot(
            image_shape=self.image_shape,
            start_time=start_time,
            current_time=current_time,
            duration=duration,
            params=params
        )
        
        # Check that the hotspot has the correct shape
        self.assertEqual(hotspot.shape, self.image_shape)
        
        # Check that the hotspot has values in the correct temperature range
        self.assertGreaterEqual(np.min(hotspot), self.config['min_temperature'])
        self.assertLessEqual(np.max(hotspot), self.config['max_temperature'])
        
        # Check that the hotspot has evolved to approximately half its maximum
        # (since we're at 50/100 seconds with linear growth)
        center_value = hotspot[params['center'][0], params['center'][1]]
        max_temp_range = self.config['max_temperature'] - self.config['min_temperature']
        expected_max = self.config['min_temperature'] + 0.5 * params['max_intensity'] * max_temp_range
        self.assertAlmostEqual(center_value, expected_max, delta=50.0)
    
    def test_generate_multiple_hotspots(self):
        """Test generating multiple hotspots."""
        # Create hotspot parameters
        hotspots = [
            {
                'center': (100, 150),
                'radius': 30,
                'intensity': 0.8,
                'shape': HotspotShape.CIRCULAR.value
            },
            {
                'center': (200, 250),
                'radius': 20,
                'intensity': 0.6,
                'shape': HotspotShape.ELLIPTICAL.value
            }
        ]
        
        result = self.hotspot_simulator.generate_multiple_hotspots(
            image_shape=self.image_shape,
            hotspots=hotspots
        )
        
        # Check that the result has the correct shape
        self.assertEqual(result.shape, self.image_shape)
        
        # Check that the result has values in the correct temperature range
        self.assertGreaterEqual(np.min(result), self.config['min_temperature'])
        self.assertLessEqual(np.max(result), self.config['max_temperature'])
        
        # Check that both hotspots are present (peaks at both centers)
        center1_value = result[hotspots[0]['center'][0], hotspots[0]['center'][1]]
        center2_value = result[hotspots[1]['center'][0], hotspots[1]['center'][1]]
        
        self.assertGreater(center1_value, self.config['min_temperature'] + 10)
        self.assertGreater(center2_value, self.config['min_temperature'] + 10)
    
    def test_generate_evolving_multiple_hotspots(self):
        """Test generating multiple evolving hotspots."""
        # Create parameters
        start_time = datetime.now()
        current_time = start_time + timedelta(seconds=50)
        
        hotspots = [
            {
                'center': (100, 150),
                'max_radius': 30,
                'max_intensity': 0.8,
                'shape': HotspotShape.CIRCULAR.value,
                'growth_pattern': GrowthPattern.LINEAR.value,
                'start_offset': 0,
                'duration': 100
            },
            {
                'center': (200, 250),
                'max_radius': 20,
                'max_intensity': 0.6,
                'shape': HotspotShape.ELLIPTICAL.value,
                'growth_pattern': GrowthPattern.EXPONENTIAL.value,
                'start_offset': 30,  # Starts 30 seconds after the first hotspot
                'duration': 70
            }
        ]
        
        result = self.hotspot_simulator.generate_evolving_multiple_hotspots(
            image_shape=self.image_shape,
            start_time=start_time,
            current_time=current_time,
            hotspots=hotspots
        )
        
        # Check that the result has the correct shape
        self.assertEqual(result.shape, self.image_shape)
        
        # Check that the result has values in the correct temperature range
        self.assertGreaterEqual(np.min(result), self.config['min_temperature'])
        self.assertLessEqual(np.max(result), self.config['max_temperature'])
        
        # Check that both hotspots are present
        # First hotspot should be at 50% growth (50/100 seconds with linear growth)
        # Second hotspot should be at 20% growth ((50-30)/70 seconds with exponential growth)
        center1_value = result[hotspots[0]['center'][0], hotspots[0]['center'][1]]
        center2_value = result[hotspots[1]['center'][0], hotspots[1]['center'][1]]
        
        self.assertGreater(center1_value, self.config['min_temperature'] + 10)
        self.assertGreater(center2_value, self.config['min_temperature'] + 10)
        
        # The first hotspot should be hotter than the second (50% vs ~4% growth)
        self.assertGreater(center1_value, center2_value)


if __name__ == '__main__':
    unittest.main()