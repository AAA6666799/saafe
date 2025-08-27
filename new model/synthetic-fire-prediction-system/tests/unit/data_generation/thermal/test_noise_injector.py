"""
Unit tests for the NoiseInjector class.
"""

import unittest
import numpy as np
from datetime import datetime

from src.data_generation.thermal.noise_injector import NoiseInjector, NoiseType


class TestNoiseInjector(unittest.TestCase):
    """Test cases for the NoiseInjector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a basic configuration for testing
        self.config = {
            'noise_types': ['gaussian', 'salt_and_pepper'],
            'noise_params': {
                'gaussian': {
                    'mean': 0,
                    'std': 2.0
                },
                'salt_and_pepper': {
                    'amount': 0.01,
                    'salt_vs_pepper': 0.5
                }
            },
            'camera_model': 'flir_lepton',
            'resolution': (288, 384)
        }
        
        # Create a test image (constant temperature)
        self.test_image = np.ones((288, 384)) * 25.0  # 25°C
        
        # Create noise injector
        self.noise_injector = NoiseInjector(self.config)
    
    def test_initialization(self):
        """Test initialization of NoiseInjector."""
        self.assertIsNotNone(self.noise_injector)
        self.assertEqual(self.noise_injector.config, self.config)
        self.assertIsNotNone(self.noise_injector.fixed_pattern)
    
    def test_validate_config_valid(self):
        """Test config validation with valid config."""
        # Should not raise an exception
        self.noise_injector.validate_config()
    
    def test_validate_config_invalid(self):
        """Test config validation with invalid config."""
        # Missing noise_params
        invalid_config = {
            'noise_types': ['gaussian', 'salt_and_pepper']
        }
        
        with self.assertRaises(ValueError):
            NoiseInjector(invalid_config)
        
        # Invalid noise type
        invalid_config = {
            'noise_types': ['invalid_noise_type'],
            'noise_params': {}
        }
        
        with self.assertRaises(ValueError):
            NoiseInjector(invalid_config)
        
        # Missing parameters for noise type
        invalid_config = {
            'noise_types': ['gaussian'],
            'noise_params': {
                'salt_and_pepper': {}  # Missing gaussian params
            }
        }
        
        with self.assertRaises(ValueError):
            NoiseInjector(invalid_config)
    
    def test_apply_noise(self):
        """Test applying noise to an image."""
        noisy_image = self.noise_injector.apply_noise(self.test_image)
        
        # Check that the image has changed
        self.assertFalse(np.array_equal(noisy_image, self.test_image))
        
        # Check that the image has the same shape
        self.assertEqual(noisy_image.shape, self.test_image.shape)
        
        # Check that the image values are still in a reasonable range
        # (within 10 degrees of original)
        self.assertTrue(np.all(noisy_image >= self.test_image.min() - 10))
        self.assertTrue(np.all(noisy_image <= self.test_image.max() + 10))
    
    def test_gaussian_noise(self):
        """Test applying Gaussian noise."""
        # Create a noise injector with only Gaussian noise
        config = {
            'noise_types': ['gaussian'],
            'noise_params': {
                'gaussian': {
                    'mean': 0,
                    'std': 2.0
                }
            }
        }
        
        noise_injector = NoiseInjector(config)
        noisy_image = noise_injector.apply_noise(self.test_image)
        
        # Check that the image has changed
        self.assertFalse(np.array_equal(noisy_image, self.test_image))
        
        # Check that the mean is approximately preserved (within 1 degree)
        self.assertAlmostEqual(np.mean(noisy_image), np.mean(self.test_image), delta=1.0)
        
        # Check that the standard deviation has increased
        self.assertGreater(np.std(noisy_image), np.std(self.test_image))
    
    def test_salt_and_pepper_noise(self):
        """Test applying salt and pepper noise."""
        # Create a noise injector with only salt and pepper noise
        config = {
            'noise_types': ['salt_and_pepper'],
            'noise_params': {
                'salt_and_pepper': {
                    'amount': 0.1,  # 10% of pixels affected
                    'salt_vs_pepper': 0.5
                }
            }
        }
        
        noise_injector = NoiseInjector(config)
        noisy_image = noise_injector.apply_noise(self.test_image)
        
        # Check that the image has changed
        self.assertFalse(np.array_equal(noisy_image, self.test_image))
        
        # Check that there are both minimum and maximum values in the image
        # (salt and pepper)
        self.assertLess(np.min(noisy_image), np.min(self.test_image))
        self.assertGreater(np.max(noisy_image), np.max(self.test_image))
        
        # Check that most pixels are unchanged (within a small delta)
        unchanged_ratio = np.sum(np.abs(noisy_image - self.test_image) < 0.1) / self.test_image.size
        self.assertGreater(unchanged_ratio, 0.8)  # At least 80% unchanged
    
    def test_camera_characteristics(self):
        """Test simulating camera characteristics."""
        processed_image = self.noise_injector.simulate_camera_characteristics(self.test_image)
        
        # Check that the image has changed
        self.assertFalse(np.array_equal(processed_image, self.test_image))
        
        # Check that the image has the same shape
        self.assertEqual(processed_image.shape, self.test_image.shape)
        
        # Check that vignetting has been applied (edges are cooler than center)
        h, w = self.test_image.shape
        center_region = processed_image[h//4:3*h//4, w//4:3*w//4]
        edge_region = processed_image[0:h//8, 0:w//8]  # Top-left corner
        
        self.assertGreater(np.mean(center_region), np.mean(edge_region))
    
    def test_multiple_noise_types(self):
        """Test applying multiple noise types."""
        # Create a noise injector with multiple noise types
        config = {
            'noise_types': ['gaussian', 'salt_and_pepper', 'fixed_pattern'],
            'noise_params': {
                'gaussian': {
                    'mean': 0,
                    'std': 1.0
                },
                'salt_and_pepper': {
                    'amount': 0.01,
                    'salt_vs_pepper': 0.5
                },
                'fixed_pattern': {
                    'strength': 0.5
                }
            }
        }
        
        noise_injector = NoiseInjector(config)
        noisy_image = noise_injector.apply_noise(self.test_image)
        
        # Check that the image has changed
        self.assertFalse(np.array_equal(noisy_image, self.test_image))
        
        # Check that the image has the same shape
        self.assertEqual(noisy_image.shape, self.test_image.shape)
        
        # Check that the standard deviation has increased significantly
        # (due to multiple noise sources)
        self.assertGreater(np.std(noisy_image), 2 * np.std(self.test_image))


if __name__ == '__main__':
    unittest.main()