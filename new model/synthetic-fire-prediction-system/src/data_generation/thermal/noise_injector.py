"""
Noise injection for thermal images.

This module provides functionality for adding realistic sensor noise and
environmental interference to synthetic thermal images.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from enum import Enum


class NoiseType(Enum):
    """Enumeration of supported noise types."""
    GAUSSIAN = "gaussian"
    SALT_AND_PEPPER = "salt_and_pepper"
    POISSON = "poisson"
    SPECKLE = "speckle"
    PERIODIC = "periodic"
    FIXED_PATTERN = "fixed_pattern"
    ROW_COLUMN = "row_column"


class NoiseInjector:
    """
    Class for adding realistic sensor noise and environmental interference to thermal images.
    
    This class provides methods for simulating various types of noise that are commonly
    found in thermal imaging systems, including sensor noise, environmental interference,
    and camera-specific artifacts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the noise injector with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters for noise injection
                   - noise_types: List of noise types to apply
                   - noise_params: Dictionary of parameters for each noise type
                   - camera_model: Optional camera model to simulate specific characteristics
        """
        self.config = config
        self.validate_config()
        
        # Initialize camera-specific noise patterns if a camera model is specified
        self.fixed_pattern = None
        if 'camera_model' in self.config:
            self.fixed_pattern = self._generate_camera_fixed_pattern()
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if 'noise_types' not in self.config:
            raise ValueError("Missing required configuration parameter: noise_types")
        
        if 'noise_params' not in self.config:
            raise ValueError("Missing required configuration parameter: noise_params")
        
        # Validate that all specified noise types have corresponding parameters
        for noise_type in self.config['noise_types']:
            try:
                noise_enum = NoiseType(noise_type)
                if noise_enum.value not in self.config['noise_params']:
                    raise ValueError(f"Missing parameters for noise type: {noise_type}")
            except ValueError:
                raise ValueError(f"Unsupported noise type: {noise_type}")
    
    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply configured noise types to the input thermal image.
        
        Args:
            image: Input thermal image as a 2D numpy array
            
        Returns:
            Thermal image with noise applied
        """
        noisy_image = image.copy().astype(np.float32)
        
        for noise_type_str in self.config['noise_types']:
            noise_type = NoiseType(noise_type_str)
            params = self.config['noise_params'][noise_type.value]
            
            if noise_type == NoiseType.GAUSSIAN:
                noisy_image = self._apply_gaussian_noise(noisy_image, params)
            elif noise_type == NoiseType.SALT_AND_PEPPER:
                noisy_image = self._apply_salt_and_pepper_noise(noisy_image, params)
            elif noise_type == NoiseType.POISSON:
                noisy_image = self._apply_poisson_noise(noisy_image, params)
            elif noise_type == NoiseType.SPECKLE:
                noisy_image = self._apply_speckle_noise(noisy_image, params)
            elif noise_type == NoiseType.PERIODIC:
                noisy_image = self._apply_periodic_noise(noisy_image, params)
            elif noise_type == NoiseType.FIXED_PATTERN:
                noisy_image = self._apply_fixed_pattern_noise(noisy_image, params)
            elif noise_type == NoiseType.ROW_COLUMN:
                noisy_image = self._apply_row_column_noise(noisy_image, params)
        
        # Ensure values stay within valid range
        min_val = np.min(image)
        max_val = np.max(image)
        noisy_image = np.clip(noisy_image, min_val, max_val)
        
        return noisy_image
    
    def _apply_gaussian_noise(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply Gaussian noise to the image.
        
        Args:
            image: Input image
            params: Parameters for Gaussian noise
                   - mean: Mean of the Gaussian distribution (default: 0)
                   - std: Standard deviation of the Gaussian distribution
            
        Returns:
            Image with Gaussian noise applied
        """
        mean = params.get('mean', 0)
        std = params.get('std', 1.0)
        
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + noise
        
        return noisy_image
    
    def _apply_salt_and_pepper_noise(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply salt and pepper noise to the image.
        
        Args:
            image: Input image
            params: Parameters for salt and pepper noise
                   - amount: Proportion of the image to be affected by noise
                   - salt_vs_pepper: Ratio of salt (white) to pepper (black) noise
            
        Returns:
            Image with salt and pepper noise applied
        """
        amount = params.get('amount', 0.05)
        salt_vs_pepper = params.get('salt_vs_pepper', 0.5)
        
        noisy_image = image.copy()
        
        # Salt (white) noise
        num_salt = int(np.ceil(amount * salt_vs_pepper * image.size))
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy_image[tuple(coords)] = np.max(image)
        
        # Pepper (black) noise
        num_pepper = int(np.ceil(amount * (1 - salt_vs_pepper) * image.size))
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy_image[tuple(coords)] = np.min(image)
        
        return noisy_image
    
    def _apply_poisson_noise(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply Poisson noise to the image.
        
        Args:
            image: Input image
            params: Parameters for Poisson noise
                   - scale: Scaling factor for the noise intensity
            
        Returns:
            Image with Poisson noise applied
        """
        scale = params.get('scale', 1.0)
        
        # Scale image for Poisson noise generation
        scaled_image = image / scale
        
        # Ensure all values are positive
        min_val = np.min(scaled_image)
        if min_val < 0:
            scaled_image = scaled_image - min_val
        
        # Generate Poisson noise
        noisy_image = np.random.poisson(scaled_image).astype(np.float32)
        
        # Scale back
        noisy_image = noisy_image * scale
        
        # Adjust range to match original image
        noisy_image = noisy_image - np.min(noisy_image) + np.min(image)
        noisy_image = noisy_image / np.max(noisy_image) * np.max(image)
        
        return noisy_image
    
    def _apply_speckle_noise(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply speckle noise to the image.
        
        Args:
            image: Input image
            params: Parameters for speckle noise
                   - std: Standard deviation of the noise
            
        Returns:
            Image with speckle noise applied
        """
        std = params.get('std', 0.1)
        
        noise = np.random.normal(0, std, image.shape)
        noisy_image = image + image * noise
        
        return noisy_image
    
    def _apply_periodic_noise(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply periodic noise to the image (simulates electrical interference).
        
        Args:
            image: Input image
            params: Parameters for periodic noise
                   - frequency_x: Frequency in x direction
                   - frequency_y: Frequency in y direction
                   - amplitude: Amplitude of the periodic noise
            
        Returns:
            Image with periodic noise applied
        """
        frequency_x = params.get('frequency_x', 0.1)
        frequency_y = params.get('frequency_y', 0.1)
        amplitude = params.get('amplitude', 5.0)
        
        h, w = image.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        noise = amplitude * np.sin(2 * np.pi * frequency_x * x / w + 2 * np.pi * frequency_y * y / h)
        noisy_image = image + noise
        
        return noisy_image
    
    def _generate_camera_fixed_pattern(self) -> np.ndarray:
        """
        Generate a fixed pattern noise specific to the configured camera model.
        
        Returns:
            Fixed pattern noise as a 2D numpy array
        """
        camera_model = self.config.get('camera_model', 'generic')
        resolution = self.config.get('resolution', (288, 384))  # Default thermal resolution
        
        # Different camera models have different fixed pattern characteristics
        if camera_model == 'flir_lepton':
            # FLIR Lepton has a specific fixed pattern noise
            pattern = np.random.normal(0, 0.5, resolution)
            # Add some structured components
            h, w = resolution
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            pattern += 0.2 * np.sin(x / 30)
            pattern += 0.1 * np.cos(y / 20)
        elif camera_model == 'seek_thermal':
            # Seek Thermal has a different pattern
            pattern = np.random.normal(0, 0.3, resolution)
            # Add grid-like pattern
            pattern[::4, :] += 0.4
            pattern[:, ::4] += 0.4
        else:
            # Generic thermal camera
            pattern = np.random.normal(0, 0.2, resolution)
        
        return pattern
    
    def _apply_fixed_pattern_noise(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply fixed pattern noise to the image.
        
        Args:
            image: Input image
            params: Parameters for fixed pattern noise
                   - strength: Strength of the fixed pattern noise
            
        Returns:
            Image with fixed pattern noise applied
        """
        strength = params.get('strength', 1.0)
        
        if self.fixed_pattern is None or self.fixed_pattern.shape != image.shape:
            # Generate a new fixed pattern if needed
            h, w = image.shape
            self.fixed_pattern = np.random.normal(0, 0.2, (h, w))
        
        noisy_image = image + strength * self.fixed_pattern
        
        return noisy_image
    
    def _apply_row_column_noise(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply row and column noise to the image (simulates sensor readout issues).
        
        Args:
            image: Input image
            params: Parameters for row/column noise
                   - row_std: Standard deviation for row noise
                   - col_std: Standard deviation for column noise
                   - row_prob: Probability of a row being affected
                   - col_prob: Probability of a column being affected
            
        Returns:
            Image with row and column noise applied
        """
        row_std = params.get('row_std', 2.0)
        col_std = params.get('col_std', 2.0)
        row_prob = params.get('row_prob', 0.05)
        col_prob = params.get('col_prob', 0.05)
        
        noisy_image = image.copy()
        h, w = image.shape
        
        # Apply row noise
        for i in range(h):
            if np.random.random() < row_prob:
                noise_value = np.random.normal(0, row_std)
                noisy_image[i, :] += noise_value
        
        # Apply column noise
        for j in range(w):
            if np.random.random() < col_prob:
                noise_value = np.random.normal(0, col_std)
                noisy_image[:, j] += noise_value
        
        return noisy_image
    
    def simulate_camera_characteristics(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate specific thermal camera characteristics beyond just noise.
        
        Args:
            image: Input thermal image
            
        Returns:
            Image with simulated camera characteristics
        """
        camera_model = self.config.get('camera_model', 'generic')
        processed_image = image.copy()
        
        # Apply camera-specific processing
        if camera_model == 'flir_lepton':
            # FLIR Lepton has lower sensitivity at extreme temperatures
            processed_image = self._apply_nonlinear_response(processed_image, 0.9)
            # Add vignetting (edges are cooler than center)
            processed_image = self._apply_vignetting(processed_image, 0.85)
        elif camera_model == 'seek_thermal':
            # Seek Thermal has different characteristics
            processed_image = self._apply_nonlinear_response(processed_image, 1.1)
            processed_image = self._apply_vignetting(processed_image, 0.9)
        else:
            # Generic thermal camera
            processed_image = self._apply_nonlinear_response(processed_image, 1.0)
            processed_image = self._apply_vignetting(processed_image, 0.95)
        
        return processed_image
    
    def _apply_nonlinear_response(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply nonlinear response curve to simulate sensor response.
        
        Args:
            image: Input image
            gamma: Gamma correction factor
            
        Returns:
            Image with nonlinear response applied
        """
        # Normalize to [0, 1] for gamma correction
        min_val = np.min(image)
        max_val = np.max(image)
        normalized = (image - min_val) / (max_val - min_val + 1e-10)
        
        # Apply gamma correction
        corrected = np.power(normalized, gamma)
        
        # Scale back to original range
        result = corrected * (max_val - min_val) + min_val
        
        return result
    
    def _apply_vignetting(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Apply vignetting effect (edges darker than center).
        
        Args:
            image: Input image
            strength: Vignetting strength (0 to 1, where 1 means no vignetting)
            
        Returns:
            Image with vignetting applied
        """
        h, w = image.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Calculate distance from center (normalized)
        center_y, center_x = h / 2, w / 2
        distance = np.sqrt(((y - center_y) / center_y) ** 2 + ((x - center_x) / center_x) ** 2)
        
        # Create vignetting mask
        vignette = strength + (1 - strength) * (1 - distance ** 2)
        vignette = np.clip(vignette, 0, 1)
        
        # Apply vignetting
        result = image * vignette
        
        return result