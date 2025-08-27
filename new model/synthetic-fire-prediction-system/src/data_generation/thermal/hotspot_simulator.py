"""
Hotspot simulation for thermal images.

This module provides functionality for simulating realistic hotspots
with configurable parameters for size, intensity, and growth rate.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
from datetime import datetime, timedelta
import math


class HotspotShape(Enum):
    """Enumeration of supported hotspot shapes."""
    CIRCULAR = "circular"
    ELLIPTICAL = "elliptical"
    RECTANGULAR = "rectangular"
    IRREGULAR = "irregular"
    POINT = "point"


class GrowthPattern(Enum):
    """Enumeration of supported growth patterns."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    SIGMOID = "sigmoid"
    PULSATING = "pulsating"
    STEP = "step"


class HotspotSimulator:
    """
    Class for simulating realistic hotspots in thermal images.
    
    This class provides methods for generating hotspots with various shapes,
    intensities, and growth patterns that mimic real-world fire behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hotspot simulator with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters for hotspot simulation
                   - min_temperature: Minimum temperature value (ambient)
                   - max_temperature: Maximum temperature value (hottest point)
                   - temperature_unit: Temperature unit ('celsius', 'fahrenheit', 'kelvin')
                   - default_shape: Default hotspot shape
                   - default_growth: Default growth pattern
        """
        self.config = config
        self.validate_config()
        
        # Set default values
        self.min_temp = self.config.get('min_temperature', 20.0)  # Default ambient temp: 20°C
        self.max_temp = self.config.get('max_temperature', 1000.0)  # Default max temp: 1000°C
        self.temp_unit = self.config.get('temperature_unit', 'celsius')
        self.default_shape = HotspotShape(self.config.get('default_shape', 'circular'))
        self.default_growth = GrowthPattern(self.config.get('default_growth', 'exponential'))
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check temperature values
        if 'min_temperature' in self.config and 'max_temperature' in self.config:
            if self.config['min_temperature'] >= self.config['max_temperature']:
                raise ValueError("min_temperature must be less than max_temperature")
        
        # Check temperature unit
        if 'temperature_unit' in self.config:
            valid_units = ['celsius', 'fahrenheit', 'kelvin']
            if self.config['temperature_unit'] not in valid_units:
                raise ValueError(f"Invalid temperature unit. Must be one of: {valid_units}")
        
        # Check shape
        if 'default_shape' in self.config:
            try:
                HotspotShape(self.config['default_shape'])
            except ValueError:
                valid_shapes = [shape.value for shape in HotspotShape]
                raise ValueError(f"Invalid default shape. Must be one of: {valid_shapes}")
        
        # Check growth pattern
        if 'default_growth' in self.config:
            try:
                GrowthPattern(self.config['default_growth'])
            except ValueError:
                valid_patterns = [pattern.value for pattern in GrowthPattern]
                raise ValueError(f"Invalid default growth pattern. Must be one of: {valid_patterns}")
    
    def generate_hotspot(self, 
                        image_shape: Tuple[int, int],
                        center: Optional[Tuple[int, int]] = None,
                        radius: Optional[int] = None,
                        intensity: Optional[float] = None,
                        shape: Optional[Union[str, HotspotShape]] = None) -> np.ndarray:
        """
        Generate a single hotspot on a blank thermal image.
        
        Args:
            image_shape: Shape of the thermal image (height, width)
            center: Optional center coordinates of the hotspot (y, x)
            radius: Optional radius of the hotspot in pixels
            intensity: Optional intensity factor (0.0 to 1.0)
            shape: Optional hotspot shape
            
        Returns:
            2D numpy array representing the hotspot
        """
        h, w = image_shape
        
        # Set default values if not provided
        if center is None:
            center = (h // 2, w // 2)
        
        if radius is None:
            radius = min(h, w) // 10  # Default radius is 10% of the smaller dimension
        
        if intensity is None:
            intensity = 1.0
        
        if shape is None:
            shape_enum = self.default_shape
        elif isinstance(shape, str):
            shape_enum = HotspotShape(shape)
        else:
            shape_enum = shape
        
        # Create blank image
        hotspot = np.zeros(image_shape, dtype=np.float32)
        
        # Generate hotspot based on shape
        if shape_enum == HotspotShape.CIRCULAR:
            hotspot = self._generate_circular_hotspot(hotspot, center, radius, intensity)
        elif shape_enum == HotspotShape.ELLIPTICAL:
            # For elliptical, interpret radius as the major axis
            minor_axis = max(3, int(radius * 0.6))  # Minor axis is 60% of major axis
            hotspot = self._generate_elliptical_hotspot(hotspot, center, radius, minor_axis, intensity)
        elif shape_enum == HotspotShape.RECTANGULAR:
            # For rectangular, interpret radius as half the side length
            width = radius * 2
            height = int(radius * 1.5)  # Rectangle is 1.5 times taller than wide
            hotspot = self._generate_rectangular_hotspot(hotspot, center, width, height, intensity)
        elif shape_enum == HotspotShape.IRREGULAR:
            hotspot = self._generate_irregular_hotspot(hotspot, center, radius, intensity)
        elif shape_enum == HotspotShape.POINT:
            hotspot = self._generate_point_hotspot(hotspot, center, intensity)
        
        # Scale hotspot to temperature range
        temp_range = self.max_temp - self.min_temp
        hotspot = hotspot * temp_range + self.min_temp
        
        return hotspot
    
    def _generate_circular_hotspot(self, 
                                 image: np.ndarray, 
                                 center: Tuple[int, int], 
                                 radius: int, 
                                 intensity: float) -> np.ndarray:
        """
        Generate a circular hotspot.
        
        Args:
            image: Base image
            center: Center coordinates (y, x)
            radius: Radius in pixels
            intensity: Intensity factor
            
        Returns:
            Image with circular hotspot
        """
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        
        # Calculate distance from center for each pixel
        dist_from_center = np.sqrt((y - center[0])**2 + (x - center[1])**2)
        
        # Create circular mask with smooth falloff
        mask = np.clip(1.0 - dist_from_center / radius, 0, 1)
        
        # Apply quadratic falloff for more realistic heat distribution
        mask = mask**2
        
        # Apply intensity
        mask = mask * intensity
        
        return mask
    
    def _generate_elliptical_hotspot(self, 
                                   image: np.ndarray, 
                                   center: Tuple[int, int], 
                                   major_axis: int, 
                                   minor_axis: int, 
                                   intensity: float,
                                   angle: float = 0.0) -> np.ndarray:
        """
        Generate an elliptical hotspot.
        
        Args:
            image: Base image
            center: Center coordinates (y, x)
            major_axis: Major axis length in pixels
            minor_axis: Minor axis length in pixels
            intensity: Intensity factor
            angle: Rotation angle in radians
            
        Returns:
            Image with elliptical hotspot
        """
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        
        # Translate to origin
        y_centered = y - center[0]
        x_centered = x - center[1]
        
        # Rotate if angle is provided
        if angle != 0.0:
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            y_rotated = y_centered * cos_angle - x_centered * sin_angle
            x_rotated = y_centered * sin_angle + x_centered * cos_angle
            y_centered = y_rotated
            x_centered = x_rotated
        
        # Calculate normalized distance from center for each pixel
        dist_from_center = np.sqrt((y_centered / major_axis)**2 + (x_centered / minor_axis)**2)
        
        # Create elliptical mask with smooth falloff
        mask = np.clip(1.0 - dist_from_center, 0, 1)
        
        # Apply quadratic falloff for more realistic heat distribution
        mask = mask**2
        
        # Apply intensity
        mask = mask * intensity
        
        return mask
    
    def _generate_rectangular_hotspot(self, 
                                    image: np.ndarray, 
                                    center: Tuple[int, int], 
                                    width: int, 
                                    height: int, 
                                    intensity: float,
                                    angle: float = 0.0) -> np.ndarray:
        """
        Generate a rectangular hotspot.
        
        Args:
            image: Base image
            center: Center coordinates (y, x)
            width: Width in pixels
            height: Height in pixels
            intensity: Intensity factor
            angle: Rotation angle in radians
            
        Returns:
            Image with rectangular hotspot
        """
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        
        # Translate to origin
        y_centered = y - center[0]
        x_centered = x - center[1]
        
        # Rotate if angle is provided
        if angle != 0.0:
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            y_rotated = y_centered * cos_angle - x_centered * sin_angle
            x_rotated = y_centered * sin_angle + x_centered * cos_angle
            y_centered = y_rotated
            x_centered = x_rotated
        
        # Calculate normalized distance from center for each pixel
        half_height = height / 2
        half_width = width / 2
        
        # Create rectangular mask with smooth falloff
        y_dist = np.abs(y_centered) / half_height
        x_dist = np.abs(x_centered) / half_width
        
        # Use smooth step function for edges
        y_mask = 1.0 - np.clip(y_dist, 0, 1)
        x_mask = 1.0 - np.clip(x_dist, 0, 1)
        
        # Combine masks
        mask = y_mask * x_mask
        
        # Apply intensity
        mask = mask * intensity
        
        return mask
    
    def _generate_irregular_hotspot(self, 
                                  image: np.ndarray, 
                                  center: Tuple[int, int], 
                                  radius: int, 
                                  intensity: float) -> np.ndarray:
        """
        Generate an irregular hotspot using perlin noise.
        
        Args:
            image: Base image
            center: Center coordinates (y, x)
            radius: Base radius in pixels
            intensity: Intensity factor
            
        Returns:
            Image with irregular hotspot
        """
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        
        # Calculate distance from center for each pixel
        dist_from_center = np.sqrt((y - center[0])**2 + (x - center[1])**2)
        
        # Create base circular mask
        base_mask = np.clip(1.0 - dist_from_center / radius, 0, 1)
        
        # Generate perlin-like noise for irregularity
        noise_scale = 0.1
        noise = self._generate_simplex_noise((h, w), scale=noise_scale)
        
        # Apply noise to distort the shape
        distortion_strength = 0.3
        distorted_mask = base_mask * (1.0 + distortion_strength * noise)
        
        # Normalize and clip
        distorted_mask = np.clip(distorted_mask, 0, 1)
        
        # Apply quadratic falloff for more realistic heat distribution
        mask = distorted_mask**2
        
        # Apply intensity
        mask = mask * intensity
        
        return mask
    
    def _generate_point_hotspot(self, 
                              image: np.ndarray, 
                              center: Tuple[int, int], 
                              intensity: float) -> np.ndarray:
        """
        Generate a point hotspot (single pixel).
        
        Args:
            image: Base image
            center: Center coordinates (y, x)
            intensity: Intensity factor
            
        Returns:
            Image with point hotspot
        """
        result = image.copy()
        
        # Ensure center is within image bounds
        h, w = image.shape
        y, x = min(max(0, center[0]), h-1), min(max(0, center[1]), w-1)
        
        # Set the center pixel to the intensity value
        result[y, x] = intensity
        
        return result
    
    def _generate_simplex_noise(self, shape: Tuple[int, int], scale: float = 0.1) -> np.ndarray:
        """
        Generate simplex-like noise for irregular shapes.
        
        Args:
            shape: Shape of the output array
            scale: Scale factor for noise frequency
            
        Returns:
            Array of noise values
        """
        h, w = shape
        
        # Create a grid of coordinates
        y_coords = np.linspace(0, h * scale, h)
        x_coords = np.linspace(0, w * scale, w)
        
        # Create meshgrid
        x, y = np.meshgrid(x_coords, y_coords)
        
        # Generate some noise using sin functions at different frequencies
        noise = np.zeros((h, w))
        
        # Add multiple octaves of noise
        noise += 0.5 * np.sin(x * 5.0 + y * 5.0)
        noise += 0.25 * np.sin(x * 10.0 + y * 10.0)
        noise += 0.125 * np.sin(x * 20.0 + y * 20.0)
        noise += 0.0625 * np.sin(x * 40.0 + y * 40.0)
        
        # Normalize to [-1, 1]
        noise = noise / np.max(np.abs(noise))
        
        return noise
    
    def calculate_growth_factor(self, 
                              time_elapsed: float, 
                              total_duration: float,
                              growth_pattern: Optional[Union[str, GrowthPattern]] = None) -> float:
        """
        Calculate growth factor based on elapsed time and growth pattern.
        
        Args:
            time_elapsed: Time elapsed since start in seconds
            total_duration: Total duration of growth in seconds
            growth_pattern: Growth pattern to use
            
        Returns:
            Growth factor between 0.0 and 1.0
        """
        # Normalize time to [0, 1]
        t = min(max(time_elapsed / total_duration, 0.0), 1.0)
        
        # Set growth pattern
        if growth_pattern is None:
            pattern = self.default_growth
        elif isinstance(growth_pattern, str):
            pattern = GrowthPattern(growth_pattern)
        else:
            pattern = growth_pattern
        
        # Calculate growth factor based on pattern
        if pattern == GrowthPattern.LINEAR:
            return t
        elif pattern == GrowthPattern.EXPONENTIAL:
            return t**2
        elif pattern == GrowthPattern.LOGARITHMIC:
            return np.log(1 + 9 * t) / np.log(10)  # log10(1 + 9t) maps [0,1] to [0,1]
        elif pattern == GrowthPattern.SIGMOID:
            # Logistic function centered at t=0.5
            k = 10  # Steepness
            return 1.0 / (1.0 + np.exp(-k * (t - 0.5)))
        elif pattern == GrowthPattern.PULSATING:
            # Pulsating pattern with overall growth
            pulse = 0.1 * np.sin(2 * np.pi * 5 * t)  # 5 pulses over the duration
            return t + pulse * t
        elif pattern == GrowthPattern.STEP:
            # Step function with 5 steps
            return np.floor(t * 5) / 4
        
        # Default to linear
        return t
    
    def generate_evolving_hotspot(self, 
                                image_shape: Tuple[int, int],
                                start_time: datetime,
                                current_time: datetime,
                                duration: timedelta,
                                params: Dict[str, Any]) -> np.ndarray:
        """
        Generate a hotspot that evolves over time.
        
        Args:
            image_shape: Shape of the thermal image (height, width)
            start_time: Start time of the hotspot
            current_time: Current time for which to generate the hotspot
            duration: Total duration of the hotspot evolution
            params: Parameters for the hotspot
                   - center: Center coordinates (y, x)
                   - max_radius: Maximum radius in pixels
                   - max_intensity: Maximum intensity factor
                   - shape: Hotspot shape
                   - growth_pattern: Growth pattern
            
        Returns:
            2D numpy array representing the evolved hotspot
        """
        # Extract parameters
        center = params.get('center')
        max_radius = params.get('max_radius', min(image_shape) // 5)
        max_intensity = params.get('max_intensity', 1.0)
        shape = params.get('shape', self.default_shape)
        growth_pattern = params.get('growth_pattern', self.default_growth)
        
        # Calculate elapsed time in seconds
        elapsed_seconds = (current_time - start_time).total_seconds()
        total_seconds = duration.total_seconds()
        
        # If before start time or after end time, return empty image
        if elapsed_seconds < 0 or elapsed_seconds > total_seconds:
            return np.zeros(image_shape, dtype=np.float32) + self.min_temp
        
        # Calculate growth factor
        growth_factor = self.calculate_growth_factor(elapsed_seconds, total_seconds, growth_pattern)
        
        # Calculate current radius and intensity
        current_radius = int(max_radius * growth_factor)
        current_intensity = max_intensity * growth_factor
        
        # Generate hotspot
        hotspot = self.generate_hotspot(
            image_shape=image_shape,
            center=center,
            radius=current_radius,
            intensity=current_intensity,
            shape=shape
        )
        
        return hotspot
    
    def generate_multiple_hotspots(self, 
                                 image_shape: Tuple[int, int],
                                 hotspots: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate multiple hotspots on a single image.
        
        Args:
            image_shape: Shape of the thermal image (height, width)
            hotspots: List of hotspot parameters, each containing:
                     - center: Center coordinates (y, x)
                     - radius: Radius in pixels
                     - intensity: Intensity factor
                     - shape: Hotspot shape
            
        Returns:
            2D numpy array representing the combined hotspots
        """
        # Create blank image with ambient temperature
        result = np.zeros(image_shape, dtype=np.float32) + self.min_temp
        
        # Generate each hotspot and combine them
        for hotspot_params in hotspots:
            hotspot = self.generate_hotspot(
                image_shape=image_shape,
                center=hotspot_params.get('center'),
                radius=hotspot_params.get('radius'),
                intensity=hotspot_params.get('intensity'),
                shape=hotspot_params.get('shape')
            )
            
            # Use maximum temperature at each pixel
            result = np.maximum(result, hotspot)
        
        return result
    
    def generate_evolving_multiple_hotspots(self, 
                                          image_shape: Tuple[int, int],
                                          start_time: datetime,
                                          current_time: datetime,
                                          hotspots: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate multiple evolving hotspots on a single image.
        
        Args:
            image_shape: Shape of the thermal image (height, width)
            start_time: Start time of the simulation
            current_time: Current time for which to generate the hotspots
            hotspots: List of hotspot parameters, each containing:
                     - center: Center coordinates (y, x)
                     - max_radius: Maximum radius in pixels
                     - max_intensity: Maximum intensity factor
                     - shape: Hotspot shape
                     - growth_pattern: Growth pattern
                     - start_offset: Offset from start_time in seconds
                     - duration: Duration of the hotspot evolution in seconds
            
        Returns:
            2D numpy array representing the combined evolved hotspots
        """
        # Create blank image with ambient temperature
        result = np.zeros(image_shape, dtype=np.float32) + self.min_temp
        
        # Generate each evolving hotspot and combine them
        for hotspot_params in hotspots:
            # Calculate hotspot-specific start time with offset
            offset_seconds = hotspot_params.get('start_offset', 0)
            hotspot_start = start_time + timedelta(seconds=offset_seconds)
            
            # Get duration
            duration_seconds = hotspot_params.get('duration', 300)  # Default: 5 minutes
            duration = timedelta(seconds=duration_seconds)
            
            # Generate evolving hotspot
            hotspot = self.generate_evolving_hotspot(
                image_shape=image_shape,
                start_time=hotspot_start,
                current_time=current_time,
                duration=duration,
                params=hotspot_params
            )
            
            # Use maximum temperature at each pixel
            result = np.maximum(result, hotspot)
        
        return result