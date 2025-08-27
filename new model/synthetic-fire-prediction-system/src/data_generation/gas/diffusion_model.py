"""
Gas diffusion model for synthetic fire data.

This module provides functionality for modeling spatial gas distribution patterns
based on physical properties of gases and environmental factors.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
from datetime import datetime, timedelta
import math


class DiffusionType(Enum):
    """Enumeration of supported diffusion types."""
    GAUSSIAN = "gaussian"
    ADVECTION_DIFFUSION = "advection_diffusion"
    PLUME = "plume"
    COMPARTMENTAL = "compartmental"
    CUSTOM = "custom"


class DiffusionModel:
    """
    Class for modeling spatial gas distribution patterns.
    
    This class provides methods for simulating gas diffusion based on physical properties
    and environmental factors, creating 2D/3D gas concentration maps.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the diffusion model with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
                - diffusion_type: Type of diffusion model to use
                - grid_resolution: Resolution of the diffusion grid (x, y, z)
                - time_step: Time step for diffusion simulation in seconds
                - boundary_conditions: Type of boundary conditions
                - environmental_factors: Dictionary of environmental factors
        """
        self.config = config
        self.validate_config()
        
        # Set default values
        self.diffusion_type = DiffusionType(self.config.get('diffusion_type', 'gaussian'))
        self.grid_resolution = self.config.get('grid_resolution', (50, 50, 10))
        self.time_step = self.config.get('time_step', 1.0)
        self.boundary_conditions = self.config.get('boundary_conditions', 'reflective')
        
        # Initialize environmental factors
        self.env_factors = self.config.get('environmental_factors', {
            'temperature': 25.0,  # °C
            'pressure': 101.3,    # kPa
            'humidity': 50.0,     # %
            'air_flow': [0.0, 0.0, 0.0]  # m/s in x, y, z directions
        })
        
        # Gas properties (diffusion coefficients in air at 25°C, 1 atm in cm²/s)
        self.gas_properties = {
            'methane': {
                'diffusion_coefficient': 0.196,
                'molar_mass': 16.04,
                'density': 0.668  # kg/m³ at 25°C, 1 atm
            },
            'propane': {
                'diffusion_coefficient': 0.10,
                'molar_mass': 44.1,
                'density': 1.882  # kg/m³ at 25°C, 1 atm
            },
            'hydrogen': {
                'diffusion_coefficient': 0.61,
                'molar_mass': 2.02,
                'density': 0.0899  # kg/m³ at 25°C, 1 atm
            },
            'carbon_monoxide': {
                'diffusion_coefficient': 0.208,
                'molar_mass': 28.01,
                'density': 1.145  # kg/m³ at 25°C, 1 atm
            }
        }
        
        # Initialize concentration grid
        self.initialize_grid()
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if 'diffusion_type' in self.config:
            try:
                DiffusionType(self.config['diffusion_type'])
            except ValueError:
                valid_types = [dt.value for dt in DiffusionType]
                raise ValueError(f"Invalid diffusion type. Must be one of: {valid_types}")
        
        if 'grid_resolution' in self.config:
            grid_res = self.config['grid_resolution']
            if not isinstance(grid_res, tuple) or len(grid_res) != 3:
                raise ValueError("grid_resolution must be a tuple of (x, y, z) dimensions")
            
            if any(dim <= 0 for dim in grid_res):
                raise ValueError("grid_resolution dimensions must be positive")
        
        if 'time_step' in self.config and self.config['time_step'] <= 0:
            raise ValueError("time_step must be positive")
        
        if 'boundary_conditions' in self.config:
            valid_bc = ['reflective', 'absorbing', 'periodic', 'fixed']
            if self.config['boundary_conditions'] not in valid_bc:
                raise ValueError(f"Invalid boundary conditions. Must be one of: {valid_bc}")
    
    def initialize_grid(self) -> None:
        """
        Initialize the concentration grid based on the configured resolution.
        """
        x, y, z = self.grid_resolution
        self.grid = np.zeros((x, y, z), dtype=np.float32)
        
        # Initialize additional grids for advection-diffusion model
        if self.diffusion_type == DiffusionType.ADVECTION_DIFFUSION:
            self.velocity_field = np.zeros((x, y, z, 3), dtype=np.float32)
            self.update_velocity_field()
    
    def update_velocity_field(self) -> None:
        """
        Update the velocity field based on environmental factors.
        """
        if self.diffusion_type != DiffusionType.ADVECTION_DIFFUSION:
            return
        
        x, y, z = self.grid_resolution
        air_flow = self.env_factors['air_flow']
        
        # Set base velocity from air flow
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    self.velocity_field[i, j, k] = air_flow
        
        # Add some spatial variation
        noise_scale = 0.2  # 20% variation
        noise = np.random.normal(0, noise_scale, (x, y, z, 3))
        self.velocity_field += noise * np.array(air_flow)
    
    def adjust_diffusion_coefficient(self, 
                                    gas_type: str, 
                                    temperature: float, 
                                    pressure: float) -> float:
        """
        Adjust diffusion coefficient based on temperature and pressure.
        
        Uses the Chapman-Enskog theory approximation.
        
        Args:
            gas_type: Type of gas
            temperature: Temperature in Celsius
            pressure: Pressure in kPa
            
        Returns:
            Adjusted diffusion coefficient in cm²/s
        """
        if gas_type not in self.gas_properties:
            raise ValueError(f"Unknown gas type: {gas_type}")
        
        # Get base diffusion coefficient at 25°C and 101.3 kPa
        D_0 = self.gas_properties[gas_type]['diffusion_coefficient']
        
        # Convert temperature to Kelvin
        T_0 = 298.15  # 25°C in Kelvin
        T = temperature + 273.15
        
        # Convert pressure to atm
        P_0 = 101.3  # kPa
        P = pressure
        
        # Adjust diffusion coefficient using Chapman-Enskog relation
        # D ∝ T^(3/2) / P
        D = D_0 * (T / T_0) ** 1.5 * (P_0 / P)
        
        return D
    
    def simulate_diffusion(self, 
                          gas_type: str, 
                          source_points: List[Dict[str, Any]], 
                          duration: float) -> np.ndarray:
        """
        Simulate gas diffusion from source points for a specified duration.
        
        Args:
            gas_type: Type of gas
            source_points: List of source points, each with:
                - position: (x, y, z) coordinates
                - strength: Emission rate in g/s
                - start_time: Start time of emission in seconds from simulation start
                - duration: Duration of emission in seconds
            duration: Total simulation duration in seconds
            
        Returns:
            3D numpy array representing the final gas concentration distribution
        """
        if self.diffusion_type == DiffusionType.GAUSSIAN:
            return self._simulate_gaussian_diffusion(gas_type, source_points, duration)
        elif self.diffusion_type == DiffusionType.ADVECTION_DIFFUSION:
            return self._simulate_advection_diffusion(gas_type, source_points, duration)
        elif self.diffusion_type == DiffusionType.PLUME:
            return self._simulate_plume_model(gas_type, source_points, duration)
        elif self.diffusion_type == DiffusionType.COMPARTMENTAL:
            return self._simulate_compartmental_model(gas_type, source_points, duration)
        else:
            # Default to Gaussian diffusion
            return self._simulate_gaussian_diffusion(gas_type, source_points, duration)
    
    def _simulate_gaussian_diffusion(self, 
                                   gas_type: str, 
                                   source_points: List[Dict[str, Any]], 
                                   duration: float) -> np.ndarray:
        """
        Simulate gas diffusion using a Gaussian diffusion model.
        
        Args:
            gas_type: Type of gas
            source_points: List of source points
            duration: Total simulation duration in seconds
            
        Returns:
            3D numpy array representing the final gas concentration distribution
        """
        # Reset grid
        self.initialize_grid()
        
        # Get diffusion coefficient adjusted for environmental conditions
        D = self.adjust_diffusion_coefficient(
            gas_type, 
            self.env_factors['temperature'], 
            self.env_factors['pressure']
        )
        
        # Convert from cm²/s to grid units²/s
        # Assuming each grid cell is 10cm x 10cm x 10cm
        grid_cell_size = 0.1  # meters
        D_grid = D * 1e-4 / (grid_cell_size ** 2)  # Convert cm²/s to m²/s to grid units²/s
        
        # Number of time steps
        n_steps = int(duration / self.time_step)
        
        for step in range(n_steps):
            current_time = step * self.time_step
            
            # Add source contributions
            for source in source_points:
                pos = source['position']
                strength = source['strength']
                start_time = source.get('start_time', 0)
                src_duration = source.get('duration', duration)
                
                # Check if source is active at current time
                if start_time <= current_time < (start_time + src_duration):
                    # Convert position to grid indices
                    i, j, k = [int(p) for p in pos]
                    
                    # Ensure indices are within grid bounds
                    if (0 <= i < self.grid_resolution[0] and 
                        0 <= j < self.grid_resolution[1] and 
                        0 <= k < self.grid_resolution[2]):
                        # Add source contribution
                        self.grid[i, j, k] += strength * self.time_step
            
            # Diffuse using discrete Laplacian
            new_grid = self.grid.copy()
            
            for i in range(1, self.grid_resolution[0] - 1):
                for j in range(1, self.grid_resolution[1] - 1):
                    for k in range(1, self.grid_resolution[2] - 1):
                        # Discrete Laplacian (6-point stencil in 3D)
                        laplacian = (
                            self.grid[i+1, j, k] + self.grid[i-1, j, k] +
                            self.grid[i, j+1, k] + self.grid[i, j-1, k] +
                            self.grid[i, j, k+1] + self.grid[i, j, k-1] -
                            6 * self.grid[i, j, k]
                        )
                        
                        # Update concentration using diffusion equation
                        new_grid[i, j, k] += D_grid * self.time_step * laplacian
            
            # Apply boundary conditions
            new_grid = self._apply_boundary_conditions(new_grid)
            
            # Update grid
            self.grid = new_grid
        
        return self.grid
    
    def _simulate_advection_diffusion(self, 
                                    gas_type: str, 
                                    source_points: List[Dict[str, Any]], 
                                    duration: float) -> np.ndarray:
        """
        Simulate gas diffusion using an advection-diffusion model.
        
        Args:
            gas_type: Type of gas
            source_points: List of source points
            duration: Total simulation duration in seconds
            
        Returns:
            3D numpy array representing the final gas concentration distribution
        """
        # Reset grid
        self.initialize_grid()
        self.update_velocity_field()
        
        # Get diffusion coefficient adjusted for environmental conditions
        D = self.adjust_diffusion_coefficient(
            gas_type, 
            self.env_factors['temperature'], 
            self.env_factors['pressure']
        )
        
        # Convert from cm²/s to grid units²/s
        grid_cell_size = 0.1  # meters
        D_grid = D * 1e-4 / (grid_cell_size ** 2)  # Convert cm²/s to m²/s to grid units²/s
        
        # Number of time steps
        n_steps = int(duration / self.time_step)
        
        for step in range(n_steps):
            current_time = step * self.time_step
            
            # Add source contributions
            for source in source_points:
                pos = source['position']
                strength = source['strength']
                start_time = source.get('start_time', 0)
                src_duration = source.get('duration', duration)
                
                # Check if source is active at current time
                if start_time <= current_time < (start_time + src_duration):
                    # Convert position to grid indices
                    i, j, k = [int(p) for p in pos]
                    
                    # Ensure indices are within grid bounds
                    if (0 <= i < self.grid_resolution[0] and 
                        0 <= j < self.grid_resolution[1] and 
                        0 <= k < self.grid_resolution[2]):
                        # Add source contribution
                        self.grid[i, j, k] += strength * self.time_step
            
            # Solve advection-diffusion equation using operator splitting
            # First solve diffusion
            new_grid = self.grid.copy()
            
            for i in range(1, self.grid_resolution[0] - 1):
                for j in range(1, self.grid_resolution[1] - 1):
                    for k in range(1, self.grid_resolution[2] - 1):
                        # Discrete Laplacian (6-point stencil in 3D)
                        laplacian = (
                            self.grid[i+1, j, k] + self.grid[i-1, j, k] +
                            self.grid[i, j+1, k] + self.grid[i, j-1, k] +
                            self.grid[i, j, k+1] + self.grid[i, j, k-1] -
                            6 * self.grid[i, j, k]
                        )
                        
                        # Update concentration using diffusion equation
                        new_grid[i, j, k] += D_grid * self.time_step * laplacian
            
            # Apply boundary conditions
            new_grid = self._apply_boundary_conditions(new_grid)
            
            # Then solve advection using upwind scheme
            advected_grid = new_grid.copy()
            
            for i in range(1, self.grid_resolution[0] - 1):
                for j in range(1, self.grid_resolution[1] - 1):
                    for k in range(1, self.grid_resolution[2] - 1):
                        vx, vy, vz = self.velocity_field[i, j, k]
                        
                        # X-direction advection
                        if vx > 0:
                            flux_x = vx * (new_grid[i, j, k] - new_grid[i-1, j, k])
                        else:
                            flux_x = vx * (new_grid[i+1, j, k] - new_grid[i, j, k])
                        
                        # Y-direction advection
                        if vy > 0:
                            flux_y = vy * (new_grid[i, j, k] - new_grid[i, j-1, k])
                        else:
                            flux_y = vy * (new_grid[i, j+1, k] - new_grid[i, j, k])
                        
                        # Z-direction advection
                        if vz > 0:
                            flux_z = vz * (new_grid[i, j, k] - new_grid[i, j, k-1])
                        else:
                            flux_z = vz * (new_grid[i, j, k+1] - new_grid[i, j, k])
                        
                        # Update concentration using advection equation
                        advected_grid[i, j, k] -= self.time_step * (flux_x + flux_y + flux_z)
            
            # Apply boundary conditions
            advected_grid = self._apply_boundary_conditions(advected_grid)
            
            # Update grid
            self.grid = advected_grid
        
        return self.grid
    
    def _simulate_plume_model(self, 
                            gas_type: str, 
                            source_points: List[Dict[str, Any]], 
                            duration: float) -> np.ndarray:
        """
        Simulate gas diffusion using a Gaussian plume model.
        
        Args:
            gas_type: Type of gas
            source_points: List of source points
            duration: Total simulation duration in seconds
            
        Returns:
            3D numpy array representing the final gas concentration distribution
        """
        # Reset grid
        self.initialize_grid()
        
        # Get wind direction and speed from environmental factors
        wind_vector = np.array(self.env_factors['air_flow'])
        wind_speed = np.linalg.norm(wind_vector)
        
        if wind_speed < 0.01:
            # If wind speed is negligible, use simple Gaussian diffusion
            return self._simulate_gaussian_diffusion(gas_type, source_points, duration)
        
        # Normalize wind direction
        wind_direction = wind_vector / wind_speed if wind_speed > 0 else np.array([1, 0, 0])
        
        # Get atmospheric stability class (simplified)
        # A: Very unstable, B: Unstable, C: Slightly unstable, D: Neutral, E: Stable, F: Very stable
        stability_class = self.config.get('stability_class', 'D')
        
        # Dispersion coefficients based on stability class (simplified Pasquill-Gifford)
        dispersion_params = {
            'A': {'a_y': 0.22, 'b_y': 0.0001, 'a_z': 0.20, 'b_z': 0},
            'B': {'a_y': 0.16, 'b_y': 0.0001, 'a_z': 0.12, 'b_z': 0},
            'C': {'a_y': 0.11, 'b_y': 0.0001, 'a_z': 0.08, 'b_z': 0.0002},
            'D': {'a_y': 0.08, 'b_y': 0.0001, 'a_z': 0.06, 'b_z': 0.0015},
            'E': {'a_y': 0.06, 'b_y': 0.0001, 'a_z': 0.03, 'b_z': 0.0003},
            'F': {'a_y': 0.04, 'b_y': 0.0001, 'a_z': 0.016, 'b_z': 0.0003}
        }
        
        params = dispersion_params.get(stability_class, dispersion_params['D'])
        
        # Grid cell size in meters
        grid_cell_size = 0.1
        
        # Calculate plume for each source point
        for source in source_points:
            pos = source['position']
            strength = source['strength']  # g/s
            
            # Convert position to grid indices and then to meters
            source_x, source_y, source_z = [p * grid_cell_size for p in pos]
            
            # Calculate concentration at each grid point
            for i in range(self.grid_resolution[0]):
                for j in range(self.grid_resolution[1]):
                    for k in range(self.grid_resolution[2]):
                        # Convert grid indices to meters
                        x = i * grid_cell_size
                        y = j * grid_cell_size
                        z = k * grid_cell_size
                        
                        # Calculate distance from source
                        dx = x - source_x
                        dy = y - source_y
                        dz = z - source_z
                        
                        # Project onto wind direction to get downwind distance
                        downwind_dist = dx * wind_direction[0] + dy * wind_direction[1] + dz * wind_direction[2]
                        
                        # Skip points upwind of the source
                        if downwind_dist < 0:
                            continue
                        
                        # Calculate crosswind distance
                        crosswind_vector = np.array([dx, dy, dz]) - downwind_dist * wind_direction
                        crosswind_dist = np.linalg.norm(crosswind_vector)
                        
                        # Calculate vertical distance
                        vertical_dist = abs(dz)
                        
                        # Calculate dispersion coefficients
                        sigma_y = params['a_y'] * downwind_dist ** 0.9
                        sigma_z = params['a_z'] * downwind_dist ** 0.9
                        
                        # Calculate concentration using Gaussian plume equation
                        if downwind_dist > 0 and sigma_y > 0 and sigma_z > 0:
                            # Reflection from ground
                            term1 = np.exp(-0.5 * (vertical_dist / sigma_z) ** 2)
                            term2 = np.exp(-0.5 * (2 * source_z - vertical_dist) ** 2 / sigma_z ** 2)
                            
                            conc = (strength / (2 * np.pi * sigma_y * sigma_z * wind_speed) * 
                                   np.exp(-0.5 * (crosswind_dist / sigma_y) ** 2) * 
                                   (term1 + term2))
                            
                            # Add to grid
                            self.grid[i, j, k] += conc
        
        return self.grid
    
    def _simulate_compartmental_model(self, 
                                    gas_type: str, 
                                    source_points: List[Dict[str, Any]], 
                                    duration: float) -> np.ndarray:
        """
        Simulate gas diffusion using a compartmental model.
        
        This model divides the space into compartments and models the flow between them.
        
        Args:
            gas_type: Type of gas
            source_points: List of source points
            duration: Total simulation duration in seconds
            
        Returns:
            3D numpy array representing the final gas concentration distribution
        """
        # Reset grid
        self.initialize_grid()
        
        # Define compartments (simplified as grid cells)
        # Each cell exchanges gas with its neighbors based on concentration differences
        
        # Get diffusion coefficient adjusted for environmental conditions
        D = self.adjust_diffusion_coefficient(
            gas_type, 
            self.env_factors['temperature'], 
            self.env_factors['pressure']
        )
        
        # Convert from cm²/s to grid units²/s
        grid_cell_size = 0.1  # meters
        D_grid = D * 1e-4 / (grid_cell_size ** 2)  # Convert cm²/s to m²/s to grid units²/s
        
        # Calculate exchange coefficients between compartments
        # Simplified as proportional to diffusion coefficient and inversely proportional to distance
        exchange_coeff = D_grid * self.time_step
        
        # Number of time steps
        n_steps = int(duration / self.time_step)
        
        for step in range(n_steps):
            current_time = step * self.time_step
            
            # Add source contributions
            for source in source_points:
                pos = source['position']
                strength = source['strength']
                start_time = source.get('start_time', 0)
                src_duration = source.get('duration', duration)
                
                # Check if source is active at current time
                if start_time <= current_time < (start_time + src_duration):
                    # Convert position to grid indices
                    i, j, k = [int(p) for p in pos]
                    
                    # Ensure indices are within grid bounds
                    if (0 <= i < self.grid_resolution[0] and 
                        0 <= j < self.grid_resolution[1] and 
                        0 <= k < self.grid_resolution[2]):
                        # Add source contribution
                        self.grid[i, j, k] += strength * self.time_step
            
            # Calculate gas exchange between compartments
            new_grid = self.grid.copy()
            
            for i in range(self.grid_resolution[0]):
                for j in range(self.grid_resolution[1]):
                    for k in range(self.grid_resolution[2]):
                        # Exchange with neighbors
                        for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                            ni, nj, nk = i + di, j + dj, k + dk
                            
                            # Check if neighbor is within bounds
                            if (0 <= ni < self.grid_resolution[0] and 
                                0 <= nj < self.grid_resolution[1] and 
                                0 <= nk < self.grid_resolution[2]):
                                
                                # Calculate concentration difference
                                diff = self.grid[ni, nj, nk] - self.grid[i, j, k]
                                
                                # Calculate flow based on concentration difference
                                flow = exchange_coeff * diff
                                
                                # Update concentration
                                new_grid[i, j, k] += flow
            
            # Apply boundary conditions
            new_grid = self._apply_boundary_conditions(new_grid)
            
            # Update grid
            self.grid = new_grid
        
        return self.grid
    
    def _apply_boundary_conditions(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply boundary conditions to the concentration grid.
        
        Args:
            grid: Concentration grid
            
        Returns:
            Grid with boundary conditions applied
        """
        if self.boundary_conditions == 'reflective':
            # Reflective boundaries (zero gradient at boundaries)
            # No need to modify the grid, as we're not updating boundary cells
            pass
        
        elif self.boundary_conditions == 'absorbing':
            # Absorbing boundaries (zero concentration at boundaries)
            x, y, z = self.grid_resolution
            
            # Set boundary cells to zero
            grid[0, :, :] = 0
            grid[x-1, :, :] = 0
            grid[:, 0, :] = 0
            grid[:, y-1, :] = 0
            grid[:, :, 0] = 0
            grid[:, :, z-1] = 0
        
        elif self.boundary_conditions == 'periodic':
            # Periodic boundaries (wrap around)
            x, y, z = self.grid_resolution
            
            # Average the opposite boundaries
            grid[0, :, :] = (grid[0, :, :] + grid[x-1, :, :]) / 2
            grid[x-1, :, :] = grid[0, :, :]
            
            grid[:, 0, :] = (grid[:, 0, :] + grid[:, y-1, :]) / 2
            grid[:, y-1, :] = grid[:, 0, :]
            
            grid[:, :, 0] = (grid[:, :, 0] + grid[:, :, z-1]) / 2
            grid[:, :, z-1] = grid[:, :, 0]
        
        elif self.boundary_conditions == 'fixed':
            # Fixed boundaries (maintain initial values)
            # No need to modify the grid, as we're not updating boundary cells
            pass
        
        return grid
    
    def get_concentration_at_point(self, point: Tuple[int, int, int]) -> float:
        """
        Get the gas concentration at a specific point in the grid.
        
        Args:
            point: (x, y, z) coordinates in grid units
            
        Returns:
            Gas concentration at the specified point
        """
        x, y, z = point
        
        # Check if point is within grid bounds
        if (0 <= x < self.grid_resolution[0] and 
            0 <= y < self.grid_resolution[1] and 
            0 <= z < self.grid_resolution[2]):
            return self.grid[x, y, z]
        else:
            return 0.0
    
    def get_concentration_slice(self, 
                               axis: str, 
                               index: int) -> np.ndarray:
        """
        Get a 2D slice of the concentration grid along the specified axis.
        
        Args:
            axis: Axis along which to slice ('x', 'y', or 'z')
            index: Index of the slice
            
        Returns:
            2D numpy array representing the concentration slice
        """
        if axis == 'x':
            if 0 <= index < self.grid_resolution[0]:
                return self.grid[index, :, :]