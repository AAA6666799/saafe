"""
Base scenario generator for synthetic sensor data.
"""

import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Iterator, Optional
import threading
import time

from .data_models import SensorReading, ScenarioConfig, SensorLimits


class BaseScenarioGenerator(ABC):
    """Abstract base class for scenario generators."""
    
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.limits = SensorLimits()
        self.current_time = datetime.now()
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._current_reading = None
        
    @abstractmethod
    def generate_baseline_values(self, sample_index: int) -> tuple:
        """Generate baseline sensor values for given sample index.
        
        Returns:
            tuple: (temperature, pm25, co2, audio_level)
        """
        pass
    
    def add_realistic_noise(self, values: tuple) -> tuple:
        """Add realistic noise to sensor values."""
        temp, pm25, co2, audio = values
        noise_scale = self.config.noise_level
        
        # Different noise characteristics for each sensor
        temp_noise = np.random.normal(0, 0.5 * noise_scale)
        pm25_noise = np.random.normal(0, 2.0 * noise_scale) 
        co2_noise = np.random.normal(0, 10.0 * noise_scale)
        audio_noise = np.random.normal(0, 3.0 * noise_scale)
        
        return (
            temp + temp_noise,
            pm25 + pm25_noise,
            co2 + co2_noise,
            audio + audio_noise
        )
    
    def add_temporal_correlation(self, values: tuple, previous_values: Optional[tuple] = None) -> tuple:
        """Add temporal correlation to make data more realistic."""
        if previous_values is None:
            return values
            
        # Apply smoothing factor to create temporal correlation
        smoothing = 0.8  # Higher values = more correlation
        temp, pm25, co2, audio = values
        prev_temp, prev_pm25, prev_co2, prev_audio = previous_values
        
        return (
            smoothing * prev_temp + (1 - smoothing) * temp,
            smoothing * prev_pm25 + (1 - smoothing) * pm25,
            smoothing * prev_co2 + (1 - smoothing) * co2,
            smoothing * prev_audio + (1 - smoothing) * audio
        )
    
    def generate_reading(self, sample_index: int, previous_reading: Optional[SensorReading] = None) -> SensorReading:
        """Generate a single sensor reading."""
        # Generate baseline values
        baseline_values = self.generate_baseline_values(sample_index)
        
        # Add temporal correlation
        if previous_reading:
            prev_values = (
                previous_reading.temperature,
                previous_reading.pm25,
                previous_reading.co2,
                previous_reading.audio_level
            )
            correlated_values = self.add_temporal_correlation(baseline_values, prev_values)
        else:
            correlated_values = baseline_values
        
        # Add realistic noise
        noisy_values = self.add_realistic_noise(correlated_values)
        
        # Create reading with current timestamp for real-time streaming
        # For batch generation, use calculated time; for streaming, use actual time
        if hasattr(self, '_streaming_mode') and self._streaming_mode:
            timestamp = datetime.now()
        else:
            timestamp = self.current_time + timedelta(seconds=sample_index / self.config.update_frequency)
        
        reading = SensorReading(
            timestamp=timestamp,
            temperature=noisy_values[0],
            pm25=noisy_values[1],
            co2=noisy_values[2],
            audio_level=noisy_values[3],
            location=self.config.location
        )
        
        # Validate and clamp values
        return self.limits.clamp_reading(reading)
    
    def generate_sequence(self) -> List[SensorReading]:
        """Generate complete sequence of sensor readings."""
        readings = []
        previous_reading = None
        
        for i in range(self.config.total_samples):
            reading = self.generate_reading(i, previous_reading)
            readings.append(reading)
            previous_reading = reading
            
        return readings
    
    def start_streaming(self) -> None:
        """Start real-time data streaming in a separate thread."""
        if self._running:
            return
            
        self._running = True
        self._streaming_mode = True  # Flag to use real-time timestamps
        self._thread = threading.Thread(target=self._stream_data, daemon=True)
        self._thread.start()
    
    def stop_streaming(self) -> None:
        """Stop real-time data streaming."""
        self._running = False
        self._streaming_mode = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def get_current_reading(self) -> Optional[SensorReading]:
        """Get the current sensor reading (thread-safe)."""
        with self._lock:
            return self._current_reading
    
    def _stream_data(self) -> None:
        """Internal method for streaming data in separate thread."""
        sample_index = 0
        previous_reading = None
        start_time = time.time()
        
        try:
            while self._running:
                current_time = time.time()
                expected_time = start_time + (sample_index / self.config.update_frequency)
                
                if current_time >= expected_time:
                    # Generate new reading
                    reading = self.generate_reading(sample_index, previous_reading)
                    
                    # Update current reading (thread-safe)
                    with self._lock:
                        self._current_reading = reading
                    
                    previous_reading = reading
                    sample_index += 1
                    
                    # Log progress for debugging (less frequent to avoid spam)
                    if sample_index % 20 == 0:  # Log every 20 samples
                        progress = (sample_index % self.config.total_samples) / self.config.total_samples * 100
                        print(f"Streaming sample {sample_index % self.config.total_samples}/{self.config.total_samples} - {progress:.1f}%")
                    
                    # Restart scenario when it reaches the end (loop continuously)
                    if sample_index >= self.config.total_samples:
                        print("Scenario completed, restarting...")
                        sample_index = 0  # Reset to beginning
                        start_time = time.time()  # Reset start time
                        previous_reading = None  # Reset previous reading
                
                # Sleep for a short time to avoid busy waiting
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error in stream_data: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
            print("Streaming thread stopped")