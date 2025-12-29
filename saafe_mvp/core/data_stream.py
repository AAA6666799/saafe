"""
Real-time data streaming pipeline for Saafe MVP.
Implements Task 3.2: Real-time data streaming with configurable update frequencies.
"""

import threading
import time
import queue
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import logging

from .data_models import SensorReading, ScenarioConfig
from .scenario_manager import ScenarioManager, ScenarioType

logger = logging.getLogger(__name__)


class DataStreamManager:
    """
    Manages real-time data streaming with thread-safe data access.
    Implements configurable update frequencies and data buffering.
    """
    
    def __init__(self, config: ScenarioConfig = None):
        self.config = config or ScenarioConfig()
        self.scenario_manager = ScenarioManager(self.config)
        
        # Thread-safe data storage
        self._current_reading = None
        self._reading_history = queue.Queue(maxsize=1000)  # Buffer last 1000 readings
        self._lock = threading.RLock()
        
        # Streaming control
        self._streaming = False
        self._stream_thread = None
        self._update_callbacks = []
        
        # Performance tracking
        self._stream_start_time = None
        self._total_readings = 0
        self._last_update_time = None
        
        logger.info("DataStreamManager initialized")
    
    def start_streaming(self, scenario_type: ScenarioType) -> bool:
        """
        Start real-time data streaming for the specified scenario.
        
        Args:
            scenario_type: Type of scenario to stream
            
        Returns:
            bool: True if streaming started successfully
        """
        with self._lock:
            if self._streaming:
                self.stop_streaming()
            
            # Start the scenario
            success = self.scenario_manager.start_scenario(scenario_type)
            if not success:
                logger.error(f"Failed to start scenario: {scenario_type}")
                return False
            
            # Start streaming thread
            self._streaming = True
            self._stream_start_time = time.time()
            self._total_readings = 0
            self._stream_thread = threading.Thread(
                target=self._stream_loop, 
                name=f"DataStream-{scenario_type.value}",
                daemon=True
            )
            self._stream_thread.start()
            
            logger.info(f"Started streaming for scenario: {scenario_type.value}")
            return True
    
    def stop_streaming(self):
        """Stop real-time data streaming."""
        with self._lock:
            if not self._streaming:
                return
            
            self._streaming = False
            
            # Stop scenario manager
            self.scenario_manager.stop_scenario()
            
            # Wait for thread to finish
            if self._stream_thread and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=2.0)
            
            logger.info("Stopped data streaming")
    
    def get_current_reading(self) -> Optional[SensorReading]:
        """
        Get the most recent sensor reading (thread-safe).
        
        Returns:
            SensorReading or None if no data available
        """
        with self._lock:
            return self._current_reading
    
    def get_reading_history(self, max_count: int = 100) -> list:
        """
        Get recent sensor readings from history buffer.
        
        Args:
            max_count: Maximum number of readings to return
            
        Returns:
            List of recent SensorReading objects
        """
        with self._lock:
            history = []
            temp_queue = queue.Queue()
            
            # Extract readings from queue
            while not self._reading_history.empty() and len(history) < max_count:
                try:
                    reading = self._reading_history.get_nowait()
                    history.append(reading)
                    temp_queue.put(reading)
                except queue.Empty:
                    break
            
            # Put readings back in queue
            while not temp_queue.empty():
                try:
                    self._reading_history.put_nowait(temp_queue.get_nowait())
                except queue.Full:
                    break
            
            return list(reversed(history))  # Most recent first
    
    def add_update_callback(self, callback: Callable[[SensorReading], None]):
        """
        Add a callback function to be called when new data arrives.
        
        Args:
            callback: Function to call with new SensorReading
        """
        with self._lock:
            self._update_callbacks.append(callback)
    
    def remove_update_callback(self, callback: Callable[[SensorReading], None]):
        """Remove an update callback."""
        with self._lock:
            if callback in self._update_callbacks:
                self._update_callbacks.remove(callback)
    
    def get_stream_status(self) -> Dict[str, Any]:
        """
        Get current streaming status and performance metrics.
        
        Returns:
            Dictionary with streaming status information
        """
        with self._lock:
            if not self._streaming or not self._stream_start_time:
                return {
                    'streaming': False,
                    'scenario': None,
                    'uptime': 0,
                    'total_readings': 0,
                    'readings_per_second': 0,
                    'last_update': None
                }
            
            uptime = time.time() - self._stream_start_time
            readings_per_second = self._total_readings / uptime if uptime > 0 else 0
            
            return {
                'streaming': True,
                'scenario': self.scenario_manager.get_current_scenario().value if self.scenario_manager.get_current_scenario() else None,
                'uptime': uptime,
                'total_readings': self._total_readings,
                'readings_per_second': readings_per_second,
                'last_update': self._last_update_time.isoformat() if self._last_update_time else None,
                'buffer_size': self._reading_history.qsize(),
                'update_frequency': self.config.update_frequency
            }
    
    def get_scenario_progress(self) -> Dict[str, Any]:
        """
        Get current scenario progress information.
        
        Returns:
            Dictionary with progress information
        """
        with self._lock:
            if not self._streaming or not self._stream_start_time:
                return {
                    'progress': 0.0,
                    'elapsed_time': 0,
                    'remaining_time': 0,
                    'phase': 'Not Started'
                }
            
            elapsed = time.time() - self._stream_start_time
            progress = min(elapsed / self.config.duration_seconds, 1.0)
            remaining = max(0, self.config.duration_seconds - elapsed)
            
            # Determine phase based on scenario type and progress
            current_scenario = self.scenario_manager.get_current_scenario()
            if current_scenario == ScenarioType.FIRE:
                phase = self._get_fire_phase(progress)
            elif current_scenario == ScenarioType.COOKING:
                phase = self._get_cooking_phase(progress)
            else:
                phase = self._get_normal_phase(progress)
            
            return {
                'progress': progress,
                'elapsed_time': elapsed,
                'remaining_time': remaining,
                'phase': phase,
                'scenario': current_scenario.value if current_scenario else None
            }
    
    def _stream_loop(self):
        """Main streaming loop running in separate thread."""
        logger.info("Data streaming loop started")
        
        try:
            while self._streaming:
                # Get current reading from scenario manager
                reading = self.scenario_manager.get_current_data()
                
                if reading:
                    # Update current reading (thread-safe)
                    with self._lock:
                        self._current_reading = reading
                        self._last_update_time = datetime.now()
                        self._total_readings += 1
                        
                        # Add to history buffer
                        try:
                            self._reading_history.put_nowait(reading)
                        except queue.Full:
                            # Remove oldest reading if buffer is full
                            try:
                                self._reading_history.get_nowait()
                                self._reading_history.put_nowait(reading)
                            except queue.Empty:
                                pass
                        
                        # Call update callbacks
                        for callback in self._update_callbacks:
                            try:
                                callback(reading)
                            except Exception as e:
                                logger.error(f"Error in update callback: {e}")
                
                # Sleep based on update frequency
                sleep_time = 1.0 / self.config.update_frequency
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("Data streaming loop ended")
    
    def _get_fire_phase(self, progress: float) -> str:
        """Get fire scenario phase description."""
        if progress < 0.20:
            return "🔍 Incipient Phase"
        elif progress < 0.35:
            return "📈 Growth Phase"
        elif progress < 0.40:
            return "⚡ Flashover Phase"
        elif progress < 0.85:
            return "🔥 Fully Developed"
        else:
            return "📉 Decay Phase"
    
    def _get_cooking_phase(self, progress: float) -> str:
        """Get cooking scenario phase description."""
        if progress < 0.3:
            return "🍳 Heating Up"
        elif progress < 0.7:
            return "🔥 Active Cooking"
        else:
            return "🔄 Cooling Down"
    
    def _get_normal_phase(self, progress: float) -> str:
        """Get normal scenario phase description."""
        return "🏠 Stable Environment"
    
    def validate_reading(self, reading: SensorReading) -> bool:
        """
        Validate sensor reading for range and sanity checks.
        
        Args:
            reading: SensorReading to validate
            
        Returns:
            bool: True if reading is valid
        """
        if not reading:
            return False
        
        # Check for reasonable ranges
        if not (0 <= reading.temperature <= 150):
            return False
        if not (0 <= reading.pm25 <= 1000):
            return False
        if not (0 <= reading.co2 <= 10000):
            return False
        if not (0 <= reading.audio_level <= 150):
            return False
        
        # Check timestamp is recent
        time_diff = abs((datetime.now() - reading.timestamp).total_seconds())
        if time_diff > 60:  # More than 1 minute old
            return False
        
        return True
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_streaming()


# Global data stream manager instance
_global_stream_manager = None


def get_data_stream_manager() -> DataStreamManager:
    """
    Get the global data stream manager instance.
    
    Returns:
        DataStreamManager: Global instance
    """
    global _global_stream_manager
    if _global_stream_manager is None:
        _global_stream_manager = DataStreamManager()
    return _global_stream_manager


def reset_data_stream_manager():
    """Reset the global data stream manager (for testing)."""
    global _global_stream_manager
    if _global_stream_manager:
        _global_stream_manager.stop_streaming()
    _global_stream_manager = None