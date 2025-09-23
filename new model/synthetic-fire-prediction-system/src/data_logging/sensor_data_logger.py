"""
Sensor Data Logger for FLIR+SCD41 Sensors.

This module provides comprehensive data logging and storage capabilities for
FLIR Lepton 3.5 thermal cameras and SCD41 CO₂ sensors.
"""

import logging
import json
import csv
import sqlite3
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import threading
from collections import deque
import gzip
import os

# Try to import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    boto3 = None


class DataLoggerConfig:
    """
    Configuration class for data logger.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data logger configuration.
        
        Args:
            config: Configuration dictionary
        """
        config = config or {}
        
        # Storage settings
        self.storage_path = Path(config.get('storage_path', './sensor_data'))
        self.max_file_size_mb = config.get('max_file_size_mb', 100)
        self.rotation_interval_hours = config.get('rotation_interval_hours', 24)
        self.compress_files = config.get('compress_files', True)
        
        # Format settings
        self.default_format = config.get('default_format', 'json')
        self.include_metadata = config.get('include_metadata', True)
        self.flatten_data = config.get('flatten_data', False)
        
        # AWS settings
        self.aws_enabled = config.get('aws_enabled', False)
        self.aws_bucket = config.get('aws_bucket', '')
        self.aws_region = config.get('aws_region', 'us-east-1')
        self.aws_credentials = config.get('aws_credentials', {})
        
        # Buffer settings
        self.buffer_size = config.get('buffer_size', 1000)
        self.flush_interval_seconds = config.get('flush_interval_seconds', 30)
        
        # Filter settings
        self.log_only_valid_data = config.get('log_only_valid_data', False)
        self.min_quality_level = config.get('min_quality_level', 'acceptable')  # excellent, good, acceptable, poor, unusable


class SensorDataLogger:
    """
    Comprehensive sensor data logger for FLIR+SCD41 sensors.
    
    This class provides multiple storage backends including local file storage,
    SQLite database, and optional AWS S3 integration.
    """
    
    def __init__(self, config: Optional[DataLoggerConfig] = None):
        """
        Initialize the sensor data logger.
        
        Args:
            config: Data logger configuration
        """
        self.config = config or DataLoggerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Create storage directory if it doesn't exist
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage backends
        self.file_logger = FileDataLogger(self.config)
        self.database_logger = None
        self.aws_logger = None
        
        # Try to initialize optional loggers
        try:
            self.database_logger = DatabaseDataLogger(self.config)
        except Exception as e:
            self.logger.warning(f"Failed to initialize database logger: {str(e)}")
        
        if self.config.aws_enabled and AWS_AVAILABLE:
            try:
                self.aws_logger = AWSDataLogger(self.config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize AWS logger: {str(e)}")
        
        # Data buffer for batch processing
        self.data_buffer = deque(maxlen=self.config.buffer_size)
        self.buffer_lock = threading.Lock()
        self.last_flush_time = datetime.now()
        
        # Background flush thread
        self.flush_thread = None
        self.stop_flush_thread = threading.Event()
        self._start_flush_thread()
        
        self.logger.info("Initialized Sensor Data Logger")
        self.logger.info(f"Storage path: {self.config.storage_path}")
        self.logger.info(f"AWS enabled: {self.config.aws_enabled}")
    
    def log_sensor_data(self, data: Dict[str, Any]) -> bool:
        """
        Log sensor data to all enabled storage backends.
        
        Args:
            data: Sensor data to log
            
        Returns:
            True if data was logged successfully to at least one backend
        """
        success = False
        
        try:
            # Add to buffer
            with self.buffer_lock:
                self.data_buffer.append(data)
            
            # Log to file
            if self.file_logger.log_data(data):
                success = True
            
            # Log to database
            if self.database_logger and self.database_logger.log_data(data):
                success = True
            
            # Log to AWS
            if self.aws_logger and self.aws_logger.log_data(data):
                success = True
                
        except Exception as e:
            self.logger.error(f"Error logging sensor data: {str(e)}")
        
        return success
    
    def flush_buffer(self) -> None:
        """
        Flush the data buffer to all storage backends.
        """
        try:
            with self.buffer_lock:
                if not self.data_buffer:
                    return
                
                data_batch = list(self.data_buffer)
                self.data_buffer.clear()
            
            # Flush to file
            self.file_logger.flush_batch(data_batch)
            
            # Flush to database
            if self.database_logger:
                self.database_logger.flush_batch(data_batch)
            
            # Flush to AWS
            if self.aws_logger:
                self.aws_logger.flush_batch(data_batch)
                
            self.last_flush_time = datetime.now()
            self.logger.debug(f"Flushed {len(data_batch)} data points to storage")
            
        except Exception as e:
            self.logger.error(f"Error flushing data buffer: {str(e)}")
    
    def _start_flush_thread(self) -> None:
        """Start the background flush thread."""
        self.stop_flush_thread.clear()
        self.flush_thread = threading.Thread(target=self._flush_loop)
        self.flush_thread.daemon = True
        self.flush_thread.start()
        self.logger.debug("Started background flush thread")
    
    def _flush_loop(self) -> None:
        """Background loop for periodic buffer flushing."""
        while not self.stop_flush_thread.is_set():
            try:
                # Check if it's time to flush
                time_since_last_flush = (datetime.now() - self.last_flush_time).total_seconds()
                if time_since_last_flush >= self.config.flush_interval_seconds:
                    self.flush_buffer()
                
                # Sleep for a short time
                self.stop_flush_thread.wait(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in flush loop: {str(e)}")
                self.stop_flush_thread.wait(5.0)  # Wait longer on error
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary containing storage statistics
        """
        stats = {
            'buffer_size': len(self.data_buffer),
            'last_flush_time': self.last_flush_time.isoformat(),
            'file_logger': self.file_logger.get_stats(),
            'database_logger': self.database_logger.get_stats() if self.database_logger else {},
            'aws_logger': self.aws_logger.get_stats() if self.aws_logger else {}
        }
        return stats
    
    def shutdown(self) -> None:
        """
        Shutdown the data logger and flush any remaining data.
        """
        self.logger.info("Shutting down data logger...")
        
        # Stop flush thread
        self.stop_flush_thread.set()
        if self.flush_thread and self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5.0)
        
        # Final flush
        self.flush_buffer()
        
        # Shutdown loggers
        self.file_logger.shutdown()
        if self.database_logger:
            self.database_logger.shutdown()
        if self.aws_logger:
            self.aws_logger.shutdown()
        
        self.logger.info("Data logger shutdown complete")


class FileDataLogger:
    """
    File-based data logger implementation.
    """
    
    def __init__(self, config: DataLoggerConfig):
        """
        Initialize the file data logger.
        
        Args:
            config: Data logger configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__ + ".FileDataLogger")
        
        # Current file tracking
        self.current_file = None
        self.current_file_start_time = None
        self.current_file_size = 0
        
        # File rotation tracking
        self.file_sequence = 0
        self._initialize_current_file()
        
        self.logger.debug("Initialized File Data Logger")
    
    def _initialize_current_file(self) -> None:
        """Initialize the current log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_sequence += 1
        
        if self.config.default_format == 'json':
            filename = f"sensor_data_{timestamp}_{self.file_sequence:04d}.json"
            if self.config.compress_files:
                filename += ".gz"
        elif self.config.default_format == 'csv':
            filename = f"sensor_data_{timestamp}_{self.file_sequence:04d}.csv"
            if self.config.compress_files:
                filename += ".gz"
        else:
            filename = f"sensor_data_{timestamp}_{self.file_sequence:04d}.log"
            if self.config.compress_files:
                filename += ".gz"
        
        self.current_file = self.config.storage_path / filename
        self.current_file_start_time = datetime.now()
        self.current_file_size = 0
        
        self.logger.info(f"Initialized log file: {self.current_file}")
    
    def _should_rotate_file(self) -> bool:
        """
        Check if the current file should be rotated.
        
        Returns:
            True if file should be rotated
        """
        # Check file size
        if self.current_file_size >= self.config.max_file_size_mb * 1024 * 1024:
            return True
        
        # Check time interval
        if self.current_file_start_time:
            time_since_start = (datetime.now() - self.current_file_start_time).total_seconds()
            if time_since_start >= self.config.rotation_interval_hours * 3600:
                return True
        
        return False
    
    def _rotate_file_if_needed(self) -> None:
        """Rotate the current log file if needed."""
        if self._should_rotate_file():
            self.logger.info(f"Rotating log file: {self.current_file}")
            self._initialize_current_file()
    
    def _write_json_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """
        Write data in JSON format.
        
        Args:
            data: Data to write
        """
        if self.config.compress_files:
            with gzip.open(self.current_file, 'at', encoding='utf-8') as f:
                if isinstance(data, list):
                    for item in data:
                        json.dump(item, f)
                        f.write('\n')
                else:
                    json.dump(data, f)
                    f.write('\n')
        else:
            with open(self.current_file, 'a', encoding='utf-8') as f:
                if isinstance(data, list):
                    for item in data:
                        json.dump(item, f)
                        f.write('\n')
                else:
                    json.dump(data, f)
                    f.write('\n')
        
        # Update file size estimate
        self.current_file_size += len(json.dumps(data)) if isinstance(data, dict) else sum(len(json.dumps(item)) for item in data)
    
    def _write_csv_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """
        Write data in CSV format.
        
        Args:
            data: Data to write
        """
        # Convert to flat format if needed
        if self.config.flatten_data:
            if isinstance(data, list):
                flat_data = [self._flatten_dict(item) for item in data]
            else:
                flat_data = [self._flatten_dict(data)]
        else:
            flat_data = data if isinstance(data, list) else [data]
        
        # Get fieldnames
        if flat_data:
            fieldnames = set()
            for item in flat_data:
                fieldnames.update(item.keys())
            fieldnames = sorted(list(fieldnames))
        else:
            fieldnames = []
        
        # Write to file
        file_mode = 'at' if self.current_file.exists() else 'wt'
        
        if self.config.compress_files:
            with gzip.open(self.current_file, file_mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if file_mode == 'wt':  # Write header for new file
                    writer.writeheader()
                writer.writerows(flat_data)
        else:
            with open(self.current_file, file_mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if file_mode == 'wt':  # Write header for new file
                    writer.writeheader()
                writer.writerows(flat_data)
        
        # Update file size estimate
        self.current_file_size += sum(len(str(v)) for item in flat_data for v in item.values())
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def log_data(self, data: Dict[str, Any]) -> bool:
        """
        Log data to file.
        
        Args:
            data: Data to log
            
        Returns:
            True if successful
        """
        try:
            # Rotate file if needed
            self._rotate_file_if_needed()
            
            # Write data
            if self.config.default_format == 'json':
                self._write_json_data(data)
            elif self.config.default_format == 'csv':
                self._write_csv_data(data)
            else:
                # Default to JSON
                self._write_json_data(data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing to file: {str(e)}")
            return False
    
    def flush_batch(self, data_batch: List[Dict[str, Any]]) -> None:
        """
        Flush a batch of data to file.
        
        Args:
            data_batch: List of data points to flush
        """
        if not data_batch:
            return
        
        try:
            # Rotate file if needed
            self._rotate_file_if_needed()
            
            # Write batch
            if self.config.default_format == 'json':
                self._write_json_data(data_batch)
            elif self.config.default_format == 'csv':
                self._write_csv_data(data_batch)
            else:
                # Default to JSON
                self._write_json_data(data_batch)
                
        except Exception as e:
            self.logger.error(f"Error flushing batch to file: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get file logger statistics.
        
        Returns:
            Dictionary containing file logger statistics
        """
        return {
            'current_file': str(self.current_file) if self.current_file else None,
            'current_file_size': self.current_file_size,
            'file_sequence': self.file_sequence
        }
    
    def shutdown(self) -> None:
        """Shutdown the file logger."""
        self.logger.info("File data logger shutdown complete")


class DatabaseDataLogger:
    """
    SQLite database-based data logger implementation.
    """
    
    def __init__(self, config: DataLoggerConfig):
        """
        Initialize the database data logger.
        
        Args:
            config: Data logger configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__ + ".DatabaseDataLogger")
        
        # Database setup
        self.db_path = self.config.storage_path / "sensor_data.db"
        self.connection = None
        self._initialize_database()
        
        self.logger.debug("Initialized Database Data Logger")
    
    def _initialize_database(self) -> None:
        """Initialize the SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # Create tables
            cursor = self.connection.cursor()
            
            # Main sensor data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    sensor_type TEXT NOT NULL,
                    sensor_id TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    quality_level TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Indexes for performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON sensor_data(timestamp)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_sensor_type ON sensor_data(sensor_type)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_sensor_id ON sensor_data(sensor_id)
            ''')
            
            self.connection.commit()
            self.logger.info(f"Initialized database: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def log_data(self, data: Dict[str, Any]) -> bool:
        """
        Log data to database.
        
        Args:
            data: Data to log
            
        Returns:
            True if successful
        """
        try:
            if not self.connection:
                return False
            
            cursor = self.connection.cursor()
            timestamp = datetime.now().isoformat()
            
            # Log FLIR data
            for sensor_id, sensor_data in data.get('flir', {}).items():
                cursor.execute('''
                    INSERT INTO sensor_data 
                    (timestamp, sensor_type, sensor_id, data_json, quality_level, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    sensor_data.get('timestamp', timestamp),
                    'flir',
                    sensor_id,
                    json.dumps(sensor_data),
                    sensor_data.get('quality_level', 'unknown'),
                    timestamp
                ))
            
            # Log SCD41 data
            for sensor_id, sensor_data in data.get('scd41', {}).items():
                cursor.execute('''
                    INSERT INTO sensor_data 
                    (timestamp, sensor_type, sensor_id, data_json, quality_level, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    sensor_data.get('timestamp', timestamp),
                    'scd41',
                    sensor_id,
                    json.dumps(sensor_data),
                    sensor_data.get('quality_level', 'unknown'),
                    timestamp
                ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging to database: {str(e)}")
            return False
    
    def flush_batch(self, data_batch: List[Dict[str, Any]]) -> None:
        """
        Flush a batch of data to database.
        
        Args:
            data_batch: List of data points to flush
        """
        if not data_batch or not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            timestamp = datetime.now().isoformat()
            
            for data in data_batch:
                # Log FLIR data
                for sensor_id, sensor_data in data.get('flir', {}).items():
                    cursor.execute('''
                        INSERT INTO sensor_data 
                        (timestamp, sensor_type, sensor_id, data_json, quality_level, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        sensor_data.get('timestamp', timestamp),
                        'flir',
                        sensor_id,
                        json.dumps(sensor_data),
                        sensor_data.get('quality_level', 'unknown'),
                        timestamp
                    ))
                
                # Log SCD41 data
                for sensor_id, sensor_data in data.get('scd41', {}).items():
                    cursor.execute('''
                        INSERT INTO sensor_data 
                        (timestamp, sensor_type, sensor_id, data_json, quality_level, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        sensor_data.get('timestamp', timestamp),
                        'scd41',
                        sensor_id,
                        json.dumps(sensor_data),
                        sensor_data.get('quality_level', 'unknown'),
                        timestamp
                    ))
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Error flushing batch to database: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database logger statistics.
        
        Returns:
            Dictionary containing database logger statistics
        """
        try:
            if not self.connection:
                return {}
            
            cursor = self.connection.cursor()
            cursor.execute('SELECT COUNT(*) FROM sensor_data')
            count = cursor.fetchone()[0]
            
            return {
                'db_path': str(self.db_path),
                'record_count': count
            }
        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {}
    
    def shutdown(self) -> None:
        """Shutdown the database logger."""
        if self.connection:
            self.connection.close()
        self.logger.info("Database data logger shutdown complete")


class AWSDataLogger:
    """
    AWS S3-based data logger implementation.
    """
    
    def __init__(self, config: DataLoggerConfig):
        """
        Initialize the AWS data logger.
        
        Args:
            config: Data logger configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__ + ".AWSDataLogger")
        
        # AWS setup
        if not AWS_AVAILABLE:
            raise RuntimeError("AWS SDK (boto3) not available")
        
        # Initialize S3 client
        if self.config.aws_credentials:
            self.s3_client = boto3.client(
                's3',
                region_name=self.config.aws_region,
                aws_access_key_id=self.config.aws_credentials.get('access_key_id'),
                aws_secret_access_key=self.config.aws_credentials.get('secret_access_key')
            )
        else:
            # Use default credentials
            self.s3_client = boto3.client('s3', region_name=self.config.aws_region)
        
        # Buffer for batch uploads
        self.upload_buffer = []
        self.upload_buffer_size = 100  # Upload in batches of 100
        
        self.logger.debug("Initialized AWS Data Logger")
    
    def log_data(self, data: Dict[str, Any]) -> bool:
        """
        Log data to AWS S3.
        
        Args:
            data: Data to log
            
        Returns:
            True if successful
        """
        try:
            # Add to upload buffer
            self.upload_buffer.append(data)
            
            # Upload if buffer is full
            if len(self.upload_buffer) >= self.upload_buffer_size:
                self._upload_batch()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging to AWS: {str(e)}")
            return False
    
    def _upload_batch(self) -> None:
        """Upload buffered data to S3."""
        if not self.upload_buffer:
            return
        
        try:
            # Create batch data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_id = f"batch_{timestamp}_{len(self.upload_buffer)}"
            
            # Convert to JSON
            json_data = json.dumps(self.upload_buffer, indent=2)
            
            # Upload to S3
            key = f"sensor_data/{datetime.now().strftime('%Y/%m/%d')}/{batch_id}.json"
            self.s3_client.put_object(
                Bucket=self.config.aws_bucket,
                Key=key,
                Body=json_data,
                ContentType='application/json'
            )
            
            self.logger.info(f"Uploaded batch to S3: {key}")
            self.upload_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Error uploading batch to S3: {str(e)}")
    
    def flush_batch(self, data_batch: List[Dict[str, Any]]) -> None:
        """
        Flush a batch of data to AWS S3.
        
        Args:
            data_batch: List of data points to flush
        """
        if not data_batch:
            return
        
        try:
            # Convert to JSON
            json_data = json.dumps(data_batch, indent=2)
            
            # Upload to S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_id = f"batch_{timestamp}_{len(data_batch)}"
            key = f"sensor_data/{datetime.now().strftime('%Y/%m/%d')}/{batch_id}.json"
            
            self.s3_client.put_object(
                Bucket=self.config.aws_bucket,
                Key=key,
                Body=json_data,
                ContentType='application/json'
            )
            
            self.logger.info(f"Flushed batch to S3: {key}")
            
        except Exception as e:
            self.logger.error(f"Error flushing batch to S3: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get AWS logger statistics.
        
        Returns:
            Dictionary containing AWS logger statistics
        """
        return {
            'aws_bucket': self.config.aws_bucket,
            'aws_region': self.config.aws_region,
            'buffer_size': len(self.upload_buffer)
        }
    
    def shutdown(self) -> None:
        """Shutdown the AWS logger."""
        # Upload any remaining data
        if self.upload_buffer:
            self._upload_batch()
        self.logger.info("AWS data logger shutdown complete")


# Convenience function for creating sensor data logger
def create_sensor_data_logger(config: Optional[Dict[str, Any]] = None) -> SensorDataLogger:
    """
    Create a sensor data logger with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured SensorDataLogger instance
    """
    default_config = {
        'storage_path': './sensor_data',
        'max_file_size_mb': 100,
        'rotation_interval_hours': 24,
        'compress_files': True,
        'default_format': 'json',
        'include_metadata': True,
        'flatten_data': False,
        'aws_enabled': False,
        'aws_bucket': '',
        'aws_region': 'us-east-1',
        'buffer_size': 1000,
        'flush_interval_seconds': 30,
        'log_only_valid_data': False,
        'min_quality_level': 'acceptable'
    }
    
    if config:
        # Merge configs
        for key, value in config.items():
            default_config[key] = value
    
    return SensorDataLogger(DataLoggerConfig(default_config))