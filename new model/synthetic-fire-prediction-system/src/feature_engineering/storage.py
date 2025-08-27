"""
Feature Storage System for the synthetic fire prediction system.

This module provides functionality for storing and retrieving extracted features,
implementing caching mechanisms, and handling feature compression.
"""

from typing import Dict, Any, List, Optional, Union, BinaryIO, Tuple
import os
import json
import logging
import pickle
import gzip
import shutil
import hashlib
import time
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import tempfile

# AWS integration (optional)
try:
    from ..aws.s3.service import S3ServiceImpl
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    S3ServiceImpl = None


class FeatureStorageSystem:
    """
    System for storing and retrieving extracted features.
    
    This class stores extracted features efficiently, provides fast retrieval
    of features for model training, implements caching mechanisms for frequently
    used features, and handles feature compression and decompression.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature storage system.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_lock = threading.RLock()
        
        # Initialize AWS S3 service if AWS integration is enabled
        self.s3_service = None
        if self.config.get('aws_integration', False):
            self.s3_service = S3ServiceImpl(self.config.get('aws_config', {}))
        
        # Validate configuration
        self._validate_config()
        
        # Create storage directories
        self._create_storage_directories()
    
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Set default values if not provided
        if 'storage_dir' not in self.config:
            self.config['storage_dir'] = 'feature_storage'
        
        if 'cache_size_mb' not in self.config:
            self.config['cache_size_mb'] = 1024  # 1 GB default cache size
        
        if 'compression_level' not in self.config:
            self.config['compression_level'] = 6  # Default gzip compression level
        
        if 'use_compression' not in self.config:
            self.config['use_compression'] = True
    
    def _create_storage_directories(self) -> None:
        """
        Create necessary directories for feature storage.
        """
        # Create main storage directory
        os.makedirs(self.config['storage_dir'], exist_ok=True)
        
        # Create subdirectories for different feature types
        feature_types = ['thermal', 'gas', 'environmental', 'fused']
        for feature_type in feature_types:
            os.makedirs(os.path.join(self.config['storage_dir'], feature_type), exist_ok=True)
        
        # Create cache directory
        cache_dir = os.path.join(self.config['storage_dir'], 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        self.logger.info(f"Created feature storage directories in {self.config['storage_dir']}")
    
    def store_features(self, 
                      features: Dict[str, Any], 
                      feature_type: str,
                      metadata: Optional[Dict[str, Any]] = None,
                      feature_id: Optional[str] = None) -> str:
        """
        Store features in the storage system.
        
        Args:
            features: Dictionary containing features to store
            feature_type: Type of features (thermal, gas, environmental, fused)
            metadata: Optional metadata about the features
            feature_id: Optional feature identifier (generated if not provided)
            
        Returns:
            Feature ID
        """
        # Generate feature ID if not provided
        if feature_id is None:
            feature_id = self._generate_feature_id(features, feature_type)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'feature_id': feature_id,
            'feature_type': feature_type,
            'storage_time': datetime.now().isoformat(),
            'feature_count': self._count_features(features)
        })
        
        # Determine storage path
        storage_path = self._get_storage_path(feature_id, feature_type)
        
        # Store features
        self._write_features_to_storage(features, storage_path, metadata)
        
        # Update cache
        self._update_cache(feature_id, features)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            self._upload_to_s3(feature_id, feature_type, storage_path)
        
        self.logger.info(f"Stored {feature_type} features with ID: {feature_id}")
        return feature_id
    
    def retrieve_features(self, 
                         feature_id: str, 
                         feature_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve features from the storage system.
        
        Args:
            feature_id: Feature identifier
            feature_type: Optional feature type (for faster retrieval)
            
        Returns:
            Dictionary containing the features
        """
        # Check cache first
        with self.cache_lock:
            if feature_id in self.cache:
                self.cache_hits += 1
                self.logger.debug(f"Cache hit for feature ID: {feature_id}")
                return self.cache[feature_id]
        
        # Determine storage path
        storage_path = self._get_storage_path(feature_id, feature_type)
        
        # If path doesn't exist and feature_type wasn't provided, try to find it
        if not os.path.exists(storage_path) and feature_type is None:
            storage_path = self._find_feature_path(feature_id)
            
            if storage_path is None:
                # Try to download from S3 if AWS integration is enabled
                if self.s3_service is not None:
                    storage_path = self._download_from_s3(feature_id)
                
                if storage_path is None:
                    raise ValueError(f"Features not found for ID: {feature_id}")
        
        # Read features from storage
        features = self._read_features_from_storage(storage_path)
        
        # Update cache
        self._update_cache(feature_id, features)
        
        self.cache_misses += 1
        self.logger.debug(f"Retrieved features with ID: {feature_id}")
        return features
    
    def delete_features(self, 
                       feature_id: str, 
                       feature_type: Optional[str] = None) -> bool:
        """
        Delete features from the storage system.
        
        Args:
            feature_id: Feature identifier
            feature_type: Optional feature type
            
        Returns:
            True if features were deleted, False otherwise
        """
        # Determine storage path
        storage_path = self._get_storage_path(feature_id, feature_type)
        
        # If path doesn't exist and feature_type wasn't provided, try to find it
        if not os.path.exists(storage_path) and feature_type is None:
            storage_path = self._find_feature_path(feature_id)
            
            if storage_path is None:
                self.logger.warning(f"Features not found for deletion: {feature_id}")
                return False
        
        # Delete from storage
        try:
            if os.path.exists(storage_path):
                if os.path.isdir(storage_path):
                    shutil.rmtree(storage_path)
                else:
                    os.remove(storage_path)
            
            # Delete metadata file if it exists
            metadata_path = f"{storage_path}.meta.json"
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Remove from cache
            with self.cache_lock:
                if feature_id in self.cache:
                    del self.cache[feature_id]
            
            # Delete from S3 if AWS integration is enabled
            if self.s3_service is not None:
                self._delete_from_s3(feature_id, feature_type)
            
            self.logger.info(f"Deleted features with ID: {feature_id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error deleting features {feature_id}: {str(e)}")
            return False
    
    def list_features(self, feature_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available features in the storage system.
        
        Args:
            feature_type: Optional feature type to filter by
            
        Returns:
            List of feature metadata dictionaries
        """
        features_list = []
        
        # Determine directory to search
        if feature_type is not None:
            search_dirs = [os.path.join(self.config['storage_dir'], feature_type)]
        else:
            # Search all feature type directories
            search_dirs = [
                os.path.join(self.config['storage_dir'], ft)
                for ft in ['thermal', 'gas', 'environmental', 'fused']
            ]
        
        # Find all metadata files
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            
            for root, _, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.meta.json'):
                        metadata_path = os.path.join(root, file)
                        
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                features_list.append(metadata)
                        except Exception as e:
                            self.logger.warning(f"Error reading metadata file {metadata_path}: {str(e)}")
        
        return features_list
    
    def get_feature_metadata(self, 
                           feature_id: str, 
                           feature_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for specific features.
        
        Args:
            feature_id: Feature identifier
            feature_type: Optional feature type
            
        Returns:
            Dictionary containing feature metadata
        """
        # Determine storage path
        storage_path = self._get_storage_path(feature_id, feature_type)
        
        # If path doesn't exist and feature_type wasn't provided, try to find it
        if not os.path.exists(storage_path) and feature_type is None:
            storage_path = self._find_feature_path(feature_id)
            
            if storage_path is None:
                raise ValueError(f"Features not found for ID: {feature_id}")
        
        # Read metadata
        metadata_path = f"{storage_path}.meta.json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading metadata for {feature_id}: {str(e)}")
                return {'error': str(e)}
        else:
            return {'error': f"Metadata not found for feature ID: {feature_id}"}
    
    def clear_cache(self) -> None:
        """
        Clear the feature cache.
        """
        with self.cache_lock:
            self.cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
        
        self.logger.info("Cleared feature cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.cache_lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            
            return {
                'cache_size': len(self.cache),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': hit_rate,
                'memory_usage_mb': self._estimate_cache_size_mb()
            }
    
    def _generate_feature_id(self, features: Dict[str, Any], feature_type: str) -> str:
        """
        Generate a unique identifier for features.
        
        Args:
            features: Dictionary containing features
            feature_type: Type of features
            
        Returns:
            Feature identifier
        """
        # Create a hash of the features
        feature_hash = hashlib.md5()
        
        # Add feature type to the hash
        feature_hash.update(feature_type.encode('utf-8'))
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().isoformat()
        feature_hash.update(timestamp.encode('utf-8'))
        
        # Add a sample of feature values to the hash
        if isinstance(features, dict):
            for key, value in list(features.items())[:10]:  # Use first 10 items for hash
                feature_hash.update(str(key).encode('utf-8'))
                
                if isinstance(value, (np.ndarray, pd.DataFrame)):
                    # For large data structures, use a sample
                    sample = str(np.array(value).flatten()[:100])
                    feature_hash.update(sample.encode('utf-8'))
                else:
                    feature_hash.update(str(value).encode('utf-8'))
        
        # Generate ID
        return f"{feature_type}_{feature_hash.hexdigest()}"
    
    def _count_features(self, features: Dict[str, Any]) -> int:
        """
        Count the number of individual features in a feature dictionary.
        
        Args:
            features: Dictionary containing features
            
        Returns:
            Number of individual features
        """
        count = 0
        
        if isinstance(features, dict):
            for key, value in features.items():
                if isinstance(value, dict):
                    count += self._count_features(value)
                elif isinstance(value, (list, tuple)):
                    count += len(value)
                elif isinstance(value, (np.ndarray, pd.DataFrame)):
                    count += value.size
                else:
                    count += 1
        
        return count
    
    def _get_storage_path(self, feature_id: str, feature_type: Optional[str]) -> str:
        """
        Get the storage path for features.
        
        Args:
            feature_id: Feature identifier
            feature_type: Optional feature type
            
        Returns:
            Storage path
        """
        if feature_type is None:
            # Try to extract feature type from ID
            parts = feature_id.split('_', 1)
            if len(parts) > 1 and parts[0] in ['thermal', 'gas', 'environmental', 'fused']:
                feature_type = parts[0]
            else:
                feature_type = 'unknown'
        
        # Create path with subdirectories based on feature ID to avoid too many files in one directory
        if len(feature_id) >= 4:
            subdir = feature_id[:2]
            return os.path.join(self.config['storage_dir'], feature_type, subdir, feature_id)
        else:
            return os.path.join(self.config['storage_dir'], feature_type, feature_id)
    
    def _find_feature_path(self, feature_id: str) -> Optional[str]:
        """
        Find the storage path for features when feature type is unknown.
        
        Args:
            feature_id: Feature identifier
            
        Returns:
            Storage path or None if not found
        """
        # Try each feature type directory
        for feature_type in ['thermal', 'gas', 'environmental', 'fused']:
            # Try with subdirectory
            if len(feature_id) >= 4:
                subdir = feature_id[:2]
                path = os.path.join(self.config['storage_dir'], feature_type, subdir, feature_id)
                if os.path.exists(path):
                    return path
            
            # Try without subdirectory
            path = os.path.join(self.config['storage_dir'], feature_type, feature_id)
            if os.path.exists(path):
                return path
        
        return None
    
    def _write_features_to_storage(self, 
                                 features: Dict[str, Any], 
                                 storage_path: str,
                                 metadata: Dict[str, Any]) -> None:
        """
        Write features to storage.
        
        Args:
            features: Dictionary containing features
            storage_path: Path to store the features
            metadata: Metadata about the features
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        
        # Determine if we should use compression
        use_compression = self.config.get('use_compression', True)
        
        # Write features
        if use_compression:
            # Use gzip compression
            with gzip.open(storage_path, 'wb', compresslevel=self.config.get('compression_level', 6)) as f:
                pickle.dump(features, f)
        else:
            # Write without compression
            with open(storage_path, 'wb') as f:
                pickle.dump(features, f)
        
        # Write metadata
        metadata_path = f"{storage_path}.meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _read_features_from_storage(self, storage_path: str) -> Dict[str, Any]:
        """
        Read features from storage.
        
        Args:
            storage_path: Path to read the features from
            
        Returns:
            Dictionary containing the features
        """
        try:
            # Try to read as gzip compressed file first
            try:
                with gzip.open(storage_path, 'rb') as f:
                    return pickle.load(f)
            except:
                # If that fails, try reading as uncompressed file
                with open(storage_path, 'rb') as f:
                    return pickle.load(f)
        
        except Exception as e:
            self.logger.error(f"Error reading features from {storage_path}: {str(e)}")
            raise
    
    def _update_cache(self, feature_id: str, features: Dict[str, Any]) -> None:
        """
        Update the feature cache.
        
        Args:
            feature_id: Feature identifier
            features: Dictionary containing features
        """
        with self.cache_lock:
            # Check if we need to evict items from cache
            cache_size_mb = self._estimate_cache_size_mb()
            max_cache_size_mb = self.config.get('cache_size_mb', 1024)
            
            if cache_size_mb > max_cache_size_mb:
                self._evict_from_cache(max_cache_size_mb * 0.8)  # Evict to 80% of max size
            
            # Add to cache
            self.cache[feature_id] = features
    
    def _estimate_cache_size_mb(self) -> float:
        """
        Estimate the current cache size in MB.
        
        Returns:
            Estimated cache size in MB
        """
        # This is a rough estimate and may not be accurate for all types of objects
        size_bytes = 0
        
        for feature_id, features in self.cache.items():
            # Add size of feature ID
            size_bytes += len(feature_id) * 2  # Assume 2 bytes per character
            
            # Add size of features
            if isinstance(features, dict):
                # Serialize to bytes to get a better estimate
                size_bytes += len(pickle.dumps(features))
            else:
                # Fallback to a very rough estimate
                size_bytes += sys.getsizeof(features)
        
        return size_bytes / (1024 * 1024)  # Convert to MB
    
    def _evict_from_cache(self, target_size_mb: float) -> None:
        """
        Evict items from cache until it's below the target size.
        
        Args:
            target_size_mb: Target cache size in MB
        """
        # Simple LRU-like eviction: remove oldest items first
        # In a real implementation, you would track access times and evict least recently used
        
        # Sort cache items by feature ID (not ideal, but simple)
        sorted_items = sorted(self.cache.items())
        
        # Remove items until we're below target size
        while self._estimate_cache_size_mb() > target_size_mb and sorted_items:
            feature_id, _ = sorted_items.pop(0)
            if feature_id in self.cache:
                del self.cache[feature_id]
        
        self.logger.debug(f"Evicted items from cache, new size: {self._estimate_cache_size_mb():.2f} MB")
    
    def _upload_to_s3(self, feature_id: str, feature_type: str, local_path: str) -> bool:
        """
        Upload features to S3.
        
        Args:
            feature_id: Feature identifier
            feature_type: Feature type
            local_path: Local storage path
            
        Returns:
            True if upload was successful, False otherwise
        """
        if self.s3_service is None:
            return False
        
        try:
            # Upload feature file
            s3_key = f"features/{feature_type}/{feature_id}"
            success = self.s3_service.upload_file(local_path, s3_key)
            
            # Upload metadata file
            metadata_path = f"{local_path}.meta.json"
            if os.path.exists(metadata_path):
                metadata_key = f"{s3_key}.meta.json"
                self.s3_service.upload_file(metadata_path, metadata_key)
            
            self.logger.info(f"Uploaded features to S3: {s3_key}")
            return success
        
        except Exception as e:
            self.logger.error(f"Error uploading features to S3: {str(e)}")
            return False
    
    def _download_from_s3(self, feature_id: str) -> Optional[str]:
        """
        Download features from S3.
        
        Args:
            feature_id: Feature identifier
            
        Returns:
            Local storage path if download was successful, None otherwise
        """
        if self.s3_service is None:
            return None
        
        try:
            # Try to determine feature type from ID
            parts = feature_id.split('_', 1)
            if len(parts) > 1 and parts[0] in ['thermal', 'gas', 'environmental', 'fused']:
                feature_type = parts[0]
            else:
                # Try each feature type
                for feature_type in ['thermal', 'gas', 'environmental', 'fused']:
                    s3_key = f"features/{feature_type}/{feature_id}"
                    
                    # Check if object exists in S3
                    objects = self.s3_service.list_objects(s3_key)
                    if objects:
                        break
                else:
                    # Not found in any feature type directory
                    return None
            
            # Determine local path
            local_path = self._get_storage_path(feature_id, feature_type)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download feature file
            s3_key = f"features/{feature_type}/{feature_id}"
            success = self.s3_service.download_file(s3_key, local_path)
            
            if not success:
                return None
            
            # Download metadata file
            metadata_key = f"{s3_key}.meta.json"
            metadata_path = f"{local_path}.meta.json"
            self.s3_service.download_file(metadata_key, metadata_path)
            
            self.logger.info(f"Downloaded features from S3: {s3_key}")
            return local_path
        
        except Exception as e:
            self.logger.error(f"Error downloading features from S3: {str(e)}")
            return None
    
    def _delete_from_s3(self, feature_id: str, feature_type: Optional[str]) -> bool:
        """
        Delete features from S3.
        
        Args:
            feature_id: Feature identifier
            feature_type: Optional feature type
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if self.s3_service is None:
            return False
        
        try:
            # Determine feature type if not provided
            if feature_type is None:
                parts = feature_id.split('_', 1)
                if len(parts) > 1 and parts[0] in ['thermal', 'gas', 'environmental', 'fused']:
                    feature_type = parts[0]
                else:
                    # Try each feature type
                    for ft in ['thermal', 'gas', 'environmental', 'fused']:
                        s3_key = f"features/{ft}/{feature_id}"
                        
                        # Check if object exists in S3
                        objects = self.s3_service.list_objects(s3_key)
                        if objects:
                            feature_type = ft
                            break
                    else:
                        # Not found in any feature type directory
                        return False
            
            # Delete feature file
            s3_key = f"features/{feature_type}/{feature_id}"
            success = self.s3_service.delete_object(s3_key)
            
            # Delete metadata file
            metadata_key = f"{s3_key}.meta.json"
            self.s3_service.delete_object(metadata_key)
            
            self.logger.info(f"Deleted features from S3: {s3_key}")
            return success
        
        except Exception as e:
            self.logger.error(f"Error deleting features from S3: {str(e)}")
            return False


# Add this import at the top of the file
import sys