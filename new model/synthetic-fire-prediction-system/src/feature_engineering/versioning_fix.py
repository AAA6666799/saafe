"""
Fixed implementation of the FeatureVersioningSystem class.

This module provides a fixed implementation of the FeatureVersioningSystem class
with a complete prune_old_versions method.
"""

from typing import Dict, Any, List, Optional, Union, Set, Tuple
import os
import json
import logging
import shutil
import hashlib
import time
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import threading

from .storage import FeatureStorageSystem
from .versioning_utils import (
    prune_old_versions,
    find_common_ancestors,
    export_version_graph,
    import_version_graph,
    visualize_version_graph,
    find_version_by_metadata,
    calculate_version_statistics
)
# AWS integration (optional)
try:
    from ..aws.s3.service import S3ServiceImpl
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    S3ServiceImpl = None


class FeatureVersioningSystemFixed:
    """
    System for tracking versions of extracted features.
    
    This class tracks versions of extracted features, maintains feature lineage
    information, provides feature comparison capabilities, and implements
    feature metadata management.
    
    This is a fixed implementation with a complete prune_old_versions method.
    """
    
    def __init__(self, config: Dict[str, Any], storage_system: Optional[FeatureStorageSystem] = None):
        """
        Initialize the feature versioning system.
        
        Args:
            config: Dictionary containing configuration parameters
            storage_system: Optional feature storage system instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage system if not provided
        if storage_system is None:
            self.storage = FeatureStorageSystem(config.get('storage_config', {}))
        else:
            self.storage = storage_system
        
        # Initialize AWS S3 service if AWS integration is enabled
        self.s3_service = None
        if self.config.get('aws_integration', False):
            self.s3_service = S3ServiceImpl(self.config.get('aws_config', {}))
        
        # Initialize version graph
        self.version_graph = nx.DiGraph()
        self.graph_lock = threading.RLock()
        
        # Validate configuration
        self._validate_config()
        
        # Create versioning directories
        self._create_versioning_directories()
        
        # Load existing version information
        self._load_version_graph()
    
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Set default values if not provided
        if 'versioning_dir' not in self.config:
            self.config['versioning_dir'] = 'feature_versions'
        
        if 'max_versions' not in self.config:
            self.config['max_versions'] = 10  # Default max versions per feature set
    
    def _create_versioning_directories(self) -> None:
        """
        Create necessary directories for feature versioning.
        """
        # Create main versioning directory
        os.makedirs(self.config['versioning_dir'], exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(self.config['versioning_dir'], 'metadata'), exist_ok=True)
        os.makedirs(os.path.join(self.config['versioning_dir'], 'lineage'), exist_ok=True)
        os.makedirs(os.path.join(self.config['versioning_dir'], 'comparisons'), exist_ok=True)
        
        self.logger.info(f"Created feature versioning directories in {self.config['versioning_dir']}")
    
    def _load_version_graph(self) -> None:
        """
        Load the version graph from disk.
        """
        graph_path = os.path.join(self.config['versioning_dir'], 'version_graph.json')
        
        if os.path.exists(graph_path):
            try:
                with open(graph_path, 'r') as f:
                    graph_data = json.load(f)
                
                with self.graph_lock:
                    # Create a new directed graph
                    self.version_graph = nx.DiGraph()
                    
                    # Add nodes
                    for node_id, node_data in graph_data.get('nodes', {}).items():
                        self.version_graph.add_node(node_id, **node_data)
                    
                    # Add edges
                    for edge in graph_data.get('edges', []):
                        if len(edge) >= 2:
                            source, target = edge[:2]
                            edge_data = edge[2] if len(edge) > 2 else {}
                            self.version_graph.add_edge(source, target, **edge_data)
                
                self.logger.info(f"Loaded version graph with {self.version_graph.number_of_nodes()} nodes and {self.version_graph.number_of_edges()} edges")
            
            except Exception as e:
                self.logger.error(f"Error loading version graph: {str(e)}")
                # Initialize an empty graph
                self.version_graph = nx.DiGraph()
    
    def _save_version_graph(self) -> None:
        """
        Save the version graph to disk.
        """
        graph_path = os.path.join(self.config['versioning_dir'], 'version_graph.json')
        
        try:
            with self.graph_lock:
                # Convert graph to serializable format
                graph_data = {
                    'nodes': {node: data for node, data in self.version_graph.nodes(data=True)},
                    'edges': [(u, v, data) for u, v, data in self.version_graph.edges(data=True)]
                }
            
            with open(graph_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            # Upload to S3 if AWS integration is enabled
            if self.s3_service is not None:
                s3_key = f"feature_versions/version_graph.json"
                self.s3_service.upload_file(graph_path, s3_key)
            
            self.logger.debug("Saved version graph")
        
        except Exception as e:
            self.logger.error(f"Error saving version graph: {str(e)}")
    
    def register_feature_version(self, 
                               feature_id: str, 
                               metadata: Dict[str, Any],
                               parent_ids: Optional[List[str]] = None) -> str:
        """
        Register a new feature version.
        
        Args:
            feature_id: Feature identifier from the storage system
            metadata: Metadata about the feature version
            parent_ids: Optional list of parent feature version IDs
            
        Returns:
            Version identifier
        """
        # Generate version ID
        version_id = self._generate_version_id(feature_id, metadata)
        
        # Prepare version metadata
        version_metadata = {
            'version_id': version_id,
            'feature_id': feature_id,
            'creation_time': datetime.now().isoformat(),
            'parent_versions': parent_ids or [],
            **metadata
        }
        
        # Add to version graph
        with self.graph_lock:
            self.version_graph.add_node(version_id, **version_metadata)
            
            # Add edges from parent versions
            if parent_ids:
                for parent_id in parent_ids:
                    if parent_id in self.version_graph:
                        self.version_graph.add_edge(parent_id, version_id, relationship='parent')
        
        # Save version metadata
        self._save_version_metadata(version_id, version_metadata)
        
        # Save updated version graph
        self._save_version_graph()
        
        self.logger.info(f"Registered feature version: {version_id}")
        return version_id
    
    def get_version_metadata(self, version_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific feature version.
        
        Args:
            version_id: Version identifier
            
        Returns:
            Dictionary containing version metadata
        """
        with self.graph_lock:
            if version_id in self.version_graph:
                return dict(self.version_graph.nodes[version_id])
        
        # If not in graph, try to load from file
        metadata_path = os.path.join(self.config['versioning_dir'], 'metadata', f"{version_id}.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading version metadata for {version_id}: {str(e)}")
                return {'error': str(e)}
        
        return {'error': f"Version not found: {version_id}"}
    
    def get_feature_versions(self, feature_id: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a specific feature.
        
        Args:
            feature_id: Feature identifier
            
        Returns:
            List of version metadata dictionaries
        """
        versions = []
        
        with self.graph_lock:
            for node_id, data in self.version_graph.nodes(data=True):
                if data.get('feature_id') == feature_id:
                    versions.append(dict(data))
        
        # Sort by creation time
        versions.sort(key=lambda v: v.get('creation_time', ''), reverse=True)
        
        return versions
    
    def get_version_lineage(self, version_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        Get the lineage (ancestry) of a feature version.
        
        Args:
            version_id: Version identifier
            max_depth: Maximum depth to traverse in the lineage graph
            
        Returns:
            Dictionary containing lineage information
        """
        if version_id not in self.version_graph:
            return {'error': f"Version not found: {version_id}"}
        
        lineage = {
            'version_id': version_id,
            'metadata': dict(self.version_graph.nodes[version_id]),
            'parents': [],
            'ancestors': set()
        }
        
        # Get immediate parents
        with self.graph_lock:
            for parent_id in self.version_graph.predecessors(version_id):
                parent_data = dict(self.version_graph.nodes[parent_id])
                lineage['parents'].append(parent_data)
                lineage['ancestors'].add(parent_id)
        
        # Recursively get ancestors up to max_depth
        if max_depth > 1:
            for parent in lineage['parents']:
                parent_id = parent['version_id']
                self._add_ancestors(parent_id, lineage['ancestors'], max_depth - 1)
        
        # Convert set to list for JSON serialization
        lineage['ancestors'] = list(lineage['ancestors'])
        
        return lineage
    
    def _add_ancestors(self, version_id: str, ancestors: Set[str], depth: int) -> None:
        """
        Recursively add ancestors to the set.
        
        Args:
            version_id: Version identifier
            ancestors: Set to add ancestors to
            depth: Maximum depth to traverse
        """
        if depth <= 0 or version_id not in self.version_graph:
            return
        
        with self.graph_lock:
            for parent_id in self.version_graph.predecessors(version_id):
                ancestors.add(parent_id)
                self._add_ancestors(parent_id, ancestors, depth - 1)
    
    def compare_versions(self, 
                        version_id1: str, 
                        version_id2: str,
                        save_comparison: bool = True) -> Dict[str, Any]:
        """
        Compare two feature versions.
        
        Args:
            version_id1: First version identifier
            version_id2: Second version identifier
            save_comparison: Whether to save the comparison results
            
        Returns:
            Dictionary containing comparison results
        """
        # Get metadata for both versions
        metadata1 = self.get_version_metadata(version_id1)
        metadata2 = self.get_version_metadata(version_id2)
        
        if 'error' in metadata1:
            return {'error': f"First version not found: {version_id1}"}
        
        if 'error' in metadata2:
            return {'error': f"Second version not found: {version_id2}"}
        
        # Get features for both versions
        feature_id1 = metadata1.get('feature_id')
        feature_id2 = metadata2.get('feature_id')
        
        if not feature_id1 or not feature_id2:
            return {'error': "Missing feature IDs in version metadata"}
        
        try:
            features1 = self.storage.retrieve_features(feature_id1)
            features2 = self.storage.retrieve_features(feature_id2)
        except Exception as e:
            return {'error': f"Error retrieving features: {str(e)}"}
        
        # Compare features
        comparison = {
            'version1': version_id1,
            'version2': version_id2,
            'comparison_time': datetime.now().isoformat(),
            'metadata_diff': self._compare_metadata(metadata1, metadata2),
            'feature_diff': self._compare_features(features1, features2),
            'common_lineage': self._find_common_lineage(version_id1, version_id2)
        }
        
        # Save comparison if requested
        if save_comparison:
            comparison_id = f"comparison_{version_id1}_{version_id2}"
            comparison_path = os.path.join(self.config['versioning_dir'], 'comparisons', f"{comparison_id}.json")
            
            os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
            
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            # Upload to S3 if AWS integration is enabled
            if self.s3_service is not None:
                s3_key = f"feature_versions/comparisons/{comparison_id}.json"
                self.s3_service.upload_file(comparison_path, s3_key)
        
        return comparison
    
    def _compare_metadata(self, metadata1: Dict[str, Any], metadata2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare metadata between two versions.
        
        Args:
            metadata1: Metadata for first version
            metadata2: Metadata for second version
            
        Returns:
            Dictionary containing metadata differences
        """
        diff = {
            'added_keys': [],
            'removed_keys': [],
            'changed_values': [],
            'unchanged_keys': []
        }
        
        # Find added and common keys
        for key in metadata2:
            if key not in metadata1:
                diff['added_keys'].append(key)
            else:
                if metadata1[key] == metadata2[key]:
                    diff['unchanged_keys'].append(key)
                else:
                    diff['changed_values'].append({
                        'key': key,
                        'value1': metadata1[key],
                        'value2': metadata2[key]
                    })
        
        # Find removed keys
        for key in metadata1:
            if key not in metadata2:
                diff['removed_keys'].append(key)
        
        return diff
    
    def _compare_features(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare features between two versions.
        
        Args:
            features1: Features for first version
            features2: Features for second version
            
        Returns:
            Dictionary containing feature differences
        """
        diff = {
            'added_features': [],
            'removed_features': [],
            'changed_features': [],
            'unchanged_features': [],
            'summary': {}
        }
        
        # Compare top-level keys
        all_keys = set(features1.keys()) | set(features2.keys())
        
        for key in all_keys:
            if key not in features1:
                diff['added_features'].append(key)
            elif key not in features2:
                diff['removed_features'].append(key)
            else:
                # Compare feature values
                if isinstance(features1[key], dict) and isinstance(features2[key], dict):
                    # For nested dictionaries, compare keys
                    nested_diff = self._compare_nested_features(features1[key], features2[key])
                    
                    if nested_diff['added'] or nested_diff['removed'] or nested_diff['changed']:
                        diff['changed_features'].append({
                            'feature': key,
                            'diff': nested_diff
                        })
                    else:
                        diff['unchanged_features'].append(key)
                
                elif isinstance(features1[key], (np.ndarray, pd.DataFrame)) and isinstance(features2[key], (np.ndarray, pd.DataFrame)):
                    # For arrays/dataframes, compare shapes and sample values
                    if self._compare_array_like(features1[key], features2[key]):
                        diff['unchanged_features'].append(key)
                    else:
                        diff['changed_features'].append({
                            'feature': key,
                            'diff': {
                                'shape1': getattr(features1[key], 'shape', None),
                                'shape2': getattr(features2[key], 'shape', None),
                                'type1': str(type(features1[key])),
                                'type2': str(type(features2[key]))
                            }
                        })
                
                elif features1[key] == features2[key]:
                    diff['unchanged_features'].append(key)
                else:
                    diff['changed_features'].append({
                        'feature': key,
                        'value1': features1[key],
                        'value2': features2[key]
                    })
        
        # Generate summary
        diff['summary'] = {
            'added_count': len(diff['added_features']),
            'removed_count': len(diff['removed_features']),
            'changed_count': len(diff['changed_features']),
            'unchanged_count': len(diff['unchanged_features']),
            'total_features': len(all_keys)
        }
        
        return diff
    
    def _compare_nested_features(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare nested feature dictionaries.
        
        Args:
            features1: Features for first version
            features2: Features for second version
            
        Returns:
            Dictionary containing nested feature differences
        """
        diff = {
            'added': [],
            'removed': [],
            'changed': []
        }
        
        # Find added and changed keys
        for key in features2:
            if key not in features1:
                diff['added'].append(key)
            elif features1[key] != features2[key]:
                diff['changed'].append(key)
        
        # Find removed keys
        for key in features1:
            if key not in features2:
                diff['removed'].append(key)
        
        return diff
    
    def _compare_array_like(self, arr1: Union[np.ndarray, pd.DataFrame], arr2: Union[np.ndarray, pd.DataFrame]) -> bool:
        """
        Compare array-like objects (numpy arrays or pandas DataFrames).
        
        Args:
            arr1: First array-like object
            arr2: Second array-like object
            
        Returns:
            True if objects are considered equal, False otherwise
        """
        # Check if shapes match
        if hasattr(arr1, 'shape') and hasattr(arr2, 'shape'):
            if arr1.shape != arr2.shape:
                return False
        
        # For DataFrames, check column names
        if isinstance(arr1, pd.DataFrame) and isinstance(arr2, pd.DataFrame):
            if not arr1.columns.equals(arr2.columns):
                return False
            
            # Check a sample of values
            sample_size = min(5, len(arr1))
            if sample_size > 0:
                sample_indices = np.random.choice(len(arr1), sample_size, replace=False)
                for idx in sample_indices:
                    if not arr1.iloc[idx].equals(arr2.iloc[idx]):
                        return False
            
            return True
        
        # For numpy arrays, check a sample of values
        elif isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
            # Check if arrays are too large for full comparison
            if arr1.size > 1000:
                # Compare shapes and a sample of values
                if arr1.shape != arr2.shape:
                    return False
                
                # Compare a random sample of elements
                sample_size = min(100, arr1.size)
                flat_indices = np.random.choice(arr1.size, sample_size, replace=False)
                
                flat_arr1 = arr1.flatten()
                flat_arr2 = arr2.flatten()
                
                for idx in flat_indices:
                    if flat_arr1[idx] != flat_arr2[idx]:
                        return False
                
                return True
            else:
                # For smaller arrays, compare all values
                return np.array_equal(arr1, arr2)
        
        # Default comparison
        return arr1 == arr2
    
    def _find_common_lineage(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        Find common ancestors in the lineage of two versions.
        
        Args:
            version_id1: First version identifier
            version_id2: Second version identifier
            
        Returns:
            Dictionary containing common lineage information
        """
        if version_id1 not in self.version_graph or version_id2 not in self.version_graph:
            return {'common_ancestors': []}
        
        # Get all ancestors for each version
        ancestors1 = set()
        ancestors2 = set()
        
        self._add_ancestors(version_id1, ancestors1, 10)  # Max depth of 10
        self._add_ancestors(version_id2, ancestors2, 10)
        
        # Find common ancestors
        common_ancestors = ancestors1.intersection(ancestors2)
        
        # Get metadata for common ancestors
        common_ancestor_data = []
        for ancestor_id in common_ancestors:
            if ancestor_id in self.version_graph:
                ancestor_data = dict(self.version_graph.nodes[ancestor_id])
                common_ancestor_data.append(ancestor_data)
        
        return {
            'common_ancestors': common_ancestor_data,
            'ancestor_count1': len(ancestors1),
            'ancestor_count2': len(ancestors2),
            'common_count': len(common_ancestors)
        }
    
    def visualize_lineage(self, 
                         version_id: str, 
                         output_path: Optional[str] = None,
                         max_depth: int = 3) -> Optional[Figure]:
        """
        Visualize the lineage of a feature version.
        
        Args:
            version_id: Version identifier
            output_path: Optional path to save the visualization
            max_depth: Maximum depth to include in the visualization
            
        Returns:
            Matplotlib figure object if available, None otherwise
        """
        if version_id not in self.version_graph:
            self.logger.warning(f"Version not found for visualization: {version_id}")
            return None
        
        try:
            # Extract subgraph for visualization
            nodes_to_include = {version_id}
            self._add_ancestors(version_id, nodes_to_include, max_depth)
            
            with self.graph_lock:
                subgraph = self.version_graph.subgraph(nodes_to_include)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(subgraph)
            
            # Draw nodes
            nx.draw_networkx_nodes(subgraph, pos, node_size=500, 
                                  node_color='lightblue', alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5)
            
            # Draw labels
            labels = {}
            for node in subgraph.nodes():
                # Use a shorter label for readability
                if node == version_id:
                    labels[node] = f"Current: {node[:8]}"
                else:
                    feature_type = subgraph.nodes[node].get('feature_type', '')
                    labels[node] = f"{feature_type}: {node[:8]}"
            
            nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)
            
            plt.title(f"Feature Version Lineage for {version_id[:8]}")
            plt.axis('off')
            
            # Save if output path is provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                
                # Upload to S3 if AWS integration is enabled
                if self.s3_service is not None:
                    s3_key = f"feature_versions/visualizations/lineage_{version_id}.png"
                    self.s3_service.upload_file(output_path, s3_key)
                
                self.logger.info(f"Saved lineage visualization to {output_path}")
            
            return plt.gcf()
        
        except Exception as e:
            self.logger.error(f"Error visualizing lineage: {str(e)}")
            return None
    
    def _generate_version_id(self, feature_id: str, metadata: Dict[str, Any]) -> str:
        """
        Generate a unique identifier for a feature version.
        
        Args:
            feature_id: Feature identifier
            metadata: Version metadata
            
        Returns:
            Version identifier
        """
        # Create a hash of the feature ID and metadata
        version_hash = hashlib.md5()
        
        # Add feature ID to the hash
        version_hash.update(feature_id.encode('utf-8'))
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().isoformat()
        version_hash.update(timestamp.encode('utf-8'))
        
        # Add key metadata fields to the hash
        for key in ['feature_type', 'extractor_version', 'dataset_id']:
            if key in metadata:
                version_hash.update(str(metadata[key]).encode('utf-8'))
        
        # Generate ID
        return f"v_{version_hash.hexdigest()}"
    
    def _save_version_metadata(self, version_id: str, metadata: Dict[str, Any]) -> None:
        """
        Save version metadata to disk.
        
        Args:
            version_id: Version identifier
            metadata: Version metadata
        """
        metadata_path = os.path.join(self.config['versioning_dir'], 'metadata', f"{version_id}.json")
        
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            s3_key = f"feature_versions/metadata/{version_id}.json"
            self.s3_service.upload_file(metadata_path, s3_key)
    
    def get_latest_version(self, feature_type: Optional[str] = None) -> Optional[str]:
        """
        Get the latest version for a feature type.
        
        Args:
            feature_type: Optional feature type to filter by
            
        Returns:
            Latest version identifier or None if not found
        """
        latest_version = None
        latest_time = None
        
        with self.graph_lock:
            for node_id, data in self.version_graph.nodes(data=True):
                # Filter by feature type if provided
                if feature_type is not None and data.get('feature_type') != feature_type:
                    continue
                
                # Check creation time
                creation_time = data.get('creation_time')
                if creation_time:
                    try:
                        time = datetime.fromisoformat(creation_time)
                        if latest_time is None or time > latest_time:
                            latest_time = time
                            latest_version = node_id
                    except ValueError:
                        pass
        
        return latest_version
    
    def prune_old_versions(self, max_versions_per_type: Optional[int] = None) -> int:
        """
        Prune old versions to save storage space.
        
        Args:
            max_versions_per_type: Maximum versions to keep per feature type
            
        Returns:
            Number of versions pruned
        """
        # Use utility function to prune old versions
        with self.graph_lock:
            pruned_count = prune_old_versions(
                self.version_graph,
                self.storage,
                self.s3_service,
                self.config['versioning_dir'],
                max_versions_per_type or self.config.get('max_versions', 10)
            )
            
            # Save updated version graph
            if pruned_count > 0:
                self._save_version_graph()
        
        return pruned_count
    
    def export_graph(self, output_path: str, format: str = 'json') -> bool:
        """
        Export the version graph to a file.
        
        Args:
            output_path: Output file path
            format: Export format ('json' or 'graphml')
            
        Returns:
            True if successful, False otherwise
        """
        with self.graph_lock:
            return export_version_graph(self.version_graph, output_path, format)
    
    def import_graph(self, input_path: str, format: str = 'json', merge: bool = False) -> bool:
        """
        Import a version graph from a file.
        
        Args:
            input_path: Input file path
            format: Import format ('json' or 'graphml')
            merge: Whether to merge with existing graph
            
        Returns:
            True if successful, False otherwise
        """
        imported_graph = import_version_graph(input_path, format)
        
        if imported_graph is None:
            return False
        
        with self.graph_lock:
            if merge:
                # Merge with existing graph
                self.version_graph = nx.compose(self.version_graph, imported_graph)
            else:
                # Replace existing graph
                self.version_graph = imported_graph
            
            # Save updated graph
            self._save_version_graph()
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the version graph.
        
        Returns:
            Dictionary containing version statistics
        """
        with self.graph_lock:
            return calculate_version_statistics(self.version_graph)