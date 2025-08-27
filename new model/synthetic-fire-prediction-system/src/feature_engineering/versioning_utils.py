"""
Utility functions for the feature versioning system.

This module provides utility functions for the feature versioning system,
including version pruning, comparison, and visualization.
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
# AWS integration (optional)
try:
    from ..aws.s3.service import S3ServiceImpl
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    S3ServiceImpl = None


def prune_old_versions(version_graph: nx.DiGraph,
                      storage: FeatureStorageSystem,
                      s3_service: Optional[S3ServiceImpl],
                      versioning_dir: str,
                      max_versions_per_type: int = 10) -> int:
    """
    Prune old versions to save storage space.
    
    Args:
        version_graph: Version graph
        storage: Feature storage system
        s3_service: Optional S3 service
        versioning_dir: Directory for version metadata
        max_versions_per_type: Maximum versions to keep per feature type
        
    Returns:
        Number of versions pruned
    """
    logger = logging.getLogger(__name__)
    pruned_count = 0
    
    # Group versions by feature type
    feature_types = {}
    
    for node_id, data in version_graph.nodes(data=True):
        feature_type = data.get('feature_type')
        if feature_type:
            if feature_type not in feature_types:
                feature_types[feature_type] = []
            
            creation_time = data.get('creation_time', '')
            feature_types[feature_type].append((node_id, creation_time))
    
    # Prune old versions for each feature type
    for feature_type, versions in feature_types.items():
        # Sort by creation time (newest first)
        sorted_versions = sorted(versions, key=lambda x: x[1], reverse=True)
        
        # Keep only the newest max_versions_per_type
        if len(sorted_versions) > max_versions_per_type:
            versions_to_prune = sorted_versions[max_versions_per_type:]
            
            for version_id, _ in versions_to_prune:
                # Get feature ID for this version
                feature_id = None
                if version_id in version_graph:
                    feature_id = version_graph.nodes[version_id].get('feature_id')
                
                # Delete version metadata
                metadata_path = os.path.join(versioning_dir, 'metadata', f"{version_id}.json")
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                # Delete from S3 if AWS integration is enabled
                if s3_service is not None:
                    s3_key = f"feature_versions/metadata/{version_id}.json"
                    s3_service.delete_object(s3_key)
                
                # Remove from graph
                if version_id in version_graph:
                    version_graph.remove_node(version_id)
                
                # Delete feature data if it's not referenced by other versions
                if feature_id:
                    referenced = False
                    for _, data in version_graph.nodes(data=True):
                        if data.get('feature_id') == feature_id:
                            referenced = True
                            break
                    
                    if not referenced:
                        # Delete feature data
                        storage.delete_features(feature_id)
                
                pruned_count += 1
    
    logger.info(f"Pruned {pruned_count} old feature versions")
    return pruned_count


def find_common_ancestors(version_graph: nx.DiGraph,
                         version_id1: str,
                         version_id2: str,
                         max_depth: int = 10) -> Set[str]:
    """
    Find common ancestors between two versions.
    
    Args:
        version_graph: Version graph
        version_id1: First version identifier
        version_id2: Second version identifier
        max_depth: Maximum depth to traverse
        
    Returns:
        Set of common ancestor version IDs
    """
    if version_id1 not in version_graph or version_id2 not in version_graph:
        return set()
    
    # Get all ancestors for each version
    ancestors1 = set()
    ancestors2 = set()
    
    # Helper function to add ancestors recursively
    def add_ancestors(version_id: str, ancestors: Set[str], depth: int) -> None:
        if depth <= 0 or version_id not in version_graph:
            return
        
        for parent_id in version_graph.predecessors(version_id):
            ancestors.add(parent_id)
            add_ancestors(parent_id, ancestors, depth - 1)
    
    # Get ancestors for both versions
    add_ancestors(version_id1, ancestors1, max_depth)
    add_ancestors(version_id2, ancestors2, max_depth)
    
    # Find common ancestors
    return ancestors1.intersection(ancestors2)


def export_version_graph(version_graph: nx.DiGraph,
                        output_path: str,
                        format: str = 'json') -> bool:
    """
    Export the version graph to a file.
    
    Args:
        version_graph: Version graph
        output_path: Output file path
        format: Export format ('json' or 'graphml')
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == 'json':
            # Convert graph to serializable format
            graph_data = {
                'nodes': {node: data for node, data in version_graph.nodes(data=True)},
                'edges': [(u, v, data) for u, v, data in version_graph.edges(data=True)]
            }
            
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
        
        elif format == 'graphml':
            # Export as GraphML
            nx.write_graphml(version_graph, output_path)
        
        else:
            logger.error(f"Unsupported export format: {format}")
            return False
        
        logger.info(f"Exported version graph to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error exporting version graph: {str(e)}")
        return False


def import_version_graph(input_path: str,
                        format: str = 'json') -> Optional[nx.DiGraph]:
    """
    Import a version graph from a file.
    
    Args:
        input_path: Input file path
        format: Import format ('json' or 'graphml')
        
    Returns:
        Imported version graph or None if import failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return None
        
        if format == 'json':
            # Import from JSON
            with open(input_path, 'r') as f:
                graph_data = json.load(f)
            
            # Create a new directed graph
            graph = nx.DiGraph()
            
            # Add nodes
            for node_id, node_data in graph_data.get('nodes', {}).items():
                graph.add_node(node_id, **node_data)
            
            # Add edges
            for edge in graph_data.get('edges', []):
                if len(edge) >= 2:
                    source, target = edge[:2]
                    edge_data = edge[2] if len(edge) > 2 else {}
                    graph.add_edge(source, target, **edge_data)
            
            return graph
        
        elif format == 'graphml':
            # Import from GraphML
            return nx.read_graphml(input_path, node_type=str)
        
        else:
            logger.error(f"Unsupported import format: {format}")
            return None
    
    except Exception as e:
        logger.error(f"Error importing version graph: {str(e)}")
        return None


def visualize_version_graph(version_graph: nx.DiGraph,
                           output_path: Optional[str] = None,
                           highlight_nodes: Optional[List[str]] = None,
                           max_nodes: int = 50) -> Optional[Figure]:
    """
    Visualize the version graph.
    
    Args:
        version_graph: Version graph
        output_path: Optional output file path
        highlight_nodes: Optional list of nodes to highlight
        max_nodes: Maximum number of nodes to include
        
    Returns:
        Matplotlib figure object if available, None otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Limit the number of nodes for visualization
        if version_graph.number_of_nodes() > max_nodes:
            logger.warning(f"Graph has {version_graph.number_of_nodes()} nodes, limiting to {max_nodes}")
            
            # Select nodes to include (prioritize highlighted nodes)
            nodes_to_include = set(highlight_nodes or [])
            
            # Add more nodes up to max_nodes
            remaining_slots = max_nodes - len(nodes_to_include)
            if remaining_slots > 0:
                other_nodes = [n for n in version_graph.nodes() if n not in nodes_to_include]
                nodes_to_include.update(other_nodes[:remaining_slots])
            
            # Create subgraph
            graph = version_graph.subgraph(nodes_to_include)
        else:
            graph = version_graph
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)
        
        # Draw nodes
        node_colors = []
        for node in graph.nodes():
            if highlight_nodes and node in highlight_nodes:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(graph, pos, node_size=500, 
                              node_color=node_colors, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
        
        # Draw labels
        labels = {}
        for node in graph.nodes():
            # Use a shorter label for readability
            feature_type = graph.nodes[node].get('feature_type', '')
            labels[node] = f"{feature_type}: {node[:8]}"
        
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
        
        plt.title(f"Feature Version Graph ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
        plt.axis('off')
        
        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved version graph visualization to {output_path}")
        
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"Error visualizing version graph: {str(e)}")
        return None


def find_version_by_metadata(version_graph: nx.DiGraph,
                           metadata_key: str,
                           metadata_value: Any) -> List[str]:
    """
    Find versions by metadata.
    
    Args:
        version_graph: Version graph
        metadata_key: Metadata key to search for
        metadata_value: Metadata value to match
        
    Returns:
        List of matching version IDs
    """
    matching_versions = []
    
    for node_id, data in version_graph.nodes(data=True):
        if metadata_key in data and data[metadata_key] == metadata_value:
            matching_versions.append(node_id)
    
    return matching_versions


def calculate_version_statistics(version_graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculate statistics about the version graph.
    
    Args:
        version_graph: Version graph
        
    Returns:
        Dictionary containing version statistics
    """
    # Count versions by feature type
    feature_types = {}
    creation_times = []
    
    for _, data in version_graph.nodes(data=True):
        feature_type = data.get('feature_type')
        if feature_type:
            feature_types[feature_type] = feature_types.get(feature_type, 0) + 1
        
        # Parse creation time
        creation_time = data.get('creation_time')
        if creation_time:
            try:
                creation_times.append(datetime.fromisoformat(creation_time))
            except ValueError:
                pass
    
    # Calculate time statistics
    time_stats = {}
    if creation_times:
        time_stats = {
            'oldest': min(creation_times).isoformat(),
            'newest': max(creation_times).isoformat(),
            'count_by_day': {}
        }
        
        # Count versions by day
        for dt in creation_times:
            day = dt.date().isoformat()
            time_stats['count_by_day'][day] = time_stats['count_by_day'].get(day, 0) + 1
    
    # Calculate graph statistics
    graph_stats = {
        'node_count': version_graph.number_of_nodes(),
        'edge_count': version_graph.number_of_edges(),
        'connected_components': nx.number_weakly_connected_components(version_graph),
        'max_depth': max(nx.shortest_path_length(version_graph, source=node).values()) 
                    if version_graph.nodes() else 0
    }
    
    return {
        'feature_types': feature_types,
        'time_stats': time_stats,
        'graph_stats': graph_stats
    }